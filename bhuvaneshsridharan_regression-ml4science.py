import torch
from torch import nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/esol.csv')


df = df[['measured log solubility in mols per litre','smiles']]
train_df, validate_df, test_df = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])
elements = {'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S','H','Si'}
vocab = set()
for indx,val in enumerate(list(df['smiles'].values)):
    temp_smile = re.sub(' ','',val)   ##remove spaces from the smile string
    for element in elements:
        search_obj = re.search(element,temp_smile)
        if search_obj:
            temp_smile = re.sub(element,'',temp_smile)
            vocab.add(element)
        search_obj = re.search(element.lower(),temp_smile)
        if search_obj:
            temp_smile = re.sub(element.lower(),'',temp_smile)
            vocab.add(element.lower())
    
    for i in temp_smile:
        vocab.add(i)
## convert vocab from set to list for indexing
vocab = list(vocab)
def smiles2vec(smile):
    indx = 0
    vec = []
    smile = smile.strip()
    while indx < len(smile):
        if (indx is not len(smile) - 1) and (smile[indx] + smile[indx+1]) in vocab:
            vec.append(vocab.index(smile[indx] + smile[indx+1]))
            indx += 2
            
        elif (smile[indx]) in vocab:
            vec.append(vocab.index(smile[indx]))
            indx += 1
            
    return vec
INPUT_SIZE = 1      # rnn input size
HIDDEN_SIZE = 120      # rnn hidden size
LR = 1e-2         # learning rate
BATCH_SIZE = 32
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,     # rnn hidden unit
            num_layers=1,                # number of rnn layer
            batch_first = True,
        )
        self.out = nn.Linear(HIDDEN_SIZE, 60)
        self.final = nn.Linear(60, 1)

    def forward(self, x, time_step):
        h_state = torch.zeros(1,1,HIDDEN_SIZE).float()
        r_out, h_state = self.rnn(x.float(), h_state)
        r_out = torch.mean(r_out,dim=1)
        r_out = self.out(r_out)
        r_out = F.relu(r_out)
        prediction = self.final(r_out)
        return prediction
model = Model()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=5)
def train(epoch,train_set):
    model.train()
    sample_index = 0
    epoch_train_loss=[]
    random_indexes=np.random.permutation(len(train_set))
    MAXITER = len(train_set)/BATCH_SIZE
    for batch_no in range(0,round(MAXITER)):
        optimizer.zero_grad()
        train_loss = torch.zeros(1, 1)
        
        for iter_no in range(0, BATCH_SIZE):
            sample_index = sample_index%len(train_set)
            index=random_indexes[sample_index]
            data = train_set[index]
    
            smile = data[1]
            smile_vec = smiles2vec(smile)
            smile_vec = torch.FloatTensor(smile_vec)
            smile_vec = smile_vec.unsqueeze(0)
            smile_vec = smile_vec.unsqueeze(2)
            y_hat = model(smile_vec,smile_vec.shape[1])
            
            y = torch.FloatTensor([data[0]])
            error = ((y_hat - y)**2 / BATCH_SIZE)
            train_loss = train_loss + error
            
            sample_index += 1

        epoch_train_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
    
    return np.mean(epoch_train_loss)
def validation():
    model.eval()
    val_loss = torch.zeros(1, 1)
    for index,data in enumerate(validate_df.values):
        smile = data[1]
        smile_vec = smiles2vec(smile)
        smile_vec = torch.FloatTensor(smile_vec)
        smile_vec = smile_vec.unsqueeze(0)
        smile_vec = smile_vec.unsqueeze(2)
        y_hat = model(smile_vec,smile_vec.shape[1])

        y = torch.FloatTensor([data[0]])
        error = (y_hat - y)**2
        val_loss = val_loss + error

    return val_loss.item()/len(validate_df)
def test():
    model.eval()
    test_loss = torch.zeros(1, 1)
    for index,data in enumerate(test_df.values):
        smile = data[1]
        smile_vec = smiles2vec(smile)
        smile_vec = torch.FloatTensor(smile_vec)
        smile_vec = smile_vec.unsqueeze(0)
        smile_vec = smile_vec.unsqueeze(2)
        y_hat = model(smile_vec,smile_vec.shape[1])

        y = torch.FloatTensor([data[0]])
        error = (y_hat - y)**2
        test_loss = test_loss + error

    return test_loss.item()/len(test_df)
train_loss_arr = []
validation_loss_arr = []
for epoch in range(100):
    train_loss = train(epoch,train_df.values)
    val_loss = validation()
    train_loss_arr.append(train_loss)
    validation_loss_arr.append(val_loss)
    print("epoch_no: ",epoch ,"training loss: ",train_loss,"validation loss: ",val_loss)
    scheduler.step(val_loss)
print("Test loss: ",test())
plt.plot(train_loss_arr,'b',label='Training loss')
plt.plot(validation_loss_arr,'r',label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.plot()
plt.show()