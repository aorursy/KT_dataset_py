import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import torch

from torchvision import datasets

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from collections import Counter

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
x_train = pd.read_csv("/kaggle/input/competicao-dsa-machine-learning-sep-2019/X_treino.csv")
y_train = pd.read_csv("/kaggle/input/competicao-dsa-machine-learning-sep-2019/y_treino.csv")
x_test = pd.read_csv("/kaggle/input/competicao-dsa-machine-learning-sep-2019/X_teste.csv")
x_train.head(10)
y_train.head(10)
x_test.head(10)
x_train.describe()
x_test.describe()
# Análise dos dados

columns=x_train.columns[3:]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in zip(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    x_train[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
x_train.plot(kind= 'box' , subplots=True, layout=(12,1),figsize=(5,50))
# normalizando os dados de treino e teste

# Aplicando a mesma escala nos dados

# Padronizando os dados (0 para a média, 1 para o desvio padrão)

scaler = StandardScaler()

scaler.fit(x_train[x_train.columns[3:]])

X_train_norm = scaler.transform(x_train[x_train.columns[3:]])

X_test_norm = scaler.transform(x_test[x_test.columns[3:]])
x_train[x_train.columns[3:]] = X_train_norm

x_test[x_test.columns[3:]] = X_test_norm
# seleciona todas as linhas com os mesmos series_ids  

def select_serie_ids(df, sid):

    df2 = df[df['series_id'] == sid]

    return df2[df2.columns[3:]].values.reshape((-1))



def select_serie_ids2(df, sid):

    df2 = df[df['series_id'] == sid]

    return df2[df2.columns[3:]].values.reshape((1,128,10))
# cria lista de inputs do modelo

def get_X(df):

    X = []

    for i in range(0, df['series_id'].max()+1):

        X.append(select_serie_ids(df, i))

    return X



def get_X2(df):

    X = []

    for i in range(0, df['series_id'].max()+1):

        X.append(select_serie_ids2(df, i))

    return X
X = get_X(x_train)

    

X_test = get_X2(x_test)
# Cria um label encoder object

le = LabelEncoder()

suf="_le"



# Iteracao para cada coluna do dataset de treino

for col in y_train.columns:

    if y_train[col].dtype == 'object':

        le.fit_transform(y_train[col].astype(str))

        y_train[col+suf] = le.transform(y_train[col])  
y = y_train['surface_le'].values
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)

X_SM, y_SM = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_SM))
X_SM = X_SM.reshape((-1, 1, 128, 10))
X_train, X_valid, Y_train, Y_valid = train_test_split(X_SM, y_SM, test_size=0.25, random_state=42)
batch_size = 35



train_target = torch.tensor(Y_train.astype(np.int64))

train = torch.tensor(X_train) 

train_tensor = torch.utils.data.TensorDataset(train, train_target) 

trainloader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)



test_target = torch.tensor(Y_valid.astype(np.int64))

test = torch.tensor(X_valid) 

test_tensor = torch.utils.data.TensorDataset(test, test_target) 

testloader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = batch_size, shuffle = False)



valid_target = torch.tensor(np.zeros((len(X_test))))

valid = torch.tensor(X_test) 

valid_tensor = torch.utils.data.TensorDataset(valid, valid_target) 

validloader = torch.utils.data.DataLoader(dataset = valid_tensor, batch_size = 1, shuffle = False)
import torch.nn as nn

import torch.nn.functional as F



# define the CNN architecture

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # convolutional layer (sees 128x10x3 image tensor)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

        # convolutional layer (sees 64x5x16 tensor)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # convolutional layer (sees 32x2x32 tensor)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # max pooling layer

        self.pool = nn.MaxPool2d(2, 2)

        # linear layer (128 * 16 * 1 -> 1000)

        self.fc1 = nn.Linear(128 * 16 * 1, 1000)

        # linear layer (1000 -> 9)

        self.fc2 = nn.Linear(1000, 9)

        # dropout layer (p=0.25)

        self.dropout = nn.Dropout(0.25)



    def forward(self, x):

        # add sequence of convolutional and max pooling layers

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input

        x = x.view(-1, 128 * 16 * 1)

        # add dropout layer

        x = self.dropout(x)

        # add 1st hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function

        x = self.fc2(x)

        return x



# create a complete CNN

model = Net().double()

print(model)



# move tensors to GPU if CUDA is available

if train_on_gpu:

    model.cuda()
import torch.optim as optim



# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# specify optimizer

optimizer = optim.Adam(model.parameters())
# number of epochs to train the model

n_epochs = 100



valid_loss_min = np.Inf # track change in validation loss



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for data, target in trainloader:

        if data.shape[0] != batch_size: continue

        #print(data.shape)

        #print(target.shape)

        data = data.double()

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item()*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval()

    for data, target in testloader:

        if data.shape[0] != batch_size: continue

        data = data.double()

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # update average validation loss 

        valid_loss += loss.item()*data.size(0)

    

    # calculate average losses

    train_loss = train_loss/len(trainloader.dataset)

    valid_loss = valid_loss/len(testloader.dataset)

        

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {} \tValidation Loss: {}'.format(

        epoch, train_loss, valid_loss))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({} --> {}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'model_PyTorch_Smote.pt')

        valid_loss_min = valid_loss
model.load_state_dict(torch.load('model_PyTorch_Smote.pt'))
classes = ['concrete', 'soft_pvc', 'wood', 'tiled', 'fine_concrete', 'hard_tiles_large_space', 'soft_tiles', 'carpet', 

           'hard_tiles']
# track test loss

test_loss = 0.0

class_correct = list(0. for i in range(9))

class_total = list(0. for i in range(9))



model.eval()

# iterate over test data

for data, target in testloader:

    if data.shape[0] != batch_size: continue

    # move tensors to GPU if CUDA is available

    data = data.double()

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class

    for i in range(batch_size):

        #print(target)

        #print(target.data)

        #print(target.data.shape)

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# average test loss

test_loss = test_loss/len(testloader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(9):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
predictions = []



model.eval()

# iterate over test data

for data, target in validloader:

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    _, preds_tensor = torch.max(output, 1)

    preds = preds_tensor.numpy()[0]

    predictions.append(preds)
surface_pred = le.inverse_transform(predictions)

#Gerando Arquivo de Submissao

df = pd.DataFrame({

    "series_id": [x for x in range(len(surface_pred))], 

    "surface": surface_pred

})

#df.to_csv('/kaggle/input/competicao-dsa-machine-learning-sep-2019/sample_submission_v4_pytorch_smote_.csv', sep=',', index=False)
df.head(10)
df.groupby('surface').size().plot(kind='bar', figsize=(6,6))

plt.title('Classes de Superficies')

plt.xlabel('Classes')

plt.ylabel('Frequencia')

plt.show()