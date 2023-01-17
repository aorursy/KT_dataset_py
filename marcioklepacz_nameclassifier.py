from collections import Counter
import string
import numpy as np
import torch

def encode_word(word, max_chars=13, alphabet=string.ascii_lowercase):   
    alphabet_char = [char for char in string.ascii_lowercase]    
    batch_size = max_chars
    nb_digits = len(alphabet_char)

    y = torch.LongTensor(batch_size,1).random_() % nb_digits

    w_pos = [alphabet_char.index(char) for char in word]
    w_pos = torch.LongTensor(w_pos)
    w_pos = w_pos.view(-1, 1)

    y_onehot = torch.FloatTensor(batch_size, nb_digits)

    y_onehot.zero_()
    y_onehot.scatter_(1, w_pos, 1)
    
    return y_onehot


import pandas as pd 

names_masc = pd.read_csv('../input/ibge-mas-10000.csv')
names_fem = pd.read_csv('../input/ibge-fem-10000.csv')
names = pd.concat([names_fem, names_masc], ignore_index=True)
name_and_sex = names[['nome', 'sexo']] 

name_lens = Counter([len(x) for x in name_and_sex['nome']])
max_chars = max(name_lens)
shuffled = name_and_sex.sample(frac=1)
print(shuffled[:30]['nome'].values)
print(shuffled[:30]['sexo'].values)

names_encoded = [encode_word(name.lower()).view(1, 13, -1) for name in name_and_sex['nome']]
names_encoded = torch.cat(names_encoded, dim=0)

labels_encoded = [0 if sex == 'F' else 1 for sex in name_and_sex['sexo']]
max_len = len(names_encoded)

train_i = int(max_len * 0.8)
test_i = int(max_len * 0.2)

test_data_encoded, train_data_encoded, _ = np.split(names_encoded, [test_i, max_len])
test_label_encoded, train_label_encoded, _ = np.split(labels_encoded, [test_i, max_len])

test_data_encoded, valid_data_encoded = np.split(test_data_encoded, 2)
test_label_encoded, valid_label_encoded = np.split(test_label_encoded, 2)

train_labels_tensor = torch.LongTensor(train_label_encoded).view(-1, 1)
test_labels_tensor = torch.LongTensor(test_label_encoded).view(-1, 1)
valid_labels_tensor = torch.LongTensor(valid_label_encoded).view(-1, 1)

train_label_encoded.shape
import torch

from torch.utils.data import TensorDataset, DataLoader

test_data = TensorDataset(test_data_encoded, test_labels_tensor)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

train_data = TensorDataset(train_data_encoded, train_labels_tensor)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

valid_data = TensorDataset(valid_data_encoded, valid_labels_tensor)
valid_loader = DataLoader(valid_data, batch_size=100, shuffle=True)

from torch import nn, optim
import torch.nn.functional as F

class NameClassfier(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(13 * 26, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
        
        

train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

net = NameClassfier()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.02)

epochs = 110

counter = 0
print_every = 100

losses = []
al_val_loss = []

if(train_on_gpu):
    net.cuda()
    
net.train()

for e in range(epochs):

    # batch loop
    for inputs, labels in train_loader:
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
            
        counter += 1

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output = net(inputs.float())

        # calculate the loss and perform backprop
        
        loss = criterion(output, labels.float())
                
        loss.backward()
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            losses.append(loss.item())
                        
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    
                output = net(inputs.float())
                val_loss = criterion(output, labels.float())

                val_losses.append(val_loss.item())

            net.train()
            al_val_loss.append(np.mean(val_losses))
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


import matplotlib.pyplot as plt
plt.plot(losses)
plt.plot(al_val_loss)
plt.show()

net.eval()
test_losses = []
num_correct = 0

for inputs, labels in test_loader:
    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    output = net(inputs.float())

    test_loss = criterion(output, labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(torch.sigmoid(output)) # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
net.eval()

names = """
Marcio
""".split()

for name in names:    
    encode = encode_word(name.lower()).view(1, -1)
    out = net(encode)
    gender = 'masc' if torch.sigmoid(out) > 0.5 else 'fem'
    print(f"{name} -> {gender} {torch.sigmoid(out).item()}%")
