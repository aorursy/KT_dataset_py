import numpy as np

import pandas as pd



# Read in files with pandas, grouping by the series id



def read_file(filename):

    df_csv = pd.read_csv(filename)

    X_out = np.zeros((len(df_csv)//128,10,128))

    for group,x in df_csv.groupby('series_id'):

        X_out[group] = x.values[:,3:].T

    return X_out



X_data = read_file('../input/X_train.csv')



# Define class - id mapping



classes = ['fine_concrete', 'concrete', 'soft_tiles', 'tiled', 'soft_pvc',

           'hard_tiles_large_space', 'carpet', 'hard_tiles', 'wood']

class_dict = {x:i for i,x in enumerate(classes)}



# Import labels



y_data_csv = pd.read_csv('../input/y_train.csv')

y_data = np.array([class_dict[x] for x in y_data_csv.values[:,2]])
import torch

import torch.utils.data

from sklearn.model_selection import train_test_split



# Create pytorch dataloaders after doing train_test_split



X_train, X_val, y_train, y_val = train_test_split(

    X_data, y_data, test_size=0.33, random_state=42

)

def get_dataset(x,y):

    return torch.utils.data.TensorDataset(torch.FloatTensor(x),torch.LongTensor(y))



train_set = get_dataset(X_train, y_train)

val_set = get_dataset(X_val,y_val)



batch_size = 32

train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)

val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size)
import torch

import torch.nn as nn



# A really simple 1d convnet:



class ConvNet(nn.Module):

    def __init__(self, hidden_size=32, in_channels=10, num_classes=9):

        super(ConvNet, self).__init__()

        

        # Map in channels to number of hidden layers, kernel size is 9, stride is 2

        self.conv = nn.Conv1d(in_channels, hidden_size, 9, stride=2)

        self.relu = nn.ReLU(inplace=True)

        

        # AdaptiveAvgPool1d will let us just specify the size of output

        # In our case we just average over the whole timeline

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        

        # We then map the output to the number of classes to finish!

        self.linear = nn.Linear(hidden_size, num_classes)



    def forward(self, x):

        # Normalize data by subtracting the mean

        x = x - torch.mean(x, -1, keepdim=True)

        

        x = self.relu(self.conv(x))

        x = self.avg_pool(x)

        return self.linear(x.view(x.size(0),-1))
# Get started! Using CEL and SGD



torch.manual_seed(1)

model = ConvNet()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=.2)
# Training loop -- only check validation every print_freq epochs since it is really fast



epochs = 100

print_freq = 10

for e in range(epochs):

    train_loss = 0.0

    train_correct = 0.0

    for x,y in train_loader:

        p = model(x)

        loss = criterion(p,y)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item() * len(y)

        _,pred = p.max(1)

        train_correct += sum(pred == y).float().item()

    

    train_loss = train_loss / len(train_set)

    train_correct = train_correct / len(train_set)

    

    if e > 0 and e % print_freq == 0:

        val_loss = 0.0

        val_correct = 0.0

        for x,y in val_loader:

            p = model(x)

            loss = criterion(p,y)



            val_loss += loss.item() * len(y)

            _,pred = p.max(1)

            val_correct += sum(pred == y).float().item()



        val_loss = val_loss / len(val_set)

        val_correct = val_correct / len(val_set)

        print("Train loss: {:.4f}, acc: {:.4f};\t Val loss: {:.4f}, acc: {:.4f}".format(

            train_loss,train_correct,val_loss,val_correct

        ))
# Run on test data



X_test = read_file('../input/X_test.csv')

y_test = ['']*len(X_test)

for i,x in enumerate(X_test):

    p = model(torch.FloatTensor(x).unsqueeze(0))

    _,pred = p.max(1)

    y_test[i] = classes[pred.item()]
# Export to CSV

df = pd.DataFrame(y_test,columns=['surface'])

df.to_csv('submission.csv',index_label='series_id')

df.head()