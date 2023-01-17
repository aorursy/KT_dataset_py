# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
train_data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(train_data.shape)
print(test_data.shape)
x=train_data.drop('label',axis=1)
y=np.array(train_data['label'])
x.shape,y.shape
x=x/255
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15)
torch_x_train=torch.from_numpy(x_train.values).type(torch.FloatTensor)
torch_y_train=torch.from_numpy(y_train).type(torch.LongTensor)
torch_x_test=torch.from_numpy(x_test.values).type(torch.FloatTensor)
torch_y_test=torch.from_numpy(y_test).type(torch.LongTensor)
torch_train=torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
torch_test=torch.utils.data.TensorDataset(torch_x_test,torch_y_test)
train_loader=torch.utils.data.DataLoader(torch_train,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(torch_train,batch_size=64,shuffle=True)
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 60)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(60, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def model(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        #x = self.dropout(F.relu(self.fc2(x)))
        #x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
        
a=Network()
a

criterion=nn.NLLLoss()
optimiser=optim.Adam(a.parameters(),lr=0.001)
epochs=10
for i in range(epochs):
    running_loss=0
    for images,labels in train_loader:
        images=images.view(images.shape[0],-1)
        optimiser.zero_grad()
        output=a.model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimiser.step()
        running_loss+=loss.item()*images.shape[0]
    print(f'epoch:{i},training_loss:{(running_loss/len(train_loader.dataset))}')
epochs=10
trainloss,testloss=[],[]
for i in range(epochs):
    running_loss=0
    for images,labels in train_loader:
        images=images.view(images.shape[0],-1)
        optimiser.zero_grad()
        output=a.model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimiser.step()
        running_loss+=loss.item()*images.shape[0]
    else:
        test_loss=0
        accuracy=0
        with torch.no_grad():
            a.eval()
            for images,labels in test_loader:
                images=images.view(images.shape[0],-1)
                log_ps=a.model(images)
                test_loss+=criterion(log_ps,labels)
                ps=torch.exp(log_ps)
                top_p,top_c=ps.topk(1,dim=1)
                equals=top_c==labels.view(top_c.shape)
                accuracy+=torch.mean(equals.type(torch.FloatTensor))
            a.train()
            trainloss.append(running_loss/len(train_loader.dataset))
            testloss.append(test_loss/len(test_loader.dataset))
            print(f'epoch:{i}--train_loss{trainloss[i]}--test_loss{testloss[i]}--accuracy{accuracy/len(test_loader)}')
print('our model : ',a)
print(a.parameters)
print('model state dict:',a.state_dict().keys())
torch.save(a.state_dict(),'checkpoint')
state_dict=torch.load('checkpoint')
state_dict.keys()
a.load_state_dict(state_dict)
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [256,128,64],
              'state_dict': a.state_dict()}

torch.save(checkpoint, 'checkpoint')
test_data=test_data.loc[:,test_data.columns!='label'].values
test_dataset=torch.from_numpy(test_data).type(torch.FloatTensor)/255
print(test_dataset.shape)
new_test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=100,shuffle=True)
count=0
for images in new_test_loader:
    count=count+1
print(count)
results=[]
with torch.no_grad():
    a.eval()
    for images in new_test_loader:
        images=images.view(images.shape[0],-1)
        log_ps=a.model(images)
        #print(log_ps.shape)
        ps=torch.exp(ps)
        top_p,top_c=log_ps.topk(1,dim=1)
        #print(len(top_c))
        results += top_c.numpy().tolist()
        
results
predictions = np.array(results).flatten()
print(predictions[:5])
print(predictions.shape)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("my_submissions.csv", index=False, header=True)
import os
os.getcwd()