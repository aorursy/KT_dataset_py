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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F



from sklearn.model_selection import train_test_split

train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sample=pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
X=train.drop(['label'],axis=1)

Y=train['label']

X=torch.from_numpy(np.array(X).reshape(-1,1,28,28)/255)

Y=torch.from_numpy(np.array(Y))

xtrain,xval,ytrain,yval=train_test_split(X,Y,test_size=0.2,random_state=2019)



train_batch_size=512

val_batch_size=512

train_dataset=torch.utils.data.TensorDataset(xtrain.type(torch.FloatTensor),ytrain.type(torch.LongTensor))

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)

val_dataset=torch.utils.data.TensorDataset(xval.type(torch.FloatTensor),yval.type(torch.LongTensor))

val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=val_batch_size,shuffle=False)
class Flatten(nn.Module):

    def forward(self,input):

        return input.view(-1,64*3*3)

model=nn.Sequential(

    nn.Conv2d(1,16,3,padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Conv2d(16,32,3,padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Conv2d(32,64,3,padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(64*3*3,120),

    nn.ReLU(),

    nn.Linear(120,64),

    nn.ReLU(),

    nn.Linear(64,10)                    

)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
def validation(model,dataload,criterion):

    running_loss = 0.0

    running_corrects = 0

    los=0

    for images,labels in dataload:

        #images=images.view(images.shape[0],784)

        images,labels=images.to(device),labels.to(device)

        output=model(images)



        test_loss=criterion(output,labels)

        los+=test_loss.item()

        preds = torch.argmax(output,1)

        running_loss += test_loss.item() * images.size(0)

        running_corrects += torch.sum(preds == labels)

    #print(output.shape)

    #epoch_loss = running_loss / len(xval)

    #print(type(running_loss))

    #print(epoch_loss),print(los/len(dataload))        

    epoch_loss = running_loss / len(xval)

    epoch_acc = running_corrects.double() / len(xval)

    

    print('{} Loss: {:.4f} Acc: {:.4f}'.format("valid", epoch_loss, epoch_acc))

    return epoch_acc,epoch_loss
#Learning_rate=0.001

loss_criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters())

epoch=200

var_loss=np.inf

train_losses,val_losses=[],[]

import time,copy

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())

best_acc = 0.0

model=model.to(device)

for e in range(epoch):

    running_loss = 0.0

    running_corrects = 0

    print('Epoch {}/{}:'.format(e+1, epoch))

    print('-' * 10)

    for images,labels in train_loader:

        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()

        #images=images.view(images.shape[0],784)

        output=model(images)

        loss=loss_criterion(output,labels)

        preds = torch.argmax(output,1)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(xtrain)

    train_losses.append(epoch_loss)

    epoch_acc = running_corrects.double() / len(xtrain)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format("Training", epoch_loss, epoch_acc))

    

    with torch.no_grad():

            model.eval()

            accuracy,epoch_loss=validation(model,val_loader,loss_criterion)

            val_losses.append(epoch_loss)

            if best_acc<accuracy:

                best_acc = accuracy

                torch.save(model.state_dict(),'model.pt')

            else:

                print("best acc is {:.4f}".format(best_acc))

    model.train()    

time_elapsed = time.time() - since

print('Training complete in {:.0f}m {:.0f}s'.format(

    time_elapsed // 60, time_elapsed % 60))

print('Best val Acc: {:4f}'.format(best_acc))

        
plt.plot(train_losses,label='Train Loss')

plt.plot(val_losses,label='Validation Loss') 

plt.legend()
model.load_state_dict(torch.load('model.pt'))

test_=np.array(test).reshape(-1,1,28,28)/255

testT=torch.tensor(test_)

test_dataset=testT.type(torch.FloatTensor)

test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=val_batch_size,shuffle=False)

model=model.to(device)

with torch.no_grad():

        model.eval()

        label=[]

        for images in test_loader:

            images=images.to(device)

            #images=images.reshape(images[0],784)

            output=model(images)

            labels=torch.argmax(output,1)

            label+=labels

        

        sub={'ImageId':[i+1 for i in range(test.shape[0])],'Label':np.array(label)}

        submission=pd.DataFrame(sub)

        submission.to_csv('submission.csv',index=False)