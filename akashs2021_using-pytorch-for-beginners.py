# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv',dtype=np.float32)# reading the csv file
data.head()#checking out the data
targets_numpy = data.label.values

features_numpy = data.loc[:,data.columns != "label"].values/255 # normalization
import torch

from torchvision import datasets,transforms

from sklearn.model_selection import train_test_split
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,

                                                                             targets_numpy,

                                                                             test_size = 0.2,

                                                                             random_state = 42)

featuresTrain = torch.FloatTensor(features_train)

targetsTrain = torch.LongTensor(targets_train) # data type is long



# create feature and targets tensor for test set.

featuresTest = torch.FloatTensor(features_test)

targetsTest = torch.LongTensor(targets_test)
print(type(featuresTest))

print(type(targetsTest))
trainset = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

testset = torch.utils.data.TensorDataset(featuresTest,targetsTest)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = False)

testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False)
import matplotlib.pyplot as plt
# visualize one of the images in data set

plt.imshow(features_numpy[10].reshape(28,28))

plt.axis("off")

plt.title(str(targets_numpy[10]))

plt.savefig('graph.png')

plt.show()
from torch import nn, optim 

import torch.nn.functional as F
#creating our classifier

class classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1=nn.Linear(784, 256)

        self.fc2=nn.Linear(256,128)

        self.fc3=nn.Linear(128,10)

        self.dropout = nn.Dropout(p=0.2)

    def forward (self,x):

        x=self.dropout(F.relu(self.fc1(x)))

        x=self.dropout(F.relu(self.fc2(x)))

        x=F.log_softmax(self.fc3(x),dim=1)

        return x

    
model=classifier()

criterion=nn.NLLLoss()

optimizer=optim.Adam(model.parameters(),lr=0.004)
train_on_gpu=torch.cuda.is_available()

if train_on_gpu:

    model.cuda()
epochs=15

train_losses, test_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        if train_on_gpu:

            images,labels=images.cuda(),labels.cuda()

        

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations

        with torch.no_grad():

            model.eval()

            for images, labels in testloader:

                if train_on_gpu:

                    images,labels=images.cuda(),labels.cuda()

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        model.train()

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(train_losses[-1]),

              "Test Loss: {:.3f}.. ".format(test_losses[-1]),

              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.legend(frameon=False)
testdata=pd.read_csv('../input/test.csv',dtype=np.float32)
testdata.head()
ssubmission=pd.read_csv('../input/sample_submission.csv',)
ssubmission.head() # Checking out how the submission has to be made
testfeatues = testdata.loc[:].values/255 # normalization
testfeatures = torch.from_numpy(testfeatues)
print(testfeatures.shape)
testlabels=torch.ones_like(testfeatures)# Creating a dummy label list so that we can make a dataset
print(testlabels.shape)
testset = torch.utils.data.TensorDataset(testfeatures,testlabels)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False)
ImageId=[]

Label=[]

model.eval()

count=-1

for images,labels in (testloader):

    count+=1

    if train_on_gpu:

        images=images.cuda()

    log_ps = model(images)

    ps = torch.exp(log_ps)

    top_p, top_class = ps.topk(1, dim=1)

    i=0

    for i in range(64):

        try:#our dataset is not exactly divisible by 64 so it will go out of bound at some point

            Label.append(top_class[i].item())#Storing label

            ImageId.append(count*64+i+1)#atoring respective ImageId

        except Exception:

            print(i)
print((Label[:50]))
print(ImageId[:50])
submission={

    'ImageId':ImageId,

    'Label':Label

}
df = pd.DataFrame(submission)
df.head()
df.to_csv('submission.csv',index=False)
dataa=pd.read_csv('submission.csv')
dataa.head()