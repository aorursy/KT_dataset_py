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
import torch

import torch.nn as nn

import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
train = pd.read_csv(r'../input/train.csv', dtype = np.float32) 



targets_np = train.label.values

features_np = train.loc[: , train.columns != 'label'].values / 255 #normalization



features_train , features_test , target_train, target_test = train_test_split(features_np,targets_np, 

                                                                              test_size = 0.2, random_state = 42)

featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(target_train).type(torch.LongTensor)



featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(target_test).type(torch.LongTensor)



batch_size = 100

n_iter = 15000

n_epochs = int(n_iter/(len(features_train) / batch_size )) 



Train = torch.utils.data.TensorDataset(featuresTrain , targetsTrain)

Test = torch.utils.data.TensorDataset(featuresTest , targetsTest)



TrainLoader = torch.utils.data.DataLoader(Train , batch_size = batch_size , shuffle = False)

TestLoader = torch.utils.data.DataLoader(Test , batch_size = batch_size , shuffle = False)

from torch import nn, optim

import torch.nn.functional as F

class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.fc1 = nn.Linear(784 , 256)

        self.fc2 = nn.Linear(256 , 128)

        self.fc3 = nn.Linear(128 ,64)

        self.fc4 = nn.Linear(64 , 10)

        self.dropout = nn.Dropout(p = 0.2)

        

    def forward(self , x):

        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        

        x = torch.sigmoid(self.fc4(x))

        

        return x

        


model = Classifier()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters() , lr = 0.3)



test_losses = []

train_losses = []

accuracy_list = []

iteration_list = []

count = 0

for e in range(n_epochs):

    #print('Epoch {}/{}'.format(e , n_epochs))

    running_loss = 0

    for images , labels in TrainLoader:

        images = images.view( images.shape[0],-1)

        

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        count += 1

        running_loss += loss.item()

        

        

    else:

        test_loss = 0;

        accuracy = 0

        with torch.no_grad():

            model.eval()

            for images , labels in TestLoader:

                images = images.view(images.shape[0] , -1)

                output = model(images)

                test_loss += criterion(output,labels)

                

                predicted = torch.max(output.data ,1 )[1]

                

                equals = predicted == labels

                accuracy += torch.mean(equals.type(torch.FloatTensor)) 

            

            model.train()

            

            

            train_losses.append(running_loss/len(TrainLoader))

            test_losses.append(test_loss/len(TestLoader))

            accuracy_list.append(accuracy/len(TestLoader))

            iteration_list.append(count)

           

           

            print("Epoch: {}/{}".format(e+1 , n_epochs),

                "Iteration: {}..".format(iteration_list[-1]),

                 "Training Loss: {:.3f}.. ".format(train_losses[-1]),

                 "Test Loss: {:.3f}.. ".format(test_losses[-1]),

                 "Test Accuracy: {:.3f}".format(accuracy_list[-1] ))

            

print('training Completed');
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



plt.plot(iteration_list,test_losses , color = 'brown')

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("ANN: Loss vs Number of iteration")

plt.show()



# visualization accuracy 

plt.plot(iteration_list , accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("ANN: Accuracy vs Number of iteration")

plt.show()



plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.title("ANN: Loss vs Number of Epochs")

plt.legend(frameon = False)

plt.show
test = pd.read_csv('../input/test.csv')

x_test = test.values

x_test = x_test.reshape([-1, 28, 28]).astype(float)

x_test = x_test / 255

x_test = torch.from_numpy(np.float32(x_test))

x_test.shape
model.eval()

pred = model(x_test.float())

pred = torch.argmax(pred, 1)

print('Prediction: ', pred)

def export_csv(model_name, predictions):

    df = pd.DataFrame(predictions.tolist(), columns=['Label'])

    df['ImageId'] = df.index + 1

    file_name = f'submission_{model_name}.csv'

    print('Saving ',file_name)

    df[['ImageId','Label']].to_csv(file_name, index = False)
export_csv('model',pred)

torch.save(model.state_dict(), 'model.ckpt')
model.state_dict()