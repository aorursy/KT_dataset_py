# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from time import time

from torch import nn, optim

from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

y = df_train.iloc[:,0].values.astype('int32')

x = df_train.iloc[:,1:].values.astype('float32')

test_data = Variable(torch.from_numpy(df_test.values.astype('float32')))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.005, random_state=2)
x_train = Variable(torch.from_numpy(x_train))

x_test = Variable(torch.from_numpy(x_test))

y_train = Variable(torch.from_numpy(y_train)).type(torch.LongTensor)

y_test = Variable(torch.from_numpy(y_test)).type(torch.LongTensor)
input_size = 784

hidden_sizes = [128, 64, 32]

output_size = 20



model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),

                      nn.ReLU(),

                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),

                      nn.ReLU(),

                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),

                      nn.ReLU(),

                      nn.Linear(hidden_sizes[2], output_size),

                      nn.LogSoftmax(dim=1))

print(model)
criterion = nn.NLLLoss()



logps = model(x_train) 

loss = criterion(logps, y_train) #calculate the NLL loss
print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

time0 = time()

epochs = 20

for e in range(epochs):

    running_loss = 0

    for i in range(100):

        # Resetting the grad

        optimizer.zero_grad()

        

        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)

        

        #This is where the model learns by backpropagating

        loss.backward()

        

        #And optimizes its weights here

        optimizer.step()

        

        running_loss += loss.item()

        if i % 100 == 0:

            print("Epoch {} - Training loss: {}".format(e, running_loss/len(df_train)))

        

print("\nTraining Time (in minutes) =",(time()-time0)/60)
with torch.no_grad():

    logps = model(x_test)



y_pred = torch.exp(logps)

probab = list(y_pred.numpy()[0])

print("Predicted Digit =", probab.index(max(probab)))

y_pred = logps

y_pred_classes = np.argmax(y_pred,axis = 1) 


def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
confusion_mtx = confusion_matrix(y_test, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = range(10))
accuracy_score(y_test, y_pred_classes)
with torch.no_grad():

    logps = model(test_data)



y_pred = torch.exp(logps)

probab = list(y_pred.numpy()[0])

print("Predicted Digit =", probab.index(max(probab)))
y_pred = np.argmax(y_pred,axis = 1)

y_pred = pd.Series(y_pred,name="Label")
submission = pd.concat([pd.Series(range(1,len(test_data)),name = "ImageId"), y_pred],axis = 1)

submission.to_csv('submission.csv', index=False)