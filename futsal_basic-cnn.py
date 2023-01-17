# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import torchvision.transforms as transforms

from torch.autograd import Variable



from sklearn.model_selection import train_test_split
train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv",dtype = np.float32)



# split data into features(pixels) and labels(numbers from 0 to 9)

targets_numpy = train.label.values

features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization



# train test split. Size of train data is 80% and size of test data is 20%. 

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,

                                                                             targets_numpy,

                                                                             test_size = 0.2,

                                                                             random_state = 42) 
featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long



featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long
class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()



        # Convolution 1

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.relu1 = nn.ReLU()

        

        # Max pool 1

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

     

        # Convolution 2

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)

        self.relu2 = nn.ReLU()

        

        # Max pool 2

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        

        # Fully connected 1

        self.fc1 = nn.Linear(32 * 4 * 4, 10) 

    

    def forward(self, x):

        # Convolution 1

        out = self.cnn1(x)

        out = self.relu1(out)

        

        # Max pool 1

        out = self.maxpool1(out)

        

        # Convolution 2 

        out = self.cnn2(out)

        out = self.relu2(out)

        

        # Max pool 2 

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)



        # Linear function (readout)

        out = self.fc1(out)

        

        return out



# batch_size, epoch and iteration

batch_size = 100

n_iters = 2500

n_iters = 500

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    

# Create ANN

model = CNNModel()



# Cross Entropy Loss 

error = nn.CrossEntropyLoss()



# SGD Optimizer

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
count = 0

loss_list = []

iteration_list = []

accuracy_list = []

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        

        train = Variable(images.view(100,1,28,28))

        labels = Variable(labels)

        

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        

        # Calculate softmax and ross entropy loss

        loss = error(outputs, labels)

        

        # Calculating gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

        count += 1

        if count % 50 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0

            # Iterate through test dataset

            for images, labels in test_loader:

                

                test = Variable(images.view(100,1,28,28))

                

                # Forward propagation

                outputs = model(test)

                

                # Get predictions from the maximum value

                predicted = torch.max(outputs.data, 1)[1]

                

                # Total number of labels

                total += len(labels)

                

                correct += (predicted == labels).sum()

            

            accuracy = 100 * correct / float(total)

            

            # store loss and iteration

            loss_list.append(loss.data)

            iteration_list.append(count)

            accuracy_list.append(accuracy)

            if count % 100 == 0:

                # Print Loss

                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data.item(), accuracy))

                

                
import matplotlib.pyplot as plt



# visualization loss 

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("CNN: Loss vs Number of iteration")

plt.show()



# visualization accuracy 

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("CNN: Accuracy vs Number of iteration")

plt.show()
predicted
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



g = sns.countplot(Y_train)



Y_train.value_counts()
X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)
# Set the random seed

random_seed = 2
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
g = plt.imshow(X_train[0][:,:,0])
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy

epochs = 30

batch_size = 86
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)