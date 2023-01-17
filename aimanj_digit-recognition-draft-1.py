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
# First lets load the data



train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

a = train.isnull().sum().sum()

print('null of train: ',a)

b = test.isnull().sum().sum()

print('null of test: ',b)
# Put the label of the picture as y_train. The rest as x_train

df_y = train['label']



df_x = train.drop(columns='label')



print(df_y.shape)

print(df_x.shape)
import seaborn as sns

import matplotlib.pyplot as plt

# check the value counts of y (surely should be 0-9)

df_y.value_counts()



# yup it is. Lets see the plot

sns.countplot(df_y)

plt.show()
# Get one-hot of df_y



df_y_OH = pd.get_dummies(df_y)

df_y_OH.shape
df_x = df_x.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



print(df_x.shape)

print(test.shape)
# Get the first index of each digit, store them in idxStore (this is just to plot and visualise the digits)

idxStore =[]

for i in range(10):

    idxStore.append(df_y[df_y==i].index[0])

# Plot the first index of each digit. Just so that I can get a feel about what im working on

for i in range(10):

    plt.subplot(2,5,i+1)

    a=df_x[idxStore[i]]

    plt.imshow(np.squeeze(a))

    

plt.show()
# Im just trying to show my wife the different handwritting digits varies in order to explain how this

# stuff works, so neglect this part

idxDigit4 = df_y[df_y==7].index[:10]

for i in range(10):

    plt.subplot(2,5,i+1)

    a=df_x[idxDigit4[i]]

    plt.imshow(np.squeeze(a))



plt.show()
# Normalize the images.

train_images = (df_x / 255) - 0.5

test_images = (test / 255) - 0.5
from sklearn.model_selection import train_test_split



# Split the train and the validation set for the fitting

# X_train, X_val, Y_train, Y_val = train_test_split(df_x, df_y_OH, test_size = 0.1, random_state=1)
# Splitting the train set into train_train and train_test.

# Im using StratifiedShuffleSplit cause I want the train_Train set to have 

# equal digit values to the train_Test set. 

from sklearn.model_selection import StratifiedShuffleSplit



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)



for train_index, test_index in sss.split(df_x, df_y_OH):

    X_train, X_test = df_x[train_index], df_x[test_index]

    y_train, y_test = df_y_OH.loc[train_index], df_y_OH.loc[test_index]
from keras import Sequential

from keras.layers import Conv2D, Flatten, Dense

from sklearn.metrics import accuracy_score

store_i = []

store_acc = []



for i in [5,10,15]:

    model = Sequential()

    model.add(Conv2D(i,kernel_size=2,activation='relu',input_shape=(28,28,1)))

    model.add(Conv2D(i,kernel_size=2,activation='relu'))

    model.add(Flatten())

    model.add(Dense(10,activation='softmax'))



    model.compile(optimizer='adam',

                 loss='categorical_crossentropy',

                 metrics=['accuracy'])

    training = model.fit(X_train,y_train,epochs=1,batch_size=86)

    preds = model.predict(X_test)

    y_test_ = np.argmax(y_test.to_numpy(), axis=1)

    preds_ = np.argmax(preds, axis=1)

    acc = accuracy_score(y_test_,preds_)

    

    store_i.append(i)

    store_acc.append(acc)



print('Units, Accuracy : ',store_i,store_acc)

from keras.layers import MaxPool2D

acc_avg_3_runs = []

for i in range(3):

    model = Sequential()

    model.add(Conv2D(10,kernel_size=2,activation='relu',input_shape=(28,28,1)))

#     model.add(MaxPool2D(2))

    model.add(Conv2D(10,kernel_size=2,activation='relu'))

    model.add(Flatten())

    model.add(Dense(10,activation='softmax'))



    model.compile(optimizer='adam',

                 loss='categorical_crossentropy',

                 metrics=['accuracy'])

    training = model.fit(X_train,y_train,epochs=1,batch_size=86)

    preds = model.predict(X_test)

    y_test_ = np.argmax(y_test.to_numpy(), axis=1)

    preds_ = np.argmax(preds, axis=1)

    acc = accuracy_score(y_test_,preds_)

    acc_avg_3_runs.append(acc)

    

print(i,np.mean(acc_avg_3_runs))
training = model.fit(X_train,y_train,epochs=5,batch_size=86)

history = training.history



plt.plot(history['loss'])



plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.show()
# Basically this is the same code as what I've done above



# The layers

model = Sequential()

model.add(Conv2D(10,kernel_size=2,activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(10,kernel_size=2,activation='relu'))

model.add(Flatten())

model.add(Dense(10,activation='softmax'))



# Compilation

model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



# Training

training = model.fit(X_train,y_train,epochs=2,batch_size=86)



#Predicting

preds = model.predict(X_test)



# Change results from one-hot encode to normal (ie: back to 1,2,3,4,5....)

y_test_ = np.argmax(y_test.to_numpy(), axis=1)

preds_ = np.argmax(preds, axis=1)
# Import confusion matrix and get them

from sklearn.metrics import confusion_matrix



conf_mtx = confusion_matrix(y_test_,preds_)

import matplotlib.pyplot as plt

import itertools



# Make the function 



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





# Use the confusion matrix and plot them using the function above

plot_confusion_matrix(conf_mtx, classes = range(10)) 
test_predict = model.predict(test)

test_predict = np.argmax(test_predict, axis=1)
sample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

Id = sample['ImageId']



submission = pd.DataFrame({ 'ImageId': Id,

                            'Label': test_predict })

submission.to_csv(path_or_buf ="MNIST_Submission.csv", index=False)

print("Submission file is formed")