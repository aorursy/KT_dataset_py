# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#preliminaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import RMSprop
#Loading the train data set
train = pd.read_csv("../input/train.csv")
print("shape for the train data is: ", train.shape)
train.head()
#Loading the tests set
test = pd.read_csv("../input/test.csv")
print("shape for the test data is: ", test.shape)
test.head()
#We check to see is there are no missing values
train.isnull().any().describe()
test.isnull().any().describe()
X_train = train.iloc[:, 1:].values #this is because i do not want the label column on X_train
X_train.shape
y_train = train['label'].values
y_train.shape

X_train[0]
#Lets tke a look at data
y_train     #y_train is an array of 42000 digits all betwenn 0 and 9
#keras work weel with floats, so we must cast the numbers to floats

X_train = X_train.astype('float32')
y_train = y_train.astype('int32')   #we want them in integers
X_test = test.values.astype('float32')
print('The length of X_train is:', len(X_train))
print('The length of y_train is:', len(y_train))
print('The length of X_test is:', len(X_test))
# X_train = X_train.reshape(len(X_train), 28, 28, 1)
# X_test = X_test.reshape(len(X_test), 28, 28, 1)

# print('After reshapping, the shape of the X_train is: ', X_train.shape)
# print('After reshapping, the shape of the X_test set is: ', X_test.shape)
# y_train[765]    #y_train of 765 is 8. lets confirm this with the corresponding x_train
# plt.imshow(X_train[765][:,:,0], cmap='Greys_r');
#Normalising all the numbers so that they are between o and 1
X_train = X_train/255.0
X_test = X_test/255.0
#As we saw, y_train cotains digits from 0 to 9. this gives us 10 classes.
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
#X_test = keras.utils.to_categorical(X_test, num_classes)
#Lets see what we did above. Remember y_train[765] = 8, now
print("Remember? before the step above, y_train[765] was 8 ")
print("Now, y_train[765] is: ", y_train[765])
print("\n")
print("This is because we did some form of one_hot_encoding ")
model = Sequential()
model.add(Dense(64, activation='relu', input_shape = (784,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
#model.add(Flatten())
model.add(Dense(10, activation='softmax')) #10 since we have 10digits from 0 to 9
#lets get a model summary
model.summary()
# Let's compile the model
learning_rate = .001
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])
#We need to split our train data into train and validation set to fed it into the neuralnet
from sklearn.model_selection import train_test_split
x_train,x_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)


#Let's fit the model
epoch = 30

history =model.fit(x_train, y_train,
                  epochs = epoch,
                  batch_size = 128,
                  verbose = 1,
                  validation_data = (x_val, y_val))
#we evaluate the performance of our model on the validation data.
score = model.evaluate(x_val, y_val, verbose = 0)
print("The accuracy of the validation is: ", score[1])
print("The loss of the validation is: ", score[0])
history.history.keys()
#We plot the cross_entropy loss vs the accuracy.
def plot_loss_accuracy(history):
    fig = plt.figure(figsize = (12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"], 'r-x', label = "train loss")
   
    ax.plot(history.history["val_loss"], 'g-x', label = "validation loss")
    ax.legend()
    ax.set_title("Cross entropy loss")
    ax.grid(True)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"], "r-x", label = "train accuracy")
    ax.plot(history.history["val_acc"], "g-x", label = "validation accuracy")
    ax.legend()
    ax.set_title("Accuracy")
    ax.grid(True)
 

plot_loss_accuracy(history)
X_test.shape
# my_prediction = model.predict_classes(X_test)

# submission_mnist = pd.DataFrame({"ImageId" : list(range(1, 28001)),
#                                 "Label" : my_prediction})
# #creating csv file
# submission_mnist.to_csv("mnist_digits_DNN.csv", index = False)
#We build a 2layer model with first layer=400 and the second layer = 300 with a dropout of 0.4
model_2 = Sequential()
model_2.add(Dense(400, activation='relu', input_shape = (784,)))
model_2.add(Dropout(0.4))

model_2.add(Dense(300, activation='relu',input_shape = (784,)))
model_2.add(Dropout(0.4))
#model.add(Flatten())

model_2.add(Dense(10, activation='softmax')) #10 since we have 10digits from 0 to 9

model_2.summary()
# Let's compile the model_2
learning_rate = .001
model_2.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])
epoch = 30

history_2 =model_2.fit(x_train, y_train,
                  epochs = epoch,
                  batch_size = 128,
                  verbose = 1,
                  validation_data = (x_val, y_val))
#we evaluate the performance of our model on the validation data.
score_2 = model_2.evaluate(x_val, y_val, verbose = 0)
print("The accuracy of the validation is: ", score_2[1])
print("The loss of the validation is: ", score_2[0])
plot_loss_accuracy(history_2)
my_prediction = model.predict_classes(X_test)

submission_mnist = pd.DataFrame({"ImageId" : list(range(1, 28001)),
                                "Label" : my_prediction})
#creating csv file
submission_mnist.to_csv("mnist_digits_DeeperNN.csv", index = False)
