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
train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')
train_data.head(2)
test_data.head(1)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

import itertools



np.random.seed(2)



from keras.models import Sequential

from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten

from keras.optimizers import Adam,RMSprop

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



sns.set(style='white', context='notebook', palette='deep')
x_train= train_data.drop(labels=['label'],axis=1)

y_train=train_data['label'] 

del train_data
#Build frequency chart of classes to look for class imbalance



g=sns.countplot(y_train)

y_train.value_counts()
#Check for missing and null values

x_train.isnull().any().describe()

test_data.isnull().any().describe()
# Split data into Train and Validation



x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=.2,random_state=2)

#One hot encode the labels using to_categorical

y_train=to_categorical(y_train)

y_val=to_categorical(y_val)
print((x_train.shape,y_train.shape))

print((x_val.shape,y_val.shape))
#Reshape the data

x_train=x_train.values.reshape(-1,28,28,1)

x_val=x_val.values.reshape(-1,28,28,1)

test_data=test_data.values.reshape(-1,28,28,1)
#Create Augmentation Function using ImageDataGenerator



train_datagen= ImageDataGenerator(

                                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

                                zoom_range = 0.1, # Randomly zoom image 

                                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                                height_shift_range=0.1)# randomly shift images vertically (fraction of total height)



val_datagen=ImageDataGenerator(

                                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

                                zoom_range = 0.1, # Randomly zoom image 

                                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                                height_shift_range=0.1)# randomly shift images vertically (fraction of total height)



test_datagen=ImageDataGenerator(

                                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

                                zoom_range = 0.1, # Randomly zoom image 

                                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                                height_shift_range=0.1)# randomly shift images vertically (fraction of total height)
train_datagen.fit(x_train)

val_datagen.fit(x_val)
#Set Learning rate annealer



lrr=ReduceLROnPlateau(

                     monitor='val_acc',

                     patience=3,

                     verbose=1,

                     factor=0.5,

                     min_lr=.00001)
# Set Optimizer

optimizer= Adam(lr=.001,beta_1=0.9, beta_2=0.999,epsilon=1e-6)

optimizer_RMS = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#Define the Model- 2 Convolutional layers with kernel size (32,5,5),Maxpool with kernel(2,2), Dropout=.25

#                  2 Convolutional layers with kernel size (64,3,3),Maxpool with kernel(2,2), Dropout=.25   

#                  1 Flatten layer, 1 Dense layer(256) ,dropout =.5 and classification layer(10)



model=Sequential()



model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(.25))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(.25))



model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(.5))

model.add(Dense(10,activation='softmax'))



#Compile the model

model.compile(optimizer=optimizer_RMS,loss='categorical_crossentropy',metrics=['accuracy'])
#Define batch_size,epochs

batch_size=86

epochs=50
#Fit the model



model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=batch_size),epochs=epochs,

                   validation_data=(x_val,y_val),

                   steps_per_epoch=x_train.shape[0]//batch_size,callbacks=[lrr],verbose=1)
#EVALUATE THE MODEL
#Plot training loss and validation loss

fig,ax= plt.subplots(2,1)

ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')

ax[0].plot(model.history.history['val_loss'],color='r',label='Validation_Loss')



#Plot Training and validation accuracy

ax[1].plot(model.history.history['acc'],color='b')

ax[1].plot(model.history.history['val_acc'],color='r')
#Confusion Matrix

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
#Predict values from validation dataset

y_pred=model.predict(x_val)

#Convert prediction classes to 1 hot vectors

y_pred_classes=np.argmax(y_pred,axis=1)

#Convert Validation classes to 1 hot vectors

y_true=np.argmax(y_val,axis=1)



#Compute the confusion matrix

confusion_mtx=confusion_matrix(y_true,y_pred_classes)



plot_confusion_matrix(confusion_mtx, classes = range(10))
#Display Top 6 Error Results



errors=(y_pred_classes-y_true !=0)

y_pred_class_errors= y_pred_classes[errors]

y_true_errors=y_true[errors] #Gives 

y_pred_errors=y_pred[errors]

x_val_errors = x_val[errors]
y_true_errors
y_pred_class_errors
def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1
# Probabilities of the wrong predicted numbers

y_pred_errors_prob = np.max(y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, x_val_errors, y_pred_class_errors, y_true_errors)
# Predict Results



results=model.predict(test_data)

# select the index with the maximum probability

results=np.argmin(results,axis=1)

results = pd.Series(results,name="Label")

submission=pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)