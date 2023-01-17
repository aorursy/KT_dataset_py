# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import glob

import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os,sys

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from memory_profiler import profile

from keras.layers import Conv3D, MaxPool3D

import random

import cv2

from imblearn.over_sampling import SMOTE



sns.set(style='white', context='notebook', palette='deep')

gc.collect()

print('gc comp')
#common



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

size =256

channel = 3



img_name_train = train['id_code']

img_name_test = test['id_code']



#y_train = np.asarray(train['diagnosis'])

y_train = train['diagnosis']



img_path_train = "../input/train_images/"

img_path_test = "../input/test_images/"
#common

g = sns.countplot(y_train)

y_train.value_counts()
#for decrease samples#



m = 0

# if diagnosis = 0 decrease the smple to 1/6,if diagnosis = 2 decrease the smple to 1/3

for id in img_name_train:

    df_comp = train[train['id_code']==id]

    val_comp = df_comp['diagnosis'].item()

    

    if val_comp == 0:

        

        if random.random() > 0.166:

            train = train.drop(m)

    

    elif val_comp == 2:

        if random.random() < 0.666:

            train = train.drop(m)

        

    m = m+1

print(train)
#for decrease samples#

y_train = train['diagnosis']

g = sns.countplot(y_train)

y_train.value_counts()
#for decrease samples#

n = d0 = d1 = d2 = d3 = d4 = 0



count_0 = np.count_nonzero(y_train == 0, axis=0)

count_1 = np.count_nonzero(y_train == 1, axis=0)

count_2 = np.count_nonzero(y_train == 2, axis=0)

count_3 = np.count_nonzero(y_train == 3, axis=0)

count_4 = np.count_nonzero(y_train == 4, axis=0)



x_train_0 = np.empty((count_0, size, size, channel), dtype=np.float32)

y_train_0 = np.zeros((count_0,2), dtype=np.int)

x_train_1 = np.empty((count_1, size, size, channel), dtype=np.float32)

y_train_1 = np.zeros((count_1,2), dtype=np.int)

x_train_2 = np.empty((count_2, size, size, channel), dtype=np.float32)

y_train_2 = np.zeros((count_2,2), dtype=np.int)

x_train_3 = np.empty((count_3, size, size, channel), dtype=np.float32)

y_train_3 = np.zeros((count_3,2), dtype=np.int)

x_train_4 = np.empty((count_4, size, size, channel), dtype=np.float32)

y_train_4 = np.zeros((count_4,2), dtype=np.int)

#for decrease samples#

img_name_train = train['id_code']

img_name_test = test['id_code']



def img_read(img_path,id):

    

    #when using cv2 and grey scale images

    if channel == 1:

        img = cv2.imread(img_path + id + '.png')

        grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_rs = np.asarray(cv2.resize(grey_img,(size,size)))

        img_rs = img_rs/255

        img_rs = img_rs.reshape(size,size,channel)

    

        return img_rs

    

    #when using rgb images

    elif channel == 3:

        #img = Image.open(img_path + id + '.png')

        img = cv2.imread(img_path + id + '.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_rs = np.asarray(cv2.resize(img,(size,size)))

        img_rs = img_rs/255

        #img_rs = img_rs.reshape(size,size,channel)

        

        return img_rs



for id in img_name_train:

    df_comp = train[train['id_code']==id]

    val_comp = df_comp['diagnosis'].item()

    val = np.asarray(df_comp['diagnosis'])

    

    if val_comp==0:



        img_rs = img_read(img_path_train,id)

        

        y_train_0[d0,:] = np.array([val_comp])

        x_train_0[d0,:,:,:] = img_rs

        d0 = d0 + 1 

        

    elif val_comp==1:

        

        img_rs = img_read(img_path_train,id)

        

        y_train_1[d1,:] = np.array([val_comp])

        x_train_1[d1,:,:,:] = img_rs

        d1 = d1 + 1

        

    elif val_comp==2:

        

        img_rs = img_read(img_path_train,id)

        

        y_train_2[d2,:] = np.array([val_comp])

        x_train_2[d2,:,:,:] = img_rs

        d2 = d2 + 1

        

    elif val_comp==3:

        

        img_rs = img_read(img_path_train,id)

        

        y_train_3[d3,:] = np.array([val_comp])

        x_train_3[d3,:,:,:] = img_rs

        

        d3 = d3 + 1

        

    elif val_comp==4:

        

        img_rs = img_read(img_path_train,id)

        

        y_train_4[d4,:] = np.array([val_comp])

        x_train_4[d4,:,:,:] = img_rs

        

        d4 = d4 + 1

    n = n + 1



print(y_train_0.shape)

print(y_train_0.shape)

print(d0)

print(n)
##non decrease samples#



#train = pd.read_csv("../input/train.csv")

#test = pd.read_csv("../input/test.csv")

#size =256



#img_name_train = train['id_code']

#img_name_test = test['id_code']

#

#N_train = train.shape[0]

#N_test = test.shape[0]

#

#x_train = np.empty((N_train, size, size, 1), dtype=np.float32)

##x_train = np.empty((N_train, size, size, 3), dtype=np.uint8)

#

#x_test = np.empty((N_test, size, size, 1), dtype=np.float32)

##x_test = np.empty((N_test, size, size, 3), dtype=np.uint8)

#

#y_train = np.asarray(train['diagnosis'])

#

#

#img_path_train = "../input/train_images/"

#img_path_test = "../input/test_images/"

#

##train

#n = 0

#for id in img_name_train:

#    #img = Image.open(img_path_train + id + '.png')

#    #img_rs = np.asarray(img.resize((size,size)))

#    

#    img = cv2.imread(img_path_train + id + '.png')

#    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#    img_rs = np.asarray(cv2.resize(grey_img,(256,256)))

#    img_rs = img_rs/255

#    

#    x_train[n, :, :,:] = img_rs.reshape(256,256,1)

#

#    n=n+1

#    

##    g = plt.imshow(x_train[3][:,:,0])

#

###test

##n = 0

##for id in img_name_test:

##    img = Image.open(img_path_test + id + '.png')

##    img_rs = np.asarray(img.resize((size,size)))

##    img_rs = img_rs/255

#    

##    x_test[n, :, :, :] = img_rs

##    #x_test[:, :, :] = img_rs

##    n = n + 1

#

#

#print(x_train.shape)

#print(y_train.shape)

##print(x_test.shape)

#print(x_train.max())

#print(y_train.max())
#print(x_train)

#print(y_train)
#for decrease samples#

x_train = np.vstack([x_train_0,x_train_1,x_train_2,x_train_3,x_train_4])

y_train = np.vstack([y_train_0,y_train_1,y_train_2,y_train_3,y_train_4])

y_train = np.delete(y_train,0,axis=1)

print(x_train.shape)

print(y_train.shape)

print(y_train)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0])



y_train_c = to_categorical(y_train, num_classes = 5)

print(y_train_c.shape)

# Set the random seed

random_seed = 2



# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train_c, test_size = 0.1, random_state=random_seed)

print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)



print(Y_val)
# Some examples

#X_train_show = np.asarray(img.resize((255,255)))

#print(X_train_show.shape)

#g = plt.imshow(X_train_show)

#print(X_train)

#print(X_val)

print(X_train[0].shape)

print(X_train.max())

print(X_train[0].max())

#temp = X_train[1][:,:,0].reshape(256,256)

temp = X_val[1][:,:,0]



temp1 = X_train[1][:,0,:]

temp2 = X_train[1][0,:,:]

temp3 = X_train[1][:,:,:]

print(temp.max())

print(temp.shape)

print(temp1.max())

print(temp1.shape)

print(temp2.max())

print(temp2.shape)

print(temp3.max())

print(temp3.shape)

#for m in range(0,3295):

#    print(X_train[0].max())

g = plt.imshow(temp)
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 32 , kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (256,256,channel)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(5, activation = "softmax"))
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
epochs = 30 # Turn epochs to 30 

batch_size = 86

# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(featurewise_center=False,

                             samplewise_center=False,

                             featurewise_std_normalization=False,

                             samplewise_std_normalization=False,

                             zca_whitening=False,

                             rotation_range=10,

                             zoom_range= 0.1,

                             width_shift_range=0.1,

                             height_shift_range=0.1,

                             horizontal_flip=False,

                             vertical_flip=False)

datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),epochs = epochs, validation_data = (X_val,Y_val),verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

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

        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(5)) 
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((size,size,channel)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
n = 0

count = np.count_nonzero(img_name_test, axis=0)

x_test = np.empty((count, size, size, channel), dtype=np.float32)



for id in img_name_test:

    #df_comp = test[test['id_code']==id]    

    img_rs = img_read(img_path_test,id)

    

    #y_test[n,:] = np.array([val_comp])

    x_test[n,:,:,:] = img_rs

        

    n = n + 1



print(n)
result = model.predict(x_test)

result = np.argmax(result,axis=1) 

print(result.shape)

print(result)

sub_data=pd.DataFrame({'id_code' : img_name_test, 'diagnosis' : result})
g = sns.countplot(sub_data['diagnosis'])

sub_data['diagnosis'].value_counts()
sub_data.to_csv('submission.csv', index=False)