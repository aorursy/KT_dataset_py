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


# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

%matplotlib inline







import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)







import pandas as pd 

pd.options.display.max_columns = 100



import numpy as np 





from PIL import Image

from scipy import ndimage



import seaborn as sns



import pylab as plot

params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

}

plot.rcParams.update(params)





sns.set_style("darkgrid", {"axes.facecolor": ".95"})

%config InlineBackend.figure_format = 'svg'



import cv2

import glob

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Nadam, Adadelta, Adamax

from keras.utils import to_categorical

from matplotlib import pyplot





images1 = []

targets1 = []







path1 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/dyed-lifted-polyps"

path2 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/dyed-resection-margins"

path3 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/esophagitis"

path4 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/normal-cecum"

path5 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/normal-pylorus"

path6 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/normal-z-line"

path7 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/polyps"

path8 = "../input/kvasir-dataset-v2/kvasir-dataset-v2/ulcerative-colitis"







for i in glob.glob(os.path.join(path1,'*jpg')):

    img = cv2.imread(i)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(0)

    

for j in glob.glob(os.path.join(path2,'*jpg')):

    img = cv2.imread(j)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(1)    

    

    

for k in glob.glob(os.path.join(path3,'*jpg')):

    img = cv2.imread(k)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(2)

    

for l in glob.glob(os.path.join(path4,'*jpg')):

    img = cv2.imread(l)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(3)    

    

    

for m in glob.glob(os.path.join(path5,'*jpg')):

    img = cv2.imread(m)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(4)

    

for n in glob.glob(os.path.join(path6,'*jpg')):

    img = cv2.imread(n)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(5)    

    

for o in glob.glob(os.path.join(path7,'*jpg')):

    img = cv2.imread(o)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(6)

    

for p in glob.glob(os.path.join(path8,'*jpg')):

    img = cv2.imread(p)

    #img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale

    img = cv2.resize(img,(50,50))  #resize

    images1.append(np.array(img))

    targets1.append(7)       

    

    





endoscope_images1 , endoscope_labels1 = np.array(images1), np.array(targets1)    

np.save("endoscope_images1",endoscope_images1)

np.save("endoscope_labels1",endoscope_labels1)
the_endoscope_images1, the_endoscope_labels1 = np.load("endoscope_images1.npy") , np.load("endoscope_labels1.npy")

fig=plt.figure(figsize=(15, 11))

n = 36

for i in range(n):

    img = np.random.randint(0, the_endoscope_images1.shape[0] , 1)

    fig.add_subplot(n**(.5), n**(.5), i+1)

    plt.imshow(the_endoscope_images1[img[0]])



    plt.title('{}:{}' .format('dyed-lifted-polyps' if the_endoscope_labels1[img[0]]==0 else 'dyed-resection-margins' if the_endoscope_labels1[img[0]]==1 else 'esophagitis' if the_endoscope_labels1[img[0]]==2 else 'normal-cecum' if the_endoscope_labels1[img[0]]==3 else 'normal-pylorus' if the_endoscope_labels1[img[0]]==4 else 'normal-z-line' if the_endoscope_labels1[img[0]]==5 else 'polyps' if the_endoscope_labels1[img[0]]==6  else 'ulcerative-colitis' ,the_endoscope_labels1[img[0]]) )



    plt.xticks([]) , plt.yticks([])

        

plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(the_endoscope_images1, the_endoscope_labels1, test_size=0.2)

    

    #converting to float and normalizing

X_train = X_train.astype('float32')/255 

X_test = X_test.astype('float32')/255

    

    #getting the numbr of unique classes in the labels

#num_classes=len(np.unique(the_endoscope_labels1))



num_classes = 8

    

    

#     #One hot encoding as classifier since we  has multiple classes

Y_train=keras.utils.to_categorical(Y_train,num_classes)

Y_test=keras.utils.to_categorical(Y_test,num_classes )

from keras.layers import Dense, Activation





nnmodel = Sequential()

nnmodel.add(Dense(32, input_shape=(50,50,3)))

nnmodel.add(Activation('relu'))

nnmodel.add(Flatten())

nnmodel.add(Dense(8,activation="softmax")) 

nnmodel.summary()
nnmodel.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['mae', 'acc'])



nnhistory =nnmodel.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)

plt.figure(figsize=(10, 6))  



# summarize history for accuracy

plt.plot(nnhistory.history['acc'])

plt.plot(nnhistory.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test' ], loc='upper left')

plt.show()

# summarize history for loss

plt.figure(figsize=(10,6))

plt.plot(nnhistory.history['loss'])

plt.plot(nnhistory.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show() 
plt.figure(figsize=(10, 6))  



plt.plot(nnhistory.history['mean_absolute_error'])

plt.title('Model training mean_absolute_error')

plt.ylabel('mean_absolute_error')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()



plt.figure(figsize=(10, 6))  



plt.plot(nnhistory.history['val_mean_absolute_error'])

plt.title('Model validation mean_absolute_error')

plt.ylabel('mean_absolute_error')

plt.xlabel('epoch')

plt.legend(['test' ], loc='upper left')

plt.show()
from keras.layers import Dense, Activation

# import regularizer

from keras.regularizers import l1

# instantiate regularizer

reg = l1(0.001)
model = Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(50,activation="relu"))

model.add(Dense(8,activation="softmax",activity_regularizer=l1(0.001)))#2 represent output layer neurons 

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['mae', 'acc'])



history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)
# summarize history for accuracy

plt.figure(figsize=(10, 6))  



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.figure(figsize=(10, 6))  



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show() 
plt.figure(figsize=(10, 6))  



plt.plot(nnhistory.history['mean_absolute_error'])

plt.title('Model training mean_absolute_error')

plt.ylabel('mean_absolute_error')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()



plt.figure(figsize=(10, 6))  



plt.plot(nnhistory.history['val_mean_absolute_error'])

plt.title('Model validation mean_absolute_error')

plt.ylabel('mean_absolute_error')

plt.xlabel('epoch')

plt.legend(['test' ], loc='upper left')

plt.show()
from sklearn.metrics import classification_report



y_pred=model.predict(X_test) 

y_pred=np.argmax(y_pred, axis=1) 

y_true = Y_test

y_true=np.argmax(y_true, axis=1) 





print(classification_report(y_true, y_pred))



#Making confusion matrix that checks accuracy of the model

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm
import scikitplot 

%matplotlib inline



#y_true = # ground truth labels

#y_probas = # predicted probabilities generated by sklearn classifier

y_probas = model.predict(X_test)

scikitplot.metrics.plot_roc(y_true, y_probas)

plt.figure(figsize=(50, 5)) 

#plt.legend(fontsize = 'xx-small')

plt.show()
plt.savefig('KvasirCnnROCcurve.png')
# import regularizer

from keras.layers import  Dropout, BatchNormalization



from keras.regularizers import l1,l2

# instantiate regularizer

reg = l1(0.001)

reg2 = l2(0.001)
nn1model = Sequential()

nn1model.add(Dense(32, input_shape=(50,50,3)))

nn1model.add(Activation('relu'))

# nn1model.add(Dense(64, activation="relu"))

# nn1model.add(Dense(128, activation="relu"))

# nn1model.add(Dense(512, activation="relu"))

nn1model.add(Flatten())

nn1model.add(Dense(8,activation="softmax", activity_regularizer=l1(0.001))) 

#nn1model.summary()



nn1model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01),metrics=['accuracy'])



nn1history =nn1model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, verbose=1)
nn1model.summary()

plt.figure(figsize=(10, 6))  





# summarize history for accuracy

plt.plot(nn1history.history['acc'])

plt.plot(nn1history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test' ], loc='upper left')

plt.show()

plt.figure(figsize=(10, 6))  



# summarize history for loss

plt.plot(nn1history.history['loss'])

plt.plot(nn1history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
score = nn1model.evaluate(X_test, Y_test, verbose=1)

print('\n', 'Test_Loss:-', score[0])

print('\n', 'Test_Accuracy:-', score[1])
from keras.layers import  Dropout, BatchNormalization
cnn1model = Sequential()

cnn1model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

#cnn1model.add(BatchNormalization())

cnn1model.add(MaxPooling2D(pool_size=2))

#cnn1model.add(Dropout(0.5))      

cnn1model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

#cnn1model.add(BatchNormalization())

cnn1model.add(MaxPooling2D(pool_size=2))

#cnn1model.add(Dropout(0.5))      

cnn1model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

cnn1model.add(MaxPooling2D(pool_size=2))

#cnn1model.add(BatchNormalization())

#cnn1model.add(Dropout(0.25))      

cnn1model.add(Flatten())

cnn1model.add(Dense(50,activation="relu"))

cnn1model.add(Dense(8,activation="softmax" ,kernel_regularizer=l2(0.01))) #,activity_regularizer=l1(0.01)))

#cnn1model.summary()

cnn1model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

cnn1history = cnn1model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, verbose=1)
cnn1model.summary()

score1 = cnn1model.evaluate(X_test, Y_test, verbose=1)

print('\n', 'Test_Loss:-', score1[0])

print('\n', 'Test_Accuracy:-', score1[1])
plt.figure(figsize=(10, 6))  



# summarize history for accuracy

plt.plot(cnn1history.history['acc'])

plt.plot(cnn1history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test' ], loc='upper left')

plt.show()



plt.figure(figsize=(10, 6))  



# summarize history for loss

plt.plot(cnn1history.history['loss'])

plt.plot(cnn1history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show() 
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau





reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
cnnmodel_rmsprop = Sequential()

cnnmodel_rmsprop.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_rmsprop.add(MaxPooling2D(pool_size=2))

cnnmodel_rmsprop.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_rmsprop.add(MaxPooling2D(pool_size=2))

cnnmodel_rmsprop.add(Flatten())

cnnmodel_rmsprop.add(Dense(50,activation="relu"))

cnnmodel_rmsprop.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_rmsprop.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])

cnnmodel_adam = Sequential()

cnnmodel_adam.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_adam.add(MaxPooling2D(pool_size=2))

cnnmodel_adam.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_adam.add(MaxPooling2D(pool_size=2))

cnnmodel_adam.add(Flatten())

cnnmodel_adam.add(Dense(50,activation="relu"))

cnnmodel_adam.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

cnnmodel_adam.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

hist_adam = cnnmodel_adam.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])

cnnmodel_nadam = Sequential()

cnnmodel_nadam.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_nadam.add(MaxPooling2D(pool_size=2))

cnnmodel_nadam.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_nadam.add(MaxPooling2D(pool_size=2))

cnnmodel_nadam.add(Flatten())

cnnmodel_nadam.add(Dense(50,activation="relu"))

cnnmodel_nadam.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

cnnmodel_nadam.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['accuracy'])

hist_nadam = cnnmodel_nadam.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])

cnnmodel_sgd= Sequential()

cnnmodel_sgd.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_sgd.add(MaxPooling2D(pool_size=2))

cnnmodel_sgd.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_sgd.add(MaxPooling2D(pool_size=2))

cnnmodel_sgd.add(Flatten())

cnnmodel_sgd.add(Dense(50,activation="relu"))

cnnmodel_sgd.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_sgd.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])

hist_sgd = cnnmodel_sgd.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])



cnnmodel_sgdnesterov = Sequential()

cnnmodel_sgdnesterov.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_sgdnesterov.add(MaxPooling2D(pool_size=2))

cnnmodel_sgdnesterov.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_sgdnesterov.add(MaxPooling2D(pool_size=2))

cnnmodel_sgdnesterov.add(Flatten())

cnnmodel_sgdnesterov.add(Dense(50,activation="relu"))

cnnmodel_sgdnesterov.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_sgdnesterov.compile(optimizer=SGD(nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])

hist_sgdnesterov = cnnmodel_sgdnesterov.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])

cnnmodel_sgdmomentum = Sequential()

cnnmodel_sgdmomentum.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_sgdmomentum.add(MaxPooling2D(pool_size=2))

cnnmodel_sgdmomentum.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_sgdmomentum.add(MaxPooling2D(pool_size=2))

cnnmodel_sgdmomentum.add(Flatten())

cnnmodel_sgdmomentum.add(Dense(50,activation="relu"))

cnnmodel_sgdmomentum.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_sgdmomentum.compile(optimizer=SGD(momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])

hist_sgdmomentum = cnnmodel_sgdmomentum.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])



cnnmodel_sgdnestmomentum = Sequential()

cnnmodel_sgdnestmomentum.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_sgdnestmomentum.add(MaxPooling2D(pool_size=2))

cnnmodel_sgdnestmomentum.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_sgdnestmomentum.add(MaxPooling2D(pool_size=2))

cnnmodel_sgdnestmomentum.add(Flatten())

cnnmodel_sgdnestmomentum.add(Dense(50,activation="relu"))

cnnmodel_sgdnestmomentum.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_sgdnestmomentum.compile(optimizer=SGD(momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])

hist_sgdnestmomentum = cnnmodel_sgdnestmomentum.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])

cnnmodel_adagrad = Sequential()

cnnmodel_adagrad.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_adagrad.add(MaxPooling2D(pool_size=2))

cnnmodel_adagrad.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_adagrad.add(MaxPooling2D(pool_size=2))

cnnmodel_adagrad.add(Flatten())

cnnmodel_adagrad.add(Dense(50,activation="relu"))

cnnmodel_adagrad.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_adagrad.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])

#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])

hist_adagrad = cnnmodel_adagrad.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])



cnnmodel_adadelta = Sequential()

cnnmodel_adadelta.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_adadelta.add(MaxPooling2D(pool_size=2))

cnnmodel_adadelta.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_adadelta.add(MaxPooling2D(pool_size=2))

cnnmodel_adadelta.add(Flatten())

cnnmodel_adadelta.add(Dense(50,activation="relu"))

cnnmodel_adadelta.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_adadelta.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])

hist_adadelta = cnnmodel_adadelta.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])

cnnmodel_adamax = Sequential()

cnnmodel_adamax.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

cnnmodel_adamax.add(MaxPooling2D(pool_size=2))

cnnmodel_adamax.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnnmodel_adamax.add(MaxPooling2D(pool_size=2))

cnnmodel_adamax.add(Flatten())

cnnmodel_adamax.add(Dense(50,activation="relu"))

cnnmodel_adamax.add(Dense(8,activation="softmax"))#2 represent output layer neurons 

#cnnmodel_rmsprop.summary()

cnnmodel_adamax.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

#hist_rmsprop = cnnmodel_rmsprop.fit(X_train, Y_train, batch_size=batch_size*2,  nb_epoch=0, validation_data=(X_test,Y_test), callbacks=[reduce_lr])

hist_adamax = cnnmodel_adamax.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, verbose=1, callbacks=[reduce_lr])



plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0.4, 1))



plt.plot(hist_rmsprop.history['val_acc'])

plt.plot(hist_adam.history['val_acc'])

plt.plot(hist_nadam.history['val_acc'])

plt.plot(hist_sgd.history['val_acc'])

plt.plot(hist_sgdnesterov.history['val_acc'])

plt.plot(hist_sgdmomentum.history['val_acc'])

plt.plot(hist_sgdnestmomentum.history['val_acc'])

plt.plot(hist_adagrad.history['val_acc'])

plt.plot(hist_adadelta.history['val_acc'])

plt.plot(hist_adamax.history['val_acc'])

plt.title('val. accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='lower right', fontsize = 'x-small')  



plt.show()
plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0.4, 1))





plt.plot(hist_rmsprop.history['acc'])  

plt.plot(hist_adam.history['acc'])  

plt.plot(hist_nadam.history['acc']) 

plt.plot(hist_sgd.history['acc']) 

plt.plot(hist_sgdnesterov.history['acc']) 

plt.plot(hist_sgdmomentum.history['acc'])

plt.plot(hist_sgdnestmomentum.history['acc'])

plt.plot(hist_adagrad.history['acc'])

plt.plot(hist_adadelta.history['acc'])

plt.plot(hist_adamax.history['acc'])

plt.title('train accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='lower right', fontsize = 'x-small')  



plt.show()
plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0, 1))





plt.plot(hist_rmsprop.history['val_loss'])  

plt.plot(hist_adam.history['val_loss'])  

plt.plot(hist_nadam.history['val_loss']) 

plt.plot(hist_sgd.history['val_loss']) 

plt.plot(hist_sgdnesterov.history['val_loss']) 

plt.plot(hist_sgdmomentum.history['val_loss'])

plt.plot(hist_sgdnestmomentum.history['val_loss'])

plt.plot(hist_adagrad.history['val_loss'])

plt.plot(hist_adadelta.history['val_loss'])

plt.plot(hist_adamax.history['val_loss'])

plt.title('val loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='upper right', fontsize = 'x-small')  



plt.show()
plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0, 1))





plt.plot(hist_rmsprop.history['loss'])  

plt.plot(hist_adam.history['loss'])  

plt.plot(hist_nadam.history['loss']) 

plt.plot(hist_sgd.history['loss']) 

plt.plot(hist_sgdnesterov.history['loss']) 

plt.plot(hist_sgdmomentum.history['loss'])

plt.plot(hist_sgdnestmomentum.history['loss'])

plt.plot(hist_adagrad.history['loss'])

plt.plot(hist_adadelta.history['loss'])

plt.plot(hist_adamax.history['loss'])

plt.title('train loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'adam', 'nadam', 'sgd', 'sgd_w/nesterov', 'sgd_w/momentum', 'sgd_w/nesterov+momentum', 'adagrad', 'adadelta', 'adamax'], loc='upper right',fontsize = 'x-small')  



plt.show()
plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0.4, 1))



plt.plot(hist_rmsprop.history['val_acc'])

plt.plot(hist_nadam.history['val_acc'])

plt.plot(hist_sgd.history['val_acc'])

plt.plot(hist_sgdnesterov.history['val_acc'])



plt.title('test. accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'nadam', 'sgd', 'sgd_w/nesterov'], loc='lower right', fontsize = 'x-small')  



plt.show()
plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0.4, 1))





plt.plot(hist_rmsprop.history['acc'])  

plt.plot(hist_nadam.history['acc']) 

plt.plot(hist_sgd.history['acc']) 

plt.plot(hist_sgdnesterov.history['acc']) 

plt.title('train accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'nadam', 'sgd', 'sgd_w/nesterov'], loc='lower right', fontsize = 'x-small')  



plt.show()
plt.figure(figsize=(10, 6))  





plt.plot(hist_rmsprop.history['val_loss'])  

plt.plot(hist_nadam.history['val_loss']) 

plt.plot(hist_sgd.history['val_loss']) 

plt.plot(hist_sgdnesterov.history['val_loss']) 



plt.title('test loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'nadam', 'sgd', 'sgd_w/nesterov'], loc='upper right', fontsize = 'x-small')  



plt.show()
plt.figure(figsize=(10, 6))  

#plt.axis((-1,14,0, 1))





plt.plot(hist_rmsprop.history['loss'])  

plt.plot(hist_nadam.history['loss']) 

plt.plot(hist_sgd.history['loss']) 

plt.plot(hist_sgdnesterov.history['loss']) 



plt.title('train loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['rmsprop', 'nadam', 'sgd', 'sgd_w/nesterov'], loc='upper right',fontsize = 'x-small')  



plt.show()
cnndeepmodel = Sequential()

# first convolution layer

cnndeepmodel.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3))) 

cnndeepmodel.add(BatchNormalization())

cnndeepmodel.add(MaxPooling2D(pool_size=2))

cnndeepmodel.add(Dropout(0.25))



#second convolution  layer

cnndeepmodel.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

cnndeepmodel.add(BatchNormalization())

cnndeepmodel.add(MaxPooling2D(pool_size=2))

cnndeepmodel.add(Dropout(0.5))



#Third convolution  layer

cnndeepmodel.add(Conv2D(64, kernel_size=2 ,padding="same",activation='relu'))

cnndeepmodel.add(BatchNormalization())

cnndeepmodel.add(MaxPooling2D(pool_size=2))

cnndeepmodel.add(Dropout(0.5))



#first Fully connected layer

cnndeepmodel.add(Flatten()) 

cnndeepmodel.add(Dense(256,kernel_regularizer=l2(0.001)))#activity_regularizer=l1(0.001)))

cnndeepmodel.add(BatchNormalization())

cnndeepmodel.add(Activation('relu')) 

cnndeepmodel.add(Dropout(0.5))      



#Final Fully connected layer

cnndeepmodel.add(Dense(8)) 

cnndeepmodel.add(Activation('softmax')) 



cnndeepmodel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])



cnndeepmodel.summary()



cnndeephistory = cnndeepmodel.fit(X_train, Y_train, epochs=100,verbose=1,shuffle=True,callbacks=[reduce_lr])



cnnscore = cnndeepmodel.evaluate(X_test, Y_test, verbose=0)



#loss and accuracy

print('Test loss:', cnnscore[0])

print('Test accuracy:', cnnscore[1])
# from keras.models import load_model



# cnndeepmodel.save('my_cnndeepmodel.h5')
from keras.callbacks import ModelCheckpoint



#modelcheckpoint = ModelCheckpoint("keras.model", savebest_only=True, verbose=1)

checkpoint=ModelCheckpoint(filepath='/input/cnndeepmodel.h5',save_best_only=True )

#print(checkpoint)