import pandas as pd

import numpy as  np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau,EarlyStopping 



import seaborn as sns

%matplotlib inline
# load csv data

train=pd.read_csv('../input/Kannada-MNIST/train.csv')

test=pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
# data perparation

test=test.drop('id',axis=1)

X_train=train.drop('label',axis=1)

Y_train=train.label



# normalize

X_train=X_train/255

test=test/255



# reshape data

X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)



print('The shape of train set now is',X_train.shape)

print('The shape of test set now is',test.shape)
# one-hot-vector

Y_train=to_categorical(Y_train)



# split train and test data

X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.1)
# Some examples

g = plt.imshow(X_train[0][:,:,0])
# image augmantaion

datagen = ImageDataGenerator(rotation_range = 10,

                             width_shift_range = 0.3,

                             height_shift_range = 0.3,

                             shear_range = 0.15,

                             zoom_range = 0.3,

                            )



datagen.fit(X_train)
# model

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
# Set hyper paramater

epochs=100

batch_size=1024



# optimizer

optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)

model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])



# learning_rate

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# early-stopping

early_stopping  = EarlyStopping(monitor='val_loss', 

                                min_delta=0, 

                                patience=10, 

                                verbose=0, 

                                mode='auto')



callbacks = [learning_rate_reduction, early_stopping]
# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,y_test),

                              verbose = 2, 

                              steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=callbacks)
#predict validation data

y_pre_test=model.predict(X_test)

y_pre_test=np.argmax(y_pre_test,axis=1)

y_test=np.argmax(y_test,axis=1)
# check predict result

x=(y_pre_test-y_test!=0).tolist()

x=[i for i,l in enumerate(x) if l!=False]



fig,ax=plt.subplots(1,4,sharey=False,figsize=(15,15))



for i in range(4):

    ax[i].imshow(X_test[x[i]][:,:,0])

    ax[i].set_xlabel('Ans {}, Pre {}'.format(y_test[x[i]],y_pre_test[x[i]]))
# output

y_pre=model.predict(test)     ##making prediction

y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels

sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)