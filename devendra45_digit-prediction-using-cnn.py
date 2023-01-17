
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
%matplotlib inline
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white',context='notebook',palette='deep')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
y_train=train['label']

X_train=train.drop(labels=['label'],axis=1)

del train

g=sns.countplot(y_train)
y_train.value_counts()
X_train.isna().any().describe()
test.isna().any().describe()
X_train=X_train/255.0
test=test/255.0
#3dimension image --> (height=28px,width=28px,canal=1)
X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
# encode labels to one hot vectors (ex: 2 -->[0,0,1,0,0,0,0,0,0,0])

y_train=to_categorical(y_train,num_classes=10)
# set the random seed
random_seed=2
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=random_seed)
plt.imshow(X_train[0][:,:,0])
plt.imshow(X_train[1][:,:,0])
plt.imshow(X_train[2][:,:,0])
# set the cnn model
# my CNN architecture is --> [[Conv2D->relu]*2 -> MaxPool2D --> Dropout]*2 --> Flatten --> Dense --> Dropout -->Output

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
# ANN
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


#define the optimizer
optimizer=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
#compile the model
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#set learning rate annealer
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc',
                                         patience=3,
                                         verbose=1,
                                         factor=0.5,
                                         min_lr=0.00001)
datagen=ImageDataGenerator(featurewise_center=False,
                          samplewise_center=False,
                          featurewise_std_normalization=False,
                          samplewise_std_normalization=False,
                          zca_whitening=False,
                          rotation_range=10,
                          zoom_range=0.1,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          horizontal_flip=False,
                          vertical_flip=False)
datagen.fit(X_train)

# don't applied vertical nor horizontal flip as it will lead to misclassify numbers 6 and 9
#fit the model

history=model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),
                            epochs=30,
                            validation_data=(X_val,y_val),
                            verbose=2,
                            steps_per_epoch=X_train.shape[0]//32,
                            callbacks=[learning_rate_reduction])
                            
fig,ax=plt.subplots(2,1)
ax[0].plot(history.history['loss'],color='b',label="Training loss")
ax[0].plot(history.history['val_loss'],color='r',label="Validation loss")
legend=ax[0].legend(loc='best',shadow=True)

ax[1].plot(history.history['accuracy'],color='b',label="Training accuracy")
ax[1].plot(history.history['val_accuracy'],color='r',label="Validation accuracy")
legend=ax[1].legend(loc='best',shadow=True)
y_pred=model.predict(X_val)
y_pred_classes=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_val,axis=1)
con_mat=confusion_matrix(y_true,y_pred_classes)
pd.DataFrame(con_mat,columns=np.arange(0,10,1),index=np.arange(0,10,1))
results=model.predict(test)
results=np.argmax(results,axis=1)
results=pd.Series(results,name='Label')
submission=pd.concat([pd.Series(range(1,28001),name='ImageId'),results],axis=1)
submission.to_csv("cnn_mnist_predict.csv",index=False)
pd.read_csv("cnn_mnist_predict.csv").head(10)
