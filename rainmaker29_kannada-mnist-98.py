import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau
# from google.colab import drive
# drive.mount('/content/drive')
train  = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')
train.head(4)
train.label.value_counts()
sns.countplot(train['label'])
X_train = train.drop(['label'],axis=1)
Y_train = train['label']
#Reshaping data

X_train = X_train/255.0
test.head()
testids = test['id']
test.drop(['id'],axis=1,inplace=True)
test = test/255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train,num_classes=10)
X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,random_state=2,test_size=0.1)
g = plt.imshow(X_train[0][:,:,0])
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu',))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#Reducing the learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lir=0.00001)
epochs=30
batch_size=86
# Data Augmentation
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=0,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1)
datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(X_test,Y_test),
                              verbose=2,
                              steps_per_epoch=X_train.shape[0]//batch_size,
                              callbacks=[learning_rate_reduction]
                              )
test = pd.read_csv('../input/Kannada-MNIST/test.csv')
test_ids = test['id']
test = test.drop(['id'],axis=1)
test = test/255.0
test=test.values.reshape(test.shape[0],28,28,1)
test.shape

y_pre = model.predict(test)
y_pre = np.argmax(y_pre,axis=1)
sample_sub = pd.DataFrame(y_pre)
sample_sub.head()
sub1 = sample_sub
sub1 = pd.concat([test_ids,sub1],axis=1)
sub1.head()
sub1 = sub1.rename(columns={0:'label'})
sub1.head()
sub1.to_csv('submission.csv',index=False)
