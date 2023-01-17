import pandas as pd

import numpy as  np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau
train=pd.read_csv('../input/Kannada-MNIST/train.csv')

train.head
train.shape
label_value_cnts=train.label.value_counts()

label_value_cnts
X_train=train.drop('label',axis=1)

X_train=X_train/255 #normalize

X_train=X_train.values.reshape(-1,28,28,1)

X_train
Y_train=train.label

Y_train=to_categorical(Y_train)
X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.2)
datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=20,  

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=False,  

        vertical_flip=False)  





datagen.fit(X_train)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
model.summary()
optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs=5 

batch_size=64
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size), epochs = epochs, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[learning_rate_reduction])
y_pre_test=model.predict(X_test)

y_pre_test=np.argmax(y_pre_test,axis=1)

y_test=np.argmax(y_test,axis=1)
conf=confusion_matrix(y_test,y_pre_test)

conf=pd.DataFrame(conf,index=range(0,10),columns=range(0,10))

conf
test=pd.read_csv('../input/Kannada-MNIST/test.csv')

test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)

test.head(5)
y_pre=model.predict(test)     ##making prediction

y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels
submission=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label']=y_pre

submission.to_csv('submission.csv',index=False)