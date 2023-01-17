import pandas as pd

import os,cv2

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras.models import Sequential

from keras import layers,models

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator

print(os.listdir("../input/aerial-cactus-identification"))



import numpy as np
train_dir='../input/aerial-cactus-identification/train/train'

test_dir='../input/aerial-cactus-identification/test/test'

train=pd.read_csv('../input/aerial-cactus-identification/train.csv')

test=pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
train.head()
train.info()
train.has_cactus=train.has_cactus.astype(str)
print('our dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
train.has_cactus.value_counts()
Image(os.path.join(train_dir,train.iloc[1,0]),width=250,height=250)
gen_data=ImageDataGenerator(rescale=1./255)

batch_size=150
test.has_cactus=test.has_cactus.astype(str)
train_generator=gen_data.flow_from_dataframe(dataframe=train[:15001],directory=train_dir,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(150,150))





validation_generator=gen_data.flow_from_dataframe(dataframe=train[15000:],directory=train_dir,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))



test_gen=gen_data.flow_from_dataframe(dataframe=test,directory=test_dir,x_col='id',y_col='has_cactus',class_mode=None,batch_size=50,target_size=(150,150))
model=Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.2))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

         
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
epochs=10

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()

plt.show()
acc=history.history['loss']    ##getting  loss of each epochs

epochs_=range(0,epochs)

plt.plot(epochs_,acc,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



acc_val=history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,acc_val,label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()

plt.show()
y_pre=model.predict(test_gen)
df=pd.DataFrame({'id':test['id'] })

df['has_cactus']=y_pre

df.to_csv("submissionchiro.csv",index=False)
df