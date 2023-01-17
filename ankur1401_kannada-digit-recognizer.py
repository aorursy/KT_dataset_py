import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten,Activation,BatchNormalization

from keras.models import Sequential

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy

from sklearn.metrics import classification_report

from IPython.display import FileLink
from keras.backend import tensorflow_backend

tensorflow_backend._get_available_gpus()
train=pd.read_csv(r'../input/Kannada-MNIST/train.csv')

test=pd.read_csv(r'../input/Kannada-MNIST/test.csv')

val=pd.read_csv(r'../input/Kannada-MNIST/Dig-MNIST.csv')
print('Train Set Shape:',train.shape)

print('Test Set Shape:',test.shape)

print('Validation Set Shape:',val.shape)
data=train.iloc[:,1:].values

data=data.reshape(-1,28,28,1)/255

print('Total training data shape:',data.shape)
labels=pd.get_dummies(train.iloc[:,0]).values
x_val=val.iloc[:,1:].values

x_val=x_val.reshape(-1,28,28,1)/255
y_val=pd.get_dummies(val.iloc[:,0]).values
fig,ax=plt.subplots(5,10)

for i in range(5):

    for j in range(10):

        ax[i][j].imshow(data[np.random.randint(0,data.shape[0]),:,:,0],cmap=plt.cm.binary)

        ax[i][j].axis('off')

plt.subplots_adjust(wspace=0, hspace=0)        

fig.set_figwidth(15)

fig.set_figheight(7)

fig.show()
aug_data=ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
fig,ax=plt.subplots(3,10)

idx=np.random.randint(0,data.shape[0])

for i in range(3):

    for j in range(10):

        ax[i][j].axis('off')

        X,y=aug_data.flow(data[idx].reshape(-1,28,28,1),labels[idx].reshape(1,10)).next()

        ax[i][j].imshow(X.reshape(28,28),cmap=plt.cm.binary)

fig.set_figheight(5)

fig.set_figwidth(15)

fig.show()
generator=aug_data.flow(data,labels,batch_size=64)
def create_model():

    model=Sequential()



    model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,(5,5),strides=(2,2),padding='same',activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128,(3,3),activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3),activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128,(5,5),strides=(2,2),padding='same',activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(256,(3,3),activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1024,activation='relu'))



    model.add(Dropout(0.4))

    model.add(Dense(10,activation='softmax'))

    

    model.compile(optimizer='adam',loss=categorical_crossentropy,metrics=['accuracy'])

    

    return model
model=create_model()
history=model.fit_generator(generator,epochs=50,validation_data=(x_val,y_val),steps_per_epoch=data.shape[0]//64,shuffle=True)
fig,(acc,loss)=plt.subplots(2,1)





acc.set_title('Accuracy vs Epochs')

acc.plot(np.arange(1,len(history.history['accuracy'])+1),history.history['accuracy'],label='Training Accuracy')

acc.plot(np.arange(1,len(history.history['val_accuracy'])+1),history.history['val_accuracy'],label='Validation Accuracy')

acc.set_xlabel('Epochs')

acc.set_ylabel('Accuracy')

acc.set_xticks(np.arange(1,len(history.history['accuracy'])+1))

acc.legend(loc='best')



loss.set_title('Loss vs Epochs')

loss.plot(np.arange(1,len(history.history['loss'])+1),history.history['loss'],label='Training loss')

loss.plot(np.arange(1,len(history.history['val_loss'])+1),history.history['val_loss'],label='Validation loss')

loss.set_xlabel('Epochs')

loss.set_ylabel('Loss')

loss.set_xticks(np.arange(1,len(history.history['loss'])+1))

loss.legend(loc='best')



fig.set_figheight(20)

fig.set_figwidth(20)

fig.show()
data=np.concatenate((data,x_val))

labels=np.concatenate((labels,y_val))
print('Data shape:',data.shape)
generator=aug_data.flow(data,labels,batch_size=64)
model=create_model()
history=model.fit_generator(generator,epochs=50,validation_data=(x_val,y_val),steps_per_epoch=data.shape[0]//64,shuffle=True)
fig,(acc,loss)=plt.subplots(2,1)





acc.set_title('Accuracy vs Epochs')

acc.plot(np.arange(1,len(history.history['accuracy'])+1),history.history['accuracy'],label='Total Accuracy')

acc.plot(np.arange(1,len(history.history['val_accuracy'])+1),history.history['val_accuracy'],label='Validation Accuracy')

acc.set_xlabel('Epochs')

acc.set_ylabel('Accuracy')

acc.set_xticks(np.arange(1,len(history.history['accuracy'])+1))

acc.legend(loc='best')



loss.set_title('Loss vs Epochs')

loss.plot(np.arange(1,len(history.history['loss'])+1),history.history['loss'],label='Total loss')

loss.plot(np.arange(1,len(history.history['val_loss'])+1),history.history['val_loss'],label='Validation loss')

loss.set_xlabel('Epochs')

loss.set_ylabel('Loss')

loss.set_xticks(np.arange(1,len(history.history['loss'])+1))

loss.legend(loc='best')



fig.set_figheight(20)

fig.set_figwidth(20)

fig.show()
print(classification_report(np.argmax(y_val,1),np.argmax(model.predict(x_val),1)))
test_id=test['id'].values
test.drop('id',axis=1,inplace=True)
x_test=test.values.reshape(-1,28,28,1)/255
pred=np.argmax(model.predict(x=x_test),1)
def submit(test_id,pred):

    df=np.concatenate((test_id.reshape(-1,1),pred.reshape(-1,1)),axis=1)

    df=pd.DataFrame(df,columns=['id','label'])

    df['id']=df['id'].astype('int32')

    df.to_csv('Submission.csv',index=False)
submit(test_id,pred)
FileLink('Submission.csv')