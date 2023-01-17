import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten ,BatchNormalization , MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
ds_train=pd.read_csv("../input/digit-recognizer/train.csv")
ds_test=pd.read_csv("../input/digit-recognizer/test.csv")
print("Train dataset :",ds_train.shape)
#print("Test dataset :",ds_test.shape)
X=ds_train.drop(['label'],axis=1)
y=ds_train['label']
print("done!")
sns.countplot(y)
X=X/255.0
ds_test=ds_test/255.0
X = X.values.reshape(-1,28,28,1)
ds_test = ds_test.values.reshape(-1,28,28,1)
fig,axs=plt.subplots(1,5,figsize=(20,5))
fig.tight_layout()

for i in range(5):
    axs[i].imshow(X[i].reshape(28,28))
    axs[i].axis('off')
    axs[i].set_title(y[i])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state=4)
len(X_train) #we will set batch size to 56 and step per epouch to 660
dataGen= ImageDataGenerator(width_shift_range=0.1,   
                            height_shift_range=0.1,
                            zoom_range=0.2,  
                            shear_range=0.1, 
                            rotation_range=10)  
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)
fig,axs=plt.subplots(1,5,figsize=(20,5))
fig.tight_layout()

for i in range(5):
    axs[i].imshow(X_batch[i].reshape(28,28))
    axs[i].axis('off')
    axs[i].set_title(y_batch[i])
plt.show()
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
model = Sequential()

#First
model.add(Conv2D(filters = 64, kernel_size = (3,3) ,activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 56, kernel_size = (3,3),activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Second
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(Conv2D(filters = 48, kernel_size = (3,3),activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Third
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.4))

#Output
model.add(Dense(10, activation = "softmax"))


model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
#you can set more than 10 epochs to get more accuracy 
history = model.fit_generator(dataGen.flow(X_train,y_train, batch_size=56),
                              epochs = 10, validation_data = (X_test,y_test),
                              verbose = 2, steps_per_epoch=660)


# For 10 epochs we get 
#  loss: 0.0614 
#  accuracy: 0.9838 
#  val_loss: 0.0364 
#  val_accuracy: 0.9921
plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.legend(['training','validation'])
ax1.set_title('loss')
ax1.set_xlabel('epoch')

ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.legend(['training','validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('epoch')



score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
results = model.predict(ds_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("CNN_Digit_Recognizer.csv",index=False)