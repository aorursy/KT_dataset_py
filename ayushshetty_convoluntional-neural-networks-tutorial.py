import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train_df=pd.read_csv('../input/fashion-mnist_train.csv',sep=',')
test_df=pd.read_csv('../input/fashion-mnist_test.csv',sep=',')
train_df.head()
test_df.head()
train_df.shape
test_df.shape
training=np.array(train_df,dtype='float32')
testing=np.array(test_df,dtype='float32')
import random
i=random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28))
label=training[i,0]
label
X_train=training[:,1:]/255
'''Normalization ((X-Xmin)/(Xmax-Xmin))=(X-0)/(255-0)
CNN works better when the data is normalized.Pixel takes values between 0 to 255'''
Y_train=training[:,0]
X_test=testing[:,1:]/255
Y_test=testing[:,0]
from sklearn.model_selection import train_test_split
X_train,X_validate,Y_train,Y_validate=train_test_split(X_train,Y_train,test_size=0.2,random_state=10)
X_train=X_train.reshape(X_train.shape[0],*(28,28,1))
X_test=X_test.reshape(X_test.shape[0],*(28,28,1))
X_validate=X_validate.reshape(X_validate.shape[0],*(28,28,1))
#validation dataset to help the model generalize
#reshape data to be in the form of 28X28
X_validate.shape
import keras
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model=Sequential()
cnn_model.add(Conv2D(32,3,3,input_shape=(28,28,1),activation='relu'))
#Conv2D=convolutional layer
#32 indicates the number of kernels,with 3X3 dimensions
#activation function=RELU
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
#pooling size=2X2 matrix
cnn_model.add(Flatten())
#Flattening 
cnn_model.add(Dense(output_dim=32,activation='relu'))
#dense layer ,hidden layer
cnn_model.add(Dense(output_dim=10,activation='sigmoid'))

cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
#Adam optimizer,loss= categorial cross entropy we use cross categorial because we are simply categorizing our dataset
epochs=40
cnn_model.fit(X_train,Y_train,batch_size=512,nb_epoch=epochs,verbose=1,validation_data=(X_validate,Y_validate))
#epochs=50, how many timeswe are going to present our dataset and updating the weights
evaluation=cnn_model.evaluate(X_test,Y_test)
print('Test Accuracy::{:.3f}'.format(evaluation[1]))

predicted_classes=cnn_model.predict_classes(X_test)
predicted_classes
L=5
W=5
fig,axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()

for i in np.arange(0,L * W):
   axes[i].imshow(X_test[i].reshape(28,28))
   axes[i].set_title("Prediction class={:0.1f}\n,true class={:0.1f}".format(predicted_classes[i],Y_test[i]))
   axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
