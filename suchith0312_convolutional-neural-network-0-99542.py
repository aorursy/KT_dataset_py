# importing the basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

print("Traning Set")

print(train.shape)

print("Test set")

print(test.shape)
train.head(10)
test.head(10)
train.isnull().any().describe()
test.isnull().any().describe()
X=train.iloc[:,1:].values

y=train.iloc[:,0].values



X_test=test.iloc[:,:].values



print("Train data shape : (%d,%d)"% X.shape)

print("Train Labels : (%d,)"% y.shape)

print("Test Data shape : (%d,%d)"% X_test.shape)
def show_image(image, shape, label="", cmp=None):

    img = np.reshape(image,shape)

    plt.imshow(img,cmap=cmp, interpolation='none')

    plt.title(label)
%matplotlib inline

plt.figure(figsize=(12,10))



z, x = 5,10

for i in range(0,(z*x)):

    plt.subplot(z, x, i+1)

    k = np.random.randint(0,X.shape[0],1)[0]

    show_image(X[k,:],(28,28), y[k], cmp="gist_gray")

plt.show()
X=X/255

X_test=X_test/255

print("min value :%d"% np.min(X))

print("max value :%d"% np.max(X))
X=X.reshape(-1,28,28,1)

X_test=X_test.reshape(-1,28,28,1)



print("Train data shape : (%d,%d,%d,%d)"% X.shape)

print("Test Data shape : (%d,%d,%d,%d)"% X_test.shape)
# first we will print y's

print(y[0:10])
from sklearn.preprocessing import OneHotEncoder

x=y.reshape(y.size,1)

onehotencoder=OneHotEncoder(categorical_features=[0])

y=onehotencoder.fit_transform(x).toarray().astype(int)

print("SHAPE : (%d,%d)\n" %y.shape)

print(y[0:10,:])
from sklearn.model_selection import train_test_split

X_train,X_cross,y_train,y_cross=train_test_split(X,y,test_size=0.1,random_state=42)



print("Train Size (%d,%d,%d,%d) \n"% X_train.shape)

print("Validation Size (%d,%d,%d,%d) \n"% X_cross.shape)

print("Train Label Size (%d,%d) \n"% y_train.shape)

print("Validation Label Size (%d,%d) \n"% y_cross.shape)
plt.imshow(X_train[0][:,:,0])
plt.imshow(X_test[0][:,:,0])

# print(X_test[0][:,:,0].shape)
# importing the libraries

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout

from keras.preprocessing.image  import ImageDataGenerator
# Initialising the CNN

model=Sequential()



# Convolution layer

model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))



# Max pooling Layer

model.add(MaxPool2D(pool_size=(2,2)))



# adding another Convolution Layer

model.add(Conv2D(64,(3,3),activation='relu'))



# adding Max pooling layer

model.add(MaxPool2D(pool_size=(2,2)))



# model dropout layer

model.add(Dropout(0.25))



# Flattening

model.add(Flatten())



# Full connection 

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))



# summary

model.summary()
# compiling the CNN

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



model.summary()
datagen=ImageDataGenerator(rotation_range=10,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)



datagen.fit(X_train)
epoch=2

batch=100

sp_epoch=X_train.shape[0]
# h=model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch),epochs=epoch,validation_data=(X_cross,y_cross),steps_per_epoch=sp_epoch)
# y_pred=model.predict_classes(X_test)

# print(y_pred.shape)
# %matplotlib inline

# plt.figure(figsize=(12,10))



# z, x = 5,10

# for i in range(0,(z*x)):

#     plt.subplot(z, x, i+1)

#     k = np.random.randint(0,X_test.shape[0],1)[0]

#     show_image(X_test[k,:],(28,28), y_pred[k], cmp="gist_gray")

# plt.show()
# imageid=np.linspace(1,28000,28000).astype(int)

# print(imageid,imageid.shape,type(imageid))
# ans=pd.DataFrame({

#     "ImageId":imageid,

#     "Label":y_pred

# })

# ans.head(10)
# ans.to_csv("CNN.csv",index=False)                

# print("Done")   