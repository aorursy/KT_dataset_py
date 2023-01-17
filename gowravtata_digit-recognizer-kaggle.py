#importing packages
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
X_train = train.iloc[:, 1:785].values
Y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 0:784].values
#Looking the shape of the data 
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
X_train,X_test=X_train/255.0,X_test/255.0
#reshaping to format  which CNN expects (batch,height,width,channels)
X_train = X_train.reshape(42000, 28, 28, 1)
X_test = X_test.reshape(28000, 28, 28, 1)
Y_train = to_categorical(Y_train, 10)
from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = 0.2, random_state=44)

print(X_train.shape)
print(X_validation.shape)
print(Y_train.shape)
print(Y_validation.shape)

#Initiating the model
model=Sequential()
#Convolution Layers
model.add(Conv2D(32, kernel_size = (3,3), input_shape= (28,28,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
#fully connected layers
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'sigmoid'))
#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
training = model.fit(X_train,Y_train,epochs=20)
loss , accuracy = model.evaluate(X_validation, Y_validation)
print("Loss : ",loss, "Accuracy : ", accuracy)
predicted_classes = model.predict_classes(X_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)), "Label": predicted_classes})

submissions.to_csv("submission.csv", index = False, header = True)
test_image=X_test[300]
plt.imshow(X_test[300].reshape(28,28))
#test_image=np.asarray(test_image)
test_image=test_image.reshape(1,28,28,1)
result=model.predict(test_image)
(np.round(result)).argmax()
plt.imshow(X_test[400].reshape(28,28))
(np.round(model.predict(X_test[400].reshape(1,28,28,1)))).argmax()