#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools
#reading data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#train data x and y
x_train = train.drop(labels = ["label"], axis = 1)
y_train = train["label"]
#normalizing and reshaping data
x_train = x_train / 255.0 #pixel values vary between 0 and 255
test = test / 255.0
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#one hot encoding for y
y_train = to_categorical(y_train)
#train test split
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
#defining the model
model = Sequential()
# 2 conv2d layers
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4)) 

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten()) #flattening the data into a 1D vector, before the fully connected layers
model.add(Dense(256, activation = "relu"))
model.add(Dense(10, activation = 'softmax'))


#optimizing the model: using Adam omptimizer
optim=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#compiling the model
model.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])
#training the model
model.fit(X_train, Y_train, epochs=10, batch_size=128)
# Predicting the values from the validation dataset
Y_pred = model.predict(X_val)
# converting one hot vectors to actual values 
Y_pred = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(Y_val,axis = 1) 
# computing the confusion matrix
cm = confusion_matrix(Y_true, Y_pred) 

#plotting the confusion matrix
plt.imshow(cm)
plt.title('confusion matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10), rotation=45)
plt.yticks(tick_marks, range(10))

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
#evaluating the model: 0.9926 accuracy
loss_and_metrics = model.evaluate(X_val, Y_val, batch_size=128)
loss_and_metrics
#predicting on the test set
y_pred = model.predict(test, batch_size = 128)
pred = np.argmax(y_pred,axis = 1)

pred = pd.Series(pred,name="Label")
#preparing submission
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)

submission.to_csv("submission.csv",index=False)