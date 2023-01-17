import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D, ZeroPadding2D, Conv2D
# Input data files are available in the "../input/" directory.
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train.shape, test.shape
train.head()
# test.head()
X=train.drop(['label'], axis=1)
y=train['label']

X.shape, y.shape
# checking manually
y.value_counts()
plt.subplots(figsize=(10,8))
plt.title('Counts of the labels')
sns.countplot(x=y)
plt.show()
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1, random_state=99)
# check the shape now
X_train.shape,X_val.shape,y_train.shape,y_val.shape,test.shape
X_train=X_train.values.astype('float32')
X_val=X_val.values.astype('float32')
test=test.values.astype('float32')
# changing the shape of X_train and y_train and test also
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val=X_val.reshape(X_val.shape[0], 28, 28, 1)
test=test.reshape(test.shape[0] , 28 , 28 , 1)
X_train.shape,X_val.shape,test.shape
# check the maximum values in the dataset
X_train.max(),X_train.min()
X_train=X_train/255
X_val=X_val/255
test=test/255
# check the maximum values in the dataset
X_train.max(),X_train.min()
input_shape=X_train[0].shape
input_shape
model = Sequential()
model.add(Conv2D(100,kernel_size=(3, 3), activation='relu',padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))


# model.add(Conv2D(350, kernel_size=(5, 5),activation='relu',padding='same') )
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))


# model.add(Conv2D(150, kernel_size=(5, 5),activation='relu',padding='same') )
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

model.add(Conv2D(50, kernel_size=(3, 3),activation='relu',padding='same') )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# check the summary[":"]
model.summary()
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
%%time
history=model.fit(X_train, y_train, batch_size=60, epochs=10, verbose=1, validation_data=(X_val,y_val))
# evaluating the model with testing data
loss, accuracy=model.evaluate(X_val,y_val)
loss, accuracy
# plot the figure now
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()
# plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(test)
y_pred
class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names
mat=confusion_matrix(y_val, y_pred[:4200])
plot_confusion_matrix(conf_mat=mat, class_names= class_names,show_normed=True, figsize=(7,7))
# predict results
results = model.predict(test)

results

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)


results
results = pd.Series(results,name="Label")
results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)


submission
submission.to_csv("submission.csv",index=False)
