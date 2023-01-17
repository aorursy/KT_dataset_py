import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
train.shape, test.shape
X=train.drop(['label'],axis=1)
y=train['label']


# check the shape
X.shape, y.shape
X_test = test.drop(['label'],axis=1)
y_test = test['label']

# check the shape 
X_test.shape,y_test.shape
y.value_counts()
y_test.value_counts()
plt.subplots(figsize = (10,8))
plt.title('Counts in numbers to their labels ')
sns.countplot(x=y, data=train)
plt.show()
plt.subplots(figsize = (10,8))
plt.title('Counts in numbers to their labels ')
sns.countplot(x=y_test, data=train)
plt.show()
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=99)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
X_train = X_train.values.astype('float32')
X_val = X_val.values.astype('float32')
X_test = X_test.values.astype('float32')
X_train
X_val
X_test
X_train = X_train.reshape(X_train.shape[0],28,28)
X_val = X_val.reshape(X_val.shape[0],28,28)
X_test = X_test.reshape(X_test.shape[0],28,28)
X_train.shape, X_val.shape, X_test.shape, 
X_train.max(),X_train.min()
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

# cheking the value range 
X_test.max(), X_test.min()
# import the model
model=Sequential()
# import the layers
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# checking the summary
model.summary()
%%time
history=model.fit(X_train,y_train, batch_size=64, epochs=10, validation_data=(X_val,y_val))
accuracy,loss=model.evaluate(X_test,y_test)
accuracy,loss
# plot the figure now
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()
# plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
y_pred
class_names=[ 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names
mat=confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names= class_names,show_normed=True, figsize=(7,7))
