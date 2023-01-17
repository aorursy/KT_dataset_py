import pandas as pd
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import  Dense, Dropout, Activation , Concatenate, Input , BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load data
data = pd.read_csv('/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv')
data.head()
data.sample(15)
data.info()
for cols in data.columns:
    data[cols] = data[cols].astype('category')
data.info()
for cols in  data.columns:
    le = preprocessing.LabelEncoder()
    le.fit(data[cols])
    data[cols] = le.transform(data[cols])
    
data.sample(10)    
y = data['Gender']
X = data.drop('Gender', axis=1)
y.head()
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
X_train.head()
y_train.head()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[:5]
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=7)
model = Sequential()
model.add(Dense(512, activation='relu',input_dim=X_train.shape[1]))
model.add(Dropout(rate = 0.2))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))
plot_model(model, show_shapes=True)
model.summary()
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=75, validation_split=0.2, verbose=1,callbacks=[es])
val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(val_loss)
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Val error','Train error'], loc='upper right')
plt.savefig('plot_error.png')
plt.show()
val_accuracy = history.history['val_accuracy']
accuracy = history.history['accuracy']

plt.plot(val_accuracy)
plt.plot(accuracy)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['Val accuracy','Train accuracy'], loc='upper right')
plt.savefig( 'plot_accuracy.png')
plt.show()
pred = model.predict(X_test)
pred = pd.DataFrame(pred , index = X_test.index)
pred['Perdiction'] = pred.idxmax(axis=1)
pred.head(5)
testy = pd.DataFrame(y_test ,index=X_test.index)
testy['label'] = testy.idxmax(axis=1)
testy.head()
common = pred[["Perdiction"]].merge(testy[['label']],left_on = pred.index , right_on = testy.index)
common = common[["Perdiction",'label']]
common.head()
mat = confusion_matrix(common['label'],common['Perdiction'])
sn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
