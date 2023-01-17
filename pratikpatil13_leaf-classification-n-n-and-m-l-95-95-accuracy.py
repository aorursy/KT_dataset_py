# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/leaf-classification/train.csv.zip')
test = pd.read_csv('/kaggle/input/leaf-classification/test.csv.zip')
sample_sub=pd.read_csv("/kaggle/input/leaf-classification/sample_submission.csv.zip")
train
plt.figure(figsize=(10,6))
sns.countplot(train["species"],palette="muted")
train["species"].value_counts()
train.isnull().sum()
train.corr()
Y=train.pop('species')
train.pop('id')
X=train.values
Y1=Y
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
Y= labelencoder_y.fit_transform(Y)
Y1=Y1.values.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
Y1 = ohe.fit_transform(Y1).toarray()
Y1.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,random_state=0)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred_svc =svc.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_svc))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
print(classification_report(y_test, pred_knn))
from sklearn.model_selection import train_test_split                 #importing train_test_split from sklearn to split data
x_train,x_test,y_train,y_test=train_test_split(X,Y1,shuffle=True,test_size=0.10,random_state=90) #split the data in 80:20 ratio 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=90)
#Neural Network Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu,softmax
y_train.shape
model = Sequential()
model.add(Dense(256, input_dim=192, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1600, activation='relu'))
model.add(Dense(99, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
checkpointer=ModelCheckpoint(filepath='Convolutional.hdf5',verbose=1,save_best_only=True)
history = model.fit(x_train, y_train, epochs=60, batch_size=64,validation_data=(x_val,y_val))
score=model.evaluate(x_test,y_test,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)
score=model.evaluate(x_train,y_train,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
