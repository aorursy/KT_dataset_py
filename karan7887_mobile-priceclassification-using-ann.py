# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X_train = pd.read_csv('../input/mobile-price-classification/train.csv')
X_test = pd.read_csv('../input/mobile-price-classification/test.csv')
X_train.info()

X_test.drop('id',inplace=True,axis=1)
X_test.info()
print(X_train.shape,X_test.shape)
X_test.describe()
print(X_train.shape,X_test.shape)
Y_train = X_train['price_range']

rel = X_train.corr()

X_train.drop('price_range',axis=1,inplace=True)
rel
import seaborn as sns
rel.shape
sns.heatmap(rel)
# X_train = X_train.drop('price_range',axis=1)
rel = abs(rel)
rel = rel['price_range'].sort_values(ascending=False)
rel = rel[1:]
rel
useful_feature = rel[0:9]
useful_feature = useful_feature.index
useful_feature
X_test.columns
X_train = X_train[useful_feature]
X_test = X_test[useful_feature]
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size= 0.3,random_state=101)
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

import xgboost as xgb

lr = GradientBoostingClassifier()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(str(accuracy_score(y_test,pred)*100)+ "% with gradient booster classifier")


lr = xgb.XGBClassifier()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(str(accuracy_score(y_test,pred)*100) + "% with xgbooster classifier")

lr = RandomForestClassifier()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
print(str(accuracy_score(y_test,pred)*100) + "% with random forest classifier")
print(y_train.shape,y_test.shape)
print(x_train.shape,x_test.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(y_train.shape,y_test.shape)
print(x_train.shape,x_test.shape)
from keras.models import Sequential
from keras.layers import Dense,Dropout

from sklearn.preprocessing import OneHotEncoder

hot = OneHotEncoder()
y_train = hot.fit_transform(y_train).toarray()
y_test = hot.fit_transform(y_test).toarray()
print(y_train.shape,y_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape)
model = Sequential()
model.add(Dense(8,input_shape = (x_train.shape[1],),activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(6,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(12,activation='relu'))

model.add(Dense(4,activation='softmax'))

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
model.summary()
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=150,validation_data = (x_test,y_test),batch_size=64,shuffle=True)
model.evaluate(x_test,y_test)
import matplotlib.pyplot as plt
hist = hist.history
plt.style.use('seaborn')
plt.plot(hist['accuracy'],label = 'Training Accuracy')
plt.plot(hist['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.style.use('seaborn')
plt.plot(hist['loss'],label = 'Training Loss')
plt.plot(hist['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()