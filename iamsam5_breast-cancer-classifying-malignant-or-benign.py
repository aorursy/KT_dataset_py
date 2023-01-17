import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline
data=pd.read_csv('../input/breast-cancer-dataset-uci-ml/cancer_classification.csv')
data.head()
data.isnull().sum()
data.describe()
sns.countplot(x='benign_0__mal_1',data=data)
sns.heatmap(data.corr())
X=data.drop('benign_0__mal_1',axis=1)
y=data['benign_0__mal_1']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

X_train= scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

model=Sequential()

model.add(Dense(30,activation='relu'))

model.add(Dense(15,activation='relu'))



model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min')
model.fit(X_train,y_train,epochs=250,validation_data=(X_test,y_test),callbacks=[early_stop])
model.history.history
loss=pd.DataFrame(model.history.history)
loss
loss.plot()
predictions=model.predict_classes(X_test)
predictions
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))