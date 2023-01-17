#import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
#fetch data

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df.head()
df.columns
df.isnull().sum()
df.drop('Unnamed: 32',axis=1,inplace=True)
df.describe().transpose()
df['diagnosis'].value_counts()
plt.figure(figsize=(12,6))

sns.countplot(x='diagnosis',data=df)
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x=='M' else 0)
plt.figure(figsize=(12,6))

df.corr()['diagnosis'].sort_values().plot(kind='bar')
plt.figure(figsize=(12,6))

sns.heatmap(df.corr())
X = df.drop('diagnosis',axis=1)

y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train.shape
model = Sequential()

model.add(Dense(30,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(15,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)

losses.head()
plt.figure(figsize=(12,6))

plt.plot(losses)

plt.show()
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))