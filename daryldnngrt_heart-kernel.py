import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.info()
df.describe()
plt.figure(figsize=(8,5))

sns.countplot(x='target',data=df)
plt.figure(figsize=(12,5))

sns.heatmap(df.corr(),annot=True)

plt.ylim(0, 14)

plt.xlim(0, None)
sns.boxplot(x='target',y='thalach',data=df)
plt.figure(figsize=(12,6))

df.corr()['target'][:-1].sort_values().plot(kind='bar')
from sklearn.model_selection import train_test_split
X = df.drop('target',axis=1).values

y = df['target'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()



model.add(Dense(18,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)
model.fit(x=X_train,y=y_train,epochs=300,validation_data=(X_test,y_test),callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses.head()
losses.plot()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))

print('')

print(classification_report(y_test,predictions))