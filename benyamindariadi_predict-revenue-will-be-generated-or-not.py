import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



df=pd.read_csv('/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv')
df.info()

sns.countplot(x='Revenue',data=df)

#the data is skew!
df
df.Weekend = df.Weekend.replace({True: 1, False: 0})

df.Revenue = df.Revenue.replace({True: 1, False: 0})
dummies= pd.get_dummies(df['VisitorType'],drop_first=True) 

df=pd.concat([df.drop('VisitorType', axis=1), dummies],axis=1) 

df=df.drop('Other',axis=1)
df.Month.unique()
df['Month'] = df['Month'].map({'Feb':2,'Mar':3,'May':5,'Oct':10,'June':6,'Jul':7,'Aug':8,'Nov':11,'Sep':9,'Dec':12})
df.info()
df = df.dropna()
df.corr()

plt.figure(figsize=(12,7))

sns.heatmap(df.corr(),annot=True,cmap='viridis')

sns.distplot(df['Revenue'],kde=False,bins=40)
df.corr()['Revenue'].sort_values().plot(kind='bar')
x = df.drop('Revenue',axis=1).values

y = df['Revenue'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

scores=[]

for i in range(1,25):

  tree=DecisionTreeClassifier(max_depth = i) 

  tree.fit(x_train, y_train) 

  scores.append(tree.score(x_test,y_test)) 

plt.plot(range(1,25),scores) 

plt.show()

tree=DecisionTreeClassifier(max_depth =5) 

tree.fit(x_train, y_train) 

tree.score(x_test,y_test)
predict=tree.predict([x_test[0]])

predict
y_test[0]
predictions = tree.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))

from sklearn.ensemble import RandomForestClassifier
scores=[]

for i in (np.arange(100,2000,100)):

  classifier = RandomForestClassifier(n_estimators =i, max_depth=5, random_state =101) 

  classifier.fit(x_train, y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.arange(100,2000,100),scores) 

plt.show()

classifier = RandomForestClassifier(n_estimators =1000,max_depth=5, random_state =101)

classifier.fit(x_train, y_train)

print(classifier.score(x_test,y_test))
classifier = RandomForestClassifier(n_estimators =1000,max_depth=5, random_state =101)

classifier.fit(x_train, y_train)

print(classifier.score(x_test,y_test))
from sklearn.svm import SVC



scores=[]

for i in (np.arange(0.01,1,0.02)):

  classifier = SVC(kernel = 'linear', C = i)

  classifier.fit(x_train,y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.arange(0.01,1,0.02),scores) 

plt.show()
scores=[]

for i in (np.arange(0.01,1,0.02)):

  classifier = SVC(kernel = 'rbf', gamma=i, C = i)

  classifier.fit(x_train,y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.arange(0.01,1,0.02),scores) 

plt.show()
classifier = SVC(kernel = 'linear', C = 1)

classifier.fit(x_train,y_train) 

classifier.score(x_test,y_test)
predictions = classifier.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout



model = Sequential()

model.add(Dense(units=35,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=35,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])



early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)



model.fit(x_train,y_train,epochs=50,validation_data=(x_test, y_test), verbose=1, callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)



losses[['accuracy','val_accuracy']].plot()

losses[['loss','val_loss']].plot()

print(model.metrics_names)

print(model.evaluate(x_test,y_test,verbose=0))
from sklearn.metrics import classification_report,confusion_matrix



predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))