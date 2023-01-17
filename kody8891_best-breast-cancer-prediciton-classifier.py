# default libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go



# ml modeling

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score 



# dl modeling

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, BatchNormalization

import keras

from keras.layers import Input, Dense

from keras.optimizers import SGD

df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

df.head()
# explore column names

df.columns
# find na values

df.isna().sum()
# drop unnamed and iq because they arent relevant to prediction

df.drop(["id","Unnamed: 32"],axis = 1,inplace = True)



#coding M and B to be 1 and 0 for classification

df["diagnosis"].replace("M",0,inplace = True)

df["diagnosis"].replace("B",1,inplace = True)
plt.figure(figsize=(25, 12))

sns.heatmap(df.corr(), annot=True,cmap='Reds')

plt.show()
M = data[(df['diagnosis'] != 0)]

B = data[(df['diagnosis'] == 0)]



trace = go.Pie(labels = ['benign','malignant'], values = data['diagnosis'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['blue', 'red'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Diagnosis')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
# train test split

X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)

Log = LogisticRegression()

Log.fit(x_train,y_train)

print("Accuracy:{}".format(Log.score(x_test,y_test)))
# Confusion Matrix

y_true = y_test 

y_pred = Log.predict(x_test) #Predict data for eveluating 

cm = confusion_matrix(y_true,y_pred)



# heat map plot

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Reds')
X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
svclassifier = SVC(kernel = 'linear', random_state = 0)

svclassifier.fit(X_train, y_train)
accuracy = svclassifier.score(X_test, y_test)

print("Accuracy:{}".format(accuracy))
# Confusion Matrix

y_true = y_test 

y_pred = svclassifier.predict(X_test) #Predict data for eveluating 

cm = confusion_matrix(y_true,y_pred)



# heat map plot

f,ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True,linewidth = 1,fmt =".0f",ax = ax,cmap='Reds')
X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values



# Encoding categorical data

labelencoder_X_1 = LabelEncoder()

y = labelencoder_X_1.fit_transform(y)



# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



#Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
model = Sequential()

model.add(Dense(256, input_dim=30))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(128))

model.add(Activation('softmax'))

model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
model.compile(loss='mean_squared_error', optimizer='adam',

              metrics = ["accuracy"])
# Predicting the Test set results

y_pred = model.predict(

    X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)



sns.heatmap(cm,annot=True,cmap='Reds')

plt.savefig('h.png')