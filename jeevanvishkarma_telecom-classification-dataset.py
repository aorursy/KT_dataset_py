import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/telecom churn.csv")
data.head()
data.state.value_counts().plot(figsize=(10,10),kind = 'bar')
df = data.iloc[ : , 4:]
df.head()
df["churn"] = df["churn"].apply(lambda x : 1 if x == True else 0)
df.head()
sns.pairplot(df)
from sklearn.model_selection import train_test_split
x= df.drop(["churn"], axis = 1)
x.head()
y = df['churn']
x["voice mail plan"] = x["voice mail plan"].map({'no' : 0, 'yes' : 1})
x["international plan"] = x["international plan"].apply(lambda x : 0 if x == 'no' else 1)
x.isnull().sum()
x = x.fillna(method = 'ffill')
x.isnull().sum()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 3)

print(x_train.shape)

print(y_train.shape)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier
forrest = RandomForestClassifier()
forrest.fit(x_train,y_train)
forrest.score(x_test,y_test)
y_pred  = forrest.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
sns.heatmap(cm, annot = True)
x_c= df.drop(["churn"], axis = 1)
y_c = df['churn']
x_c = x_c.fillna(method = 'ffill')
x_c.head()
x_c.shape
x_c['international plan'] = x_c['international plan'].apply(lambda x:0 if x == 'no' else 1)
x_c['voice mail plan'] = x_c['voice mail plan'].apply(lambda x: 0 if x == "no" else 1)
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation = 'sigmoid'))
x_ctrain,x_ctest,y_ctrain,y_ctest = train_test_split(x_c,y_c, test_size = .25, random_state = 3)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_ctrain,y_ctrain, batch_size = 10, epochs = 100)

y_predict =  classifier.predict(x_ctest)
y_predict = (y_predict > 0.5)