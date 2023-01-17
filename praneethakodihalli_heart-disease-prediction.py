# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,mean_squared_error,accuracy_score

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.columns=['age','gender','paintype','bp','cholestoral','blood_sugar','electrocardiographic_results','max_heartrate','angina','oldpeak','slope','no_of_vessels','thal','target']
df['slope'].value_counts()
df.head(10)
df.dtypes
df.isnull().sum()
sns.countplot(x='target', data=df)
df['target'].value_counts()
sns.countplot(x='gender', data=df)
df['gender'].value_counts()
sns.countplot(x='paintype', data=df)
df['paintype'].value_counts()
sns.countplot(x='thal', data=df)
df['thal'].value_counts()
sns.countplot(x='slope', data=df)
df['slope'].value_counts()
df.groupby(['gender', 'target']).size().reset_index().pivot(columns='target', index='gender', values=0).plot(kind='bar', stacked=True)
sns.scatterplot(x="bp", y="max_heartrate", data=df)
ax=sns.scatterplot(x="cholestoral", y="max_heartrate", data=df)

ax.set(xticks=np.arange(0, 500, 200),

      yticks=np.arange(100, 300, 100))
sns.scatterplot(x="bp", y="cholestoral", data=df)
a = pd.get_dummies(df['paintype'], prefix = "paintype")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]

df = pd.concat(frames, axis = 1)

df.head()

df = df.drop(columns = ['paintype', 'thal', 'slope'])

df.head()
X = df.copy().drop("target",axis=1)

y = df["target"]



## Split the data into trainx, testx, trainy, testy with test_size = 0.20 using sklearn

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.20)



## Print the shape of X_train, X_test, y_train, y_test

print(trainx.shape)

print(testx.shape)

print(trainy.shape)

print(testy.shape)
from sklearn.preprocessing import StandardScaler



## Scale the numeric attributes

scaler = StandardScaler()

scaler.fit(trainx.iloc[:,:5])



trainx.iloc[:,:5] = scaler.transform(trainx.iloc[:,:5])

testx.iloc[:,:5] = scaler.transform(testx.iloc[:,:5])
ax=sns.scatterplot(x="cholestoral", y="max_heartrate", data=trainx)
X = trainx

y = trainy

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 

model = LogisticRegression()

model.fit(X , y)

predicted_classes = model.predict(X)

accuracy = accuracy_score(y,predicted_classes)

parameters = model.coef_
print(accuracy)

print(parameters)

print(model)
predicted_classes_test = model.predict(testx)

accuracy = accuracy_score(testy,predicted_classes_test)

print(accuracy)
from sklearn.metrics import confusion_matrix

data = confusion_matrix(testy,predicted_classes_test)

df_cm = pd.DataFrame(data, columns=np.unique(testy), index = np.unique(predicted_classes_test))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 15})# font size
tp=data[1][1]

tn=data[0][0]

fp=data[0][1]

fn=data[1][0]

print('tp=',tp)

print('tn=',tn)

print('fp=',fp)

print('fn=',fn)
print('recall=',tp/(tp+fn))

print('precision=',tp/(tp+fp))

print('accuracy=',(tp+tn)/(tp+tn+fp+fn))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

print(rfc)
rfc.fit(trainx,trainy)

## Predict

rfc_train_predictions = rfc.predict(trainx)

rfc_test_predictions = rfc.predict(testx)



### Train data accuracy

from sklearn.metrics import accuracy_score

print(accuracy_score(trainy,rfc_train_predictions))

      

### Test data accuracy

print(accuracy_score(testy,rfc_test_predictions))
data = confusion_matrix(testy,rfc_test_predictions)

df_cm = pd.DataFrame(data, columns=np.unique(testy), index = np.unique(rfc_test_predictions))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 15})# font size
tp=data[1][1]

tn=data[0][0]

fp=data[0][1]

fn=data[1][0]

print('tp=',tp)

print('tn=',tn)

print('fp=',fp)

print('fn=',fn)
print('recall=',tp/(tp+fn))

print('precision=',tp/(tp+fp))

print('accuracy=',(tp+tn)/(tp+tn+fp+fn))
from sklearn.naive_bayes import GaussianNB



NB = GaussianNB()



NB.fit(X , y)



NB_train_pred = NB.predict(X)

print(accuracy_score(y,NB_train_pred))



NB_test_pred = NB.predict(testx)

print(accuracy_score(testy,NB_test_pred))
data = confusion_matrix(testy,NB_test_pred)

df_cm = pd.DataFrame(data, columns=np.unique(testy), index = np.unique(NB_test_pred))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 15})# font size
tp=data[1][1]

tn=data[0][0]

fp=data[0][1]

fn=data[1][0]

print('tp=',tp)

print('tn=',tn)

print('fp=',fp)

print('fn=',fn)
print('recall=',tp/(tp+fn))

print('precision=',tp/(tp+fp))

print('accuracy=',(tp+tn)/(tp+tn+fp+fn))
knn_classifier = KNeighborsClassifier(algorithm='brute',weights='distance')

params = {'n_neighbors':[1,11,25],'metric':["euclidean",'cityblock']}

grid = GridSearchCV(knn_classifier,param_grid=params,scoring='accuracy',cv=10)

grid.fit(trainx,trainy)

print(grid.best_score_)

print(grid.best_params_)
best_knn = grid.best_estimator_

pred_train = best_knn.predict(trainx) 

pred_test = best_knn.predict(testx)

print("Accuracy on train is:",accuracy_score(trainy,pred_train))

print("Accuracy on test is:",accuracy_score(testy,pred_test))
data = confusion_matrix(testy,pred_test)

df_cm = pd.DataFrame(data, columns=np.unique(testy), index = np.unique(pred_test))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 15})# font size
tp=data[1][1]

tn=data[0][0]

fp=data[0][1]

fn=data[1][0]

print('tp=',tp)

print('tn=',tn)

print('fp=',fp)

print('fn=',fn)
print('recall=',tp/(tp+fn))

print('precision=',tp/(tp+fp))

print('accuracy=',(tp+tn)/(tp+tn+fp+fn))