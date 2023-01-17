#importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import itertools
# Reading the data 

data=pd.read_csv("../input/diabetes.csv")
#displaying the data

data.head()
#describing the data

data.describe()
# finding the null values in data

data.isnull().sum()
# data type used

data.info()

#Visualising the Count of Target Variable

sns.countplot(x='Outcome',data=data)

plt.show()
# Count of pregnancies

sns.countplot(data['Pregnancies'])

plt.show()

# we can see the outliers in the data 
dat=data.loc[data['Pregnancies']>10]

dat
sns.countplot(dat['Pregnancies'])
#CLosure look at glucose

plt.hist(data['Glucose'],bins=10,edgecolor='black')

plt.xlabel('Range')

plt.ylabel("Glucose Level")

plt.title('Closure look at Glucose')

plt.show()
# Outliers in glucose

dt=data.loc[data['Glucose']<25]

dt.shape[0]
print("Before concatenation rows :" ,dat.shape[0])

dat=dat.append(dt)

print("After concatenation rows :" ,dat.shape[0])
# Closure look at BloodPressure

plt.hist(data['BloodPressure'],bins=10,edgecolor='black')

plt.xlabel('Range')

plt.ylabel("BloodPressure")

plt.title('Closure look at BloodPressure')

plt.show()
dt=data.loc[data['BloodPressure']<25]

dt.shape[0]
print("Before concatenation rows :" ,dat.shape[0])

dat=dat.append(dt)

print("After concatenation rows :" ,dat.shape[0])
# closure look at Skinthicknes

plt.hist(data['SkinThickness'],bins=10,edgecolor='black')

plt.xlabel('Range')

plt.ylabel("SkinThickness Level")

plt.title('Closure look at Skinthickness')

plt.show()
dt=data.loc[data['SkinThickness']>80]

dt.shape[0]
print("Before concatenation rows :" ,dat.shape[0])

dat=dat.append(dt)

print("After concatenation rows :" ,dat.shape[0])
# closure look at 

plt.hist(data['Insulin'],bins=10,edgecolor='black')

plt.xlabel('Range')

plt.ylabel("Insulin")

plt.title('Closure look at Insulin')

plt.show()
dt=data.loc[data['Insulin']>600]

dt.shape[0]
print("Before concatenation rows :" ,dat.shape[0])

dat=dat.append(dt)

print("After concatenation rows :" ,dat.shape[0])
# closure look at BMI

plt.hist(data['BMI'],bins=10,edgecolor='black')

plt.xlabel('Range')

plt.ylabel("BMI Level")

plt.title('Closure look at BMI')

plt.show()
dt=data.loc[data['BMI']<10]

dt.shape[0]
print("Before concatenation rows :" ,dat.shape[0])

dat=dat.append(dt)

print("After concatenation rows :" ,dat.shape[0])
# closure look at Age

plt.hist(data['Age'],bins=10,edgecolor='black')

plt.xlabel('Range')

plt.ylabel("Age Level")

plt.title('Closure look at Age')

plt.show()
# closure look at only positive cases of Diabetes with Outliers

diab1=data[data['Outcome']==1]

columns=data.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    diab1[i].hist(bins=10,edgecolor='black')

    plt.title(i)

plt.show()
data1=data.copy()
data1=data1.loc[data1['Pregnancies']<=10]

data1=data1.loc[data1['Glucose']>=25]

data1=data1.loc[data1['BloodPressure']>=25]

data1=data1.loc[data1['SkinThickness']<=80]

data1=data1.loc[data1['Insulin']<=600]

dt=data.loc[data['BMI']>10]
data1.shape[0]
# closure look at only positive cases of Diabetes without Outliers

diab1=data1[data1['Outcome']==1]

columns=data1.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    diab1[i].hist(bins=10,edgecolor='black',color='lightblue')

    plt.title(i)

plt.show()
# Visualising the preprocessed data to check biasing of the class in each attribute

sns.pairplot(data=data1,hue='Outcome',diag_kind='kde')

plt.show()
# Correlation of the preprocessed data

sns.heatmap(data1.corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
#importing models through sklearn

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import KFold
outcome=data1['Outcome']

data=data1[data1.columns[:8]]

train,test=train_test_split(data1,test_size=0.3,random_state=0,stratify=data1['Outcome'])

train_X=train[train.columns[:8]]

test_X=test[test.columns[:8]]

train_Y=train['Outcome']

test_Y=test['Outcome']
print("Train data :",train_X.shape,"\nTrain Output :",train_Y.shape,"\nTest Data :",test_X.shape,"\nTest Output :",test_Y.shape)
model = svm.SVC()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y)*100)
model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_Y)*100)
model = KNeighborsClassifier()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y)*100)
dat=data1.copy()

del dat['Outcome']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(dat,data1['Outcome'],test_size=0.4,random_state=42, stratify=data1['Outcome'])
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold



test_scores = []

train_scores = []

validation_scores = []

X_train_values = X_train.values

y_train_values = y_train.values



## cross validation with KFold algorithm

kfold = KFold(5, shuffle=True, random_state=42)

for i in range(1,15):



    knn = KNeighborsClassifier(i)



    tr_scores = []

    ts_scores = []

    for train_ix, test_ix in kfold.split(X_train_values):

        # define train/test X/y

        X_train_fold, y_train_fold = X_train_values[train_ix],y_train_values[train_ix]

        X_test_fold, y_test_fold = X_train_values[test_ix], y_train_values[test_ix]

        knn.fit(X_train_fold,y_train_fold)

        ts_scores.append(knn.score(X_test_fold,y_test_fold))

        tr_scores.append(knn.score(X_train_fold,y_train_fold))

    validation_scores.append(np.mean(ts_scores))

    train_scores.append(np.mean(tr_scores))

    test_scores.append(knn.score(X_test,y_test))
plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')

p = sns.lineplot(range(1,15),validation_scores,marker='v',label='Validation Score')

y_pred = knn.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')