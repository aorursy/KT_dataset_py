# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_dataset=pd.read_csv('../input/train.csv')
test_dataset=pd.read_csv('../input/test.csv')
test_y=pd.read_csv('../input/gender_submission.csv')
comb=[train_dataset,test_dataset]
train_dataset.head()
train_dataset.Name
sns.countplot(x=train_dataset['Survived'],color='blue')
#Data Description
train_dataset.describe()
#To find any null values
train_dataset.isnull().sum()
train_dataset.corr()
#Correlation between survival and Pclass
print(train_dataset[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean())
 
 #draw plots
g = sns.FacetGrid(train_dataset, col="Survived")
g = g.map(plt.hist, "Pclass",  color="r")
plt.show()
print(train_dataset[['Sex','Survived']].groupby(['Sex'],as_index=False).mean())



print(train_dataset[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean())
bins=np.arange(0,9,1)
g=sns.FacetGrid(train_dataset,col='Survived')
g=g.map(plt.hist,'SibSp',bins=[0,1,2,3,4,5,6,7,8],color="b")
plt.show()
print(train_dataset[['Parch','Survived']].groupby(['Parch'],as_index=False).mean())

g=sns.FacetGrid(train_dataset,col='Survived')
g=g.map(plt.hist,'SibSp',bins=[0,1,2,3,4,5,6],color="b")
plt.show()
print(train_dataset[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean())

g=sns.FacetGrid(train_dataset,row='Embarked',size=2.2,aspect=1.6)
g=g.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
g.add_legend()
for dataset in comb:
    dataset['Title']=dataset.Name.str.split(',',expand=True)[1].str.split('.',expand=True)[0]

pd.crosstab(train_dataset.Title,train_dataset.Survived)



#for dataset in comb:
 #   dataset['Title'] = dataset['Title'].replace(['Capt','Col'], 'Rare')

  #  dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
   # dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    #dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
#train_dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
for dataset in comb:
    dataset['family']=dataset['SibSp']+dataset['Parch']+1
train_dataset[['family','Survived']].groupby(['family'],as_index=False).mean()
for dataset in comb:
    dataset['Alone']=1
    dataset['Alone'].loc[dataset['family']>1]=0
    
train_dataset[['Alone','Survived']].groupby(['Alone'],as_index=False).mean()
## SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
#train_dataset[["Title",'Survived']].groupby(['Title'],as_index=False).mean()
train_dataset.columns
test_dataset.columns
train_dataset=train_dataset.drop(['PassengerId','Name','SibSp','Parch','Cabin','Ticket','Cabin','family','Title'],axis=1)
test_dataset=test_dataset.drop(['PassengerId','Name','SibSp','Parch','Cabin','Ticket','Cabin','family','Title'],axis=1)

train_dataset.columns
test_dataset.columns
comb=[train_dataset,test_dataset]
#To check which is the most occured value in data for Embarked
statistics.mode(train_dataset['Embarked'])
#To get the unique value of column
train_dataset['Embarked'].unique().tolist()
#impute missing value for a single column

train_dataset["Embarked"]=train_dataset["Embarked"].fillna("S")
test_dataset['Fare']=train_dataset['Fare'].fillna(train_dataset['Fare'].mean())
#Data Encoding for Embarked and Sex
for data in comb:
    data['Embarked']=data['Embarked'].map({'S':1,'C':2,'Q':3})
    data['Sex']=data['Sex'].map({'male':0,'female':1})
train_dataset.head()
#Prediction of Age by applying KMeans Clusters
#X3=train_dataset.iloc[:,[1,2,4,5,6]].values
#from sklearn.cluster import KMeans
#wcss=[]
#for i in range(1,21):
 #   kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
#    kmeans.fit(X3)
  #  wcss.append(kmeans.inertia_)

#plt.plot(range(1,21),wcss)
#plt.show()
#X3.shape
#from sklearn.cluster import KMeans
#kmeans=KMeans(n_clusters=4,init='k-means++',random_state=42)
#kmeans.fit(X3)
#y_kmeans=kmeans.predict(X3)
#print(y_kmeans)

#Find a range of fare 
train_dataset['Farebin']=pd.cut(train_dataset['Fare'],5)

train_dataset[['Farebin','Survived']].groupby(['Farebin'],as_index=False).mean()
for data in comb:
    data.loc[data['Fare']<102,'Fare']=0
    data.loc[(data['Fare']>102) & (data['Fare']<204),'Fare']=1
    data.loc[(data['Fare']>204) & (data['Fare']<307),'Fare']=2
    data.loc[data['Fare']>307, 'Fare']=3
#Applying Regression Alogorithm to calculate the Missing_Age values

missing_data1=train_dataset[train_dataset['Age'].isnull()]
non_missing_data1=train_dataset[train_dataset['Age'].notna()]
missing_data2=test_dataset[test_dataset['Age'].isnull()]
non_missing_data2=test_dataset[test_dataset['Age'].notna()]

#missing_data1 contains missing row of age for Train dataset
#non_missing_data1 contains non missing row of age for Train dataset
#missing_data2 contains missing row of age for Test dataset
#non_missing_data2 contains non missing row of age for Test dataset
X4_train1=non_missing_data1.iloc[:,[1,2,4,5,6]].values
Y4_train1=non_missing_data1.iloc[:,[3]].values
X4_test1=missing_data1.iloc[:,[1,2,4,5,6]].values
#Apply Linear Regression to calculate Age
#from sklearn.linear_model import LinearRegression
#regression_classifier=LinearRegression()
#regression_classifier.fit(X4_train1,Y4_train1)
#Y4_pred1=regression_classifier.predict(X4_test1)

#ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
from sklearn.ensemble import RandomForestRegressor
random_regressor=RandomForestRegressor(n_estimators=50,criterion='mse',min_samples_split=8,
                                      min_samples_leaf=8,max_features=4,min_weight_fraction_leaf=0.05,                                    max_depth=6)
random_regressor.fit(X4_train1,Y4_train1.ravel())
Y4_pred1=random_regressor.predict(X4_test1)

#Y4_pred1_df=pd.DataFrame(Y4_pred1,columns=["Age"])
missing_data1=missing_data1.drop(['Age'],axis=1)

missing_data1['Age']=Y4_pred1
X4_train2=non_missing_data2.iloc[:,[0,1,3,4,5]].values
Y4_train2=non_missing_data2.iloc[:,[2]].values
X4_test2=missing_data2.iloc[:,[0,1,3,4,5]].values

#from sklearn.linear_model import LinearRegression
#regression_classifier=LinearRegression()
#regression_classifier.fit(X4_train2,Y4_train2)
#Y4_pred2=regression_classifier.predict(X4_test2)
from sklearn.ensemble import RandomForestRegressor
random_regressor=RandomForestRegressor(n_estimators=50,criterion='mse',min_samples_split=8,
                                      min_samples_leaf=8,max_features=4,min_weight_fraction_leaf=0.05,                                    max_depth=6)
random_regressor.fit(X4_train2,Y4_train2.ravel())
Y4_pred2=random_regressor.predict(X4_test2)
missing_data2=missing_data2.drop(['Age'],axis=1)

missing_data2['Age']=Y4_pred2

missing_data1=missing_data1.drop(["Farebin"],axis=1)
non_missing_data1=non_missing_data1.drop(["Farebin"],axis=1)

frame=[non_missing_data1,missing_data1]
dataframe1=pd.concat(frame)
dataframe1=dataframe1.sort_index(ascending=True,axis=0)


frame=[non_missing_data2,missing_data2]
dataframe2=pd.concat(frame)
dataframe2=dataframe2.sort_index(ascending=True,axis=0)

dataframe1['AgeBin']=pd.cut(dataframe1['Age'],5)

dataframe1[['AgeBin','Survived']].groupby(['AgeBin'],as_index=False).mean()
comb2=[dataframe1,dataframe2]

for data in comb2:
    data.loc[data['Age']<16.5,'Age']=0
    data.loc[(data['Age']>16.5) & (data['Age']<32.5),'Age']=1
    data.loc[(data['Age']>32.5) & (data['Age']<48.5),'Age']=2
    data.loc[(data['Age']>48.5) & (data['Age']<64),'Age']=3
    data.loc[data['Age']>64,'Age']=4
dataframe1=dataframe1.drop(["AgeBin"],axis=1)
comb2=[dataframe1,dataframe2]
X_train=dataframe1.iloc[:,0:6].values
y_train=dataframe1.iloc[:,6].values
X_test=dataframe2.iloc[:,0:6].values
#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
y_test=test_y.iloc[:,1].values

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
r_classifier=RandomForestClassifier(n_estimators=70,min_samples_split=10,max_features=4,
                                    min_samples_leaf=10,random_state=0)
r_classifier.fit(X_train,y_train)
y_p=r_classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_p))
print(confusion_matrix(y_test,y_p))
print(accuracy_score(y_test,y_p))
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=r_classifier,X=X_train,y=y_train,scoring='accuracy',cv=10)
print(accuracies)
print(accuracies.mean())
print(accuracies.std())
parameters={'n_estimators':[70,100,130],'min_samples_split':[8,10,12],'min_samples_leaf':[8,10,12],
           'max_features':[3,4,5],'min_weight_fraction_leaf':[0.5,0.05,0.005]}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=r_classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid.fit(X_train,y_train)
best_param=grid_search.best_params_

ac=grid_search.best_score_
print(ac)
print(best_param)
#Editing is going on..
#Any Suggestions will be appreciated