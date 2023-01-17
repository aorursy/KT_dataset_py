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

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
data=pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data.shape # (891, 12)
data.isnull().sum()
print(data["Age"].max(), ', ', data["Age"].min(), ', ', data["Age"].mean())
data['Title']=0
for i in data:
    data['Title']=data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
pd.crosstab(data.Title,data.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Title with the Sex
# Replacing the miss spelled values with the correct ones
data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data.groupby('Title')['Age'].mean() # Checking the average by Titles
# data["Age_was_missing"] = data["Age"].isnull()
## Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull())&(data.Title=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Title=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Title=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Title=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Title=='Other'),'Age']=46
# data["Age_was_missing"].value_counts()
data["Embarked"].value_counts()
data['Embarked'].fillna('S',inplace=True) # Replace NaNs with the most frequent value: S
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()
f,ax=plt.subplots(1,2,figsize=(20,10))
data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
#### Age and Pclass
f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=data,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
f, ax = plt.subplots(1,2, figsize=(20,6))
sns.barplot('SibSp','Survived', data=data, ax=ax[0], palette='Set3')
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=data,ax=ax[1],hue='Pclass', palette='Set2')
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()
f, ax = plt.subplots(1,2,figsize=(20,6))
sns.barplot('Parch','Survived', data=data, ax=ax[0], palette='Set3')
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived', data=data, ax=ax[1], palatte='Set3', hue='Pclass')
ax[1].set_title('Parch vs Survived')
plt.close(2)
plt.show()
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(data[data['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(data[data['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
# Age_band: binned Age Column
data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
data.head(2)
data['Age_band'].value_counts()
data['Family_Size']=0
data['Family_Size']=data['Parch']+data['SibSp']#family size
data['Alone']=0
data.loc[data.Family_Size==0,'Alone']=1#Alone

data['Fare_Range']=pd.qcut(data['Fare'],4)
data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
data.head()
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.metrics import mean_absolute_error
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']
# # Define the models
# model_1 = RandomForestClassifier(n_estimators=50, random_state=0)
# model_2 = RandomForestClassifier(n_estimators=100, random_state=0)
# model_3 = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=0)
# model_4 = RandomForestClassifier(n_estimators=50, min_samples_split=20, max_depth=4, random_state=0)
# model_5 = RandomForestClassifier(n_estimators=100,  min_samples_split=20, max_depth=7, random_state=0)

# models = [model_1, model_2, model_3, model_4, model_5]
# # Function for comparing different models
# def score_model(model, X_t=train_X, X_v=test_X, y_t=train_Y, y_v=test_Y):
#     model.fit(X_t, y_t)
#     preds = model.predict(X_v)
#     return metrics.accuracy_score(preds, y_v)

# for i in range(0, len(models)):
#     ac_score = score_model(models[i])
#     print(f"Model {i+1} MAE: {ac_score}")
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

data_submission_test = test_data.copy()
# Some Analysis of the new data
data_submission_test.shape # (891, 12)
data_submission_test.describe()
data_submission_test.isnull().sum() 
# Replacing the miss spelled values with the correct ones
data_submission_test['Title']=0
for i in data_submission_test:
    data_submission_test['Title']=data_submission_test.Name.str.extract('([A-Za-z]+)\.')
data_submission_test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Dona','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

# Replace NaN with the previous average values
data_submission_test.loc[(data_submission_test.Age.isnull())&(data_submission_test.Title=='Mr'),'Age']=33
data_submission_test.loc[(data_submission_test.Age.isnull())&(data_submission_test.Title=='Mrs'),'Age']=36
data_submission_test.loc[(data_submission_test.Age.isnull())&(data_submission_test.Title=='Master'),'Age']=5
data_submission_test.loc[(data_submission_test.Age.isnull())&(data_submission_test.Title=='Miss'),'Age']=22
data_submission_test.loc[(data_submission_test.Age.isnull())&(data_submission_test.Title=='Other'),'Age']=46
# Age_Band: binned Age Column
data_submission_test['Age_band']=0
data_submission_test.loc[data_submission_test['Age']<=16,'Age_band']=0
data_submission_test.loc[(data_submission_test['Age']>16)&(data_submission_test['Age']<=32),'Age_band']=1
data_submission_test.loc[(data_submission_test['Age']>32)&(data_submission_test['Age']<=48),'Age_band']=2
data_submission_test.loc[(data_submission_test['Age']>48)&(data_submission_test['Age']<=64),'Age_band']=3
data_submission_test.loc[data_submission_test['Age']>64,'Age_band']=4
data_submission_test.head(2)
data_submission_test['Age_band'].value_counts()
data_submission_test['Family_Size']=0
data_submission_test['Family_Size']=data_submission_test['Parch']+data_submission_test['SibSp']#family size
data_submission_test['Alone']=0
data_submission_test.loc[data_submission_test.Family_Size==0,'Alone']=1 

data_submission_test['Fare_Range']=pd.qcut(data_submission_test['Fare'],4)

data_submission_test['Fare_cat']=0
data_submission_test.loc[data_submission_test['Fare']<=7.91,'Fare_cat']=0
data_submission_test.loc[(data_submission_test['Fare']>7.91)&(data_submission_test['Fare']<=14.454),'Fare_cat']=1
data_submission_test.loc[(data_submission_test['Fare']>14.454)&(data_submission_test['Fare']<=31),'Fare_cat']=2
data_submission_test.loc[(data_submission_test['Fare']>31)&(data_submission_test['Fare']<=513),'Fare_cat']=3
data_submission_test['Sex'].replace(['male','female'],[0,1],inplace=True)
data_submission_test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data_submission_test['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

data_submission_test.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
data_submission_test.shape
clf = RandomForestClassifier(n_estimators=50, min_samples_split=20, max_depth=4, random_state=0)
clf.fit(X, Y)

submissions_predictions = clf.predict(data_submission_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': submissions_predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")






