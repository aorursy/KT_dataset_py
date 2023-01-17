#Importing libraries for Data Manipulation and Visualization



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Modelling Algorithms

from sklearn.ensemble import RandomForestClassifier 



# Modelling Helpers

#from sklearn.preprocessing import  Normalizer , scale

from sklearn.model_selection import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



#Modelling Metrics

#from sklearn.metrics import r2_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
#Reading the CSV

df=pd.read_csv("/kaggle/input/titanic/train.csv")
#Checking the first few rows

df.head()
#Checking null values

df.isnull().sum()
df.describe()
df.pivot_table('Survived', index='Age').plot()
plt.figure(figsize=(8, 6))

sns.countplot(x='Survived',data=df)

plt.title("Survived",fontsize=30)

plt.ylabel("Count",fontsize=20)
df['Cabin'].head()
#Dropping Cabin Column

df.drop('Cabin',axis=1,inplace=True)
df.head()
#Creating a new variable while concatenating other 2 columns

df['Family']=df['SibSp']+df['Parch']
#Checking the correlation

sns.heatmap(df.corr(),annot=True)
#Dropping few columns

df.drop(['Name','Ticket','Fare','Parch','SibSp'],axis=1,inplace=True)
df.plot(by='Age')

df.info()
df.isnull().sum()
df.shape
df.drop(df[df['Embarked'].isnull()].index,inplace=True)
df.isnull().sum()
#Getting the mean age to replace null values in Age column

age_mean=df['Age'].mean()
print(age_mean)
df['Age'].fillna(age_mean,inplace=True)
df.info()
df.isnull().sum()
Sex=pd.get_dummies(df['Sex'],drop_first=True)
Emb=pd.get_dummies(df['Embarked'],drop_first=True)
Emb.head()
df1=df
df1=pd.concat([df,Sex,Emb],axis=1)
df1.head()
df1.drop(['Sex','Embarked'],axis=1,inplace=True)
X=df1[df1.loc[:,df1.columns!='Survived'].columns]

y=df1['Survived']
print (X.dtypes )
#Splitting the data into Training and Test 

X_train, X_test, y_train, y_test=train_test_split(X ,y, test_size=.2, random_state=10)
modelRF=RandomForestClassifier(n_estimators=100)
modelRF.fit(X_train,y_train )

print (modelRF.score(X_train,y_train), modelRF.score(X_test,y_test))

rfecvRF = RFECV( estimator = modelRF , step = 1 , cv = 5, scoring = 'accuracy', )

rfecvRF.fit( X_train,y_train)
y_predRF=rfecvRF.predict(X_test)
#Getting the accuracy Score

accuracy_score(y_test,y_predRF)
confusion_matrix(y_test,y_predRF)
#Getting the Test Data for Titanic



testdf=df=pd.read_csv("/kaggle/input/titanic/test.csv")
#Manipulating Data to match the Features

testdf['Family']=testdf['SibSp']+testdf['Parch']
Sex=pd.get_dummies(testdf['Sex'],drop_first=True)
Emb=pd.get_dummies(testdf['Embarked'],drop_first=True)
testdf1=testdf
testdf1=pd.concat([testdf,Sex,Emb],axis=1)
testdf1.head()
testdf1.drop(['SibSp','Parch','Sex','Embarked','Cabin','Name','Ticket','Fare'],axis=1,inplace=True)
testdf1.head()
testdf1.info()
age_mean=testdf1['Age'].mean()
testdf1['Age'].fillna(age_mean,inplace=True)
X_train.head()
X_Testnew=testdf1
y_Testnew=rfecvRF.predict(X_Testnew)
y_Testnew.shape
submissionnew = pd.DataFrame({'PassengerId':X_Testnew['PassengerId'],'Survived':y_Testnew})
filename = 'Titanic_Predictionsnew.csv'



submissionnew.to_csv(filename,index=False)



print('Saved file: ' + filename)