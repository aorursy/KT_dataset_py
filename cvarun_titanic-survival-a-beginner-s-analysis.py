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
#for analysis of data, dataframe

import numpy as np

import pandas as pd



#for plotting and stuffs

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

#the above line of code is known as a magic function, helps to display our plots just below our code in the notebook.



#for model training & prediction

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
#read training data into 'train_df' dataframe

train_df=pd.read_csv('../input/train.csv')



#read testing data into 'test_df' dataframe

test_df=pd.read_csv('../input/test.csv')



#combined dataset, will be handy in wrangling steps.

combined_df=[train_df,test_df]
train_df.columns
test_df.columns
#to know what type of data columns hold ; 'object' type means they hold string values

train_df.dtypes
test_df.dtypes
train_df.info()
test_df.info()
#train_df.info(verbose=False) will give a compact version of the above output, it set to True by default(in above case).

train_df.info(verbose=False)
train_df.head() #by default it prints first 5 rows, any other integer can also be given inside parenthesis.
test_df.head()
train_df.describe()

#this gives metric/stats of various columns.
ax=train_df['Sex'].value_counts().plot.bar(title='Sex Distribution aboard Titanic',figsize=(8,4))



#below loop is to print numeric value above the bars

for p in ax.patches:

    ax.annotate(str(p.get_height()),(p.get_x(),p.get_height()*1.005))



sns.despine()  #to remove borders (by default : from top & right side)
sns.set(style='whitegrid')

ax=sns.kdeplot(train_df['Age'])

ax.set_title('Age Distribution aboard the Titanic')

ax.set_xlabel('<---AGE--->')
print(train_df['Survived'].value_counts())

l=['Not Survived','Survived']

ax=train_df['Survived'].value_counts().plot.pie(autopct='%.2f%%',figsize=(6,6),labels=l)

#autopct='%.2f%%' is to show the percentage text on the plot

ax.set_ylabel('')
sns.countplot(train_df['Pclass'])

sns.despine()
sns.countplot(train_df['Embarked'])
train_df[['Sex','Survived']].groupby('Sex').mean()
train_df[['Pclass','Survived']].groupby('Pclass').mean()
train_df.groupby(['Pclass','Survived'])['Pclass'].count()
sns.countplot(x='Pclass',hue='Survived',data=train_df)
train_df[['Embarked','Survived']].groupby('Embarked').mean()
train_df[['Parch','Survived']].groupby('Parch').mean()
train_df[['SibSp','Survived']].groupby('SibSp').mean()
ax=train_df[['Parch','Survived']].groupby('Parch').mean().plot.line(figsize=(8,4))

ax.set_ylabel('Survival')

sns.despine()
ax=train_df[['SibSp','Survived']].groupby('SibSp').mean().plot.line(figsize=(8,4))

ax.set_ylabel('Survival')

sns.despine()
a=sns.FacetGrid(train_df,col='Survived')

a.map(sns.distplot, 'Age')
a=sns.FacetGrid(train_df,col='Pclass',row='Survived')

a.map(plt.hist,'Age')
train_df['Embarked'].value_counts()
a=sns.FacetGrid(train_df,col='Embarked')

a.map(sns.distplot,'Survived')
train_df.groupby(['Embarked','Survived'])['Embarked'].count()
a=sns.FacetGrid(train_df,col='Embarked')

a.map(sns.pointplot, 'Pclass','Survived','Sex') #colum order is x='Pclass', y='Survived', hue='Sex'

a.add_legend()
train_df.groupby(['Embarked','Sex'])['Embarked'].count()
a=sns.FacetGrid(train_df,col='Survived')

a.map(sns.barplot,'Sex', 'Fare')
combined_df[0].head(3) #[0] is train_df
combined_df[1].head(3)  #[1] is test_df
print('training data dimensions :',train_df.shape)

print('testing data dimensions :', test_df.shape)

print('combined data\'s dimension are :\n',combined_df[0].shape,'\n',combined_df[1].shape)
train_df[['PassengerId','Name','Ticket','Cabin']].head()
#removing mentioned columns from dataset

train_df=train_df.drop(['Name','Ticket','Cabin','SibSp','Parch','PassengerId'],axis=1)

test_df=test_df.drop(['Name','Ticket','Cabin','SibSp','Parch'],axis=1)
# the combined data

combined_df=[train_df, test_df]
#lets check the new dimensions

print('new training data dimensions :',train_df.shape)

print('new testing data dimensions :', test_df.shape)

print('new combined data\'s dimension are :\n',combined_df[0].shape,'\n',combined_df[1].shape)
train_df.head(3)
#checking for any null values

train_df.isnull().any() #True means null present
test_df.isnull().any()
# age columns

print('mean age in train data :',train_df['Age'].mean())

print('mean age in test data :',test_df['Age'].mean())
#replacing null values with 30 in age column

for df in combined_df:

    df['Age']=df['Age'].replace(np.nan,30).astype(int)
train_df['Embarked'].value_counts()
#most people embarked from 'S'. So, we'll replace the missing missing Embarked value by 'S'.

train_df['Embarked']=train_df['Embarked'].replace(np.nan,'S')
#finding mean fare in test data

test_df['Fare'].mean()
#replace missing fare values in test data by mean

test_df['Fare']=test_df['Fare'].replace(np.nan,36).astype(int)
combined_df=[train_df,test_df]

for df in combined_df:

    print(df.isnull().any()) #bool value = False means that there are no nulls in the column.
#will code female as 1 and male as 0

for df in combined_df:

    df['Sex']=df['Sex'].map({'female':1,'male':0}).astype(int)
train_df.head(3)
#coding Embarked column as: S=2, C=1, Q=0

for df in combined_df:

    df['Embarked']=df['Embarked'].map({'S':2,'C':1,'Q':0}).astype(int)
train_df.head(3)
#binning or making bands of age into intervals and then assigning labels to them(encoding the bands as 0,1,2,3,4)

for df in combined_df:

    df['Age']=pd.cut(df['Age'],5,labels=[0,1,2,3,4]).astype(int) #pandas cut will help us divide age in bins
train_df.head(3)
#binning fares and assigning label 0,1,2,3 to their respective bins

for df in combined_df:

    df['Fare']=pd.qcut(df['Fare'],4,labels=[0,1,2,3]).astype(int)
train_df.head(3)
test_df.head(3)
X_train=train_df.drop('Survived',axis=1)

Y_train=train_df['Survived']



#X_train is the entire training data except the Survived column, which is separately stored in Y_train. We will use these to train our MODEL !



X_test=test_df.drop('PassengerId',axis=1).copy()

#X_test is the test data, for on which we will apply model and predict the "SURVIVED" column for its entries.
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
#first applying Logistic Regression



lg = LogisticRegression()

lg.fit(X_train, Y_train)

Y_pred1 = lg.predict(X_test)

accu_lg = (lg.score(X_train, Y_train))

round(accu_lg*100,2)
#applying decision tree



dtree = DecisionTreeClassifier()

dtree.fit(X_train, Y_train)

Y_pred2 = dtree.predict(X_test)

accu_dtree = (dtree.score(X_train, Y_train))

round(accu_dtree*100,2)
#applying random forest



rafo = RandomForestClassifier(n_estimators=100)

rafo.fit(X_train, Y_train)

Y_pred3 = rafo.predict(X_test)

accu_rafo = rafo.score(X_train, Y_train)

round(accu_rafo*100,2)
#our goal was to predict survived column for test data, and were asked to submit a dataframe with 'PassengerId' and 'Survived' columns



submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_pred3})
submission.shape
submission.head(10)
submission.to_csv('submission.csv', index=False)