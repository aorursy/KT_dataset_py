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
# Reading data

df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')

df_test_copy=df_test.copy()
# Have a first look at train data

print(df_train.shape)
# Now, lets explore first five data from training set.

df_train.head()
# We will use describe function to calculate count,mean,max and other for numerical feature.

df_train.describe().transpose()
# The feature survived contain binary data which can also be seen from its max(1) and min(0) value.
# Have a look for possible missing values

df_train.info()
df_train.isnull().sum()
# We see that Age, Cabin and Embarked feature have NULL values.
# Have a first look at test data

print(df_test.shape)
# Have a look at train and test columns

print('Train columns:', df_train.columns.tolist())

print('Test columns:', df_test.columns.tolist())
# Let's look at the figures and Understand the Survival Ratio

df_train.Survived.value_counts(normalize=True)
# We observe that less people survived.
# To get better understanding of count of people who survived, we will plot it.
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='Survived',data=df_train)
sns.countplot(x='Pclass',data=df_train,hue='Survived')
sns.countplot(x='Sex',data=df_train,hue='Survived')
sns.catplot(x='Sex' , y='Age' , data=df_train , hue='Survived' , kind='violin' , palette=['r','g'] , split=True)
sns.kdeplot(df_train.Age , shade=True , color='r')
# To fill the missing values, we will calculate median of age with respect to Pclass.

df_train.groupby('Pclass').median()
# Now we will create a function to fill missing age values. This function is used to fill the age according to Pclass.

def fill_age(cols):

    

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 37

        

        elif Pclass == 2:

            return 29

        

        else:

            return 24

        

    else:

        

        return Age
df_train['Age'] = df_train[['Age','Pclass']].apply(fill_age,axis=1)
print(df_train.Age.count())  # Null values filled
sns.factorplot(x='Sex',y='Age' , col='Pclass', data=df_train , hue='Survived' , kind = 'box', palette=['r','g'])
# Understanding Box Plot :



# The bottom line indicates the min value of Age.

# The upper line indicates the max value.

# The middle line of the box is the median or the 50% percentile.

# The side lines of the box are the 25 and 75 percentiles respectively.
plt.figure(figsize=(20,30))

sns.factorplot(x='Embarked' , y ='Fare' , kind='bar', data=df_train , hue='Survived' , palette=['r','g'])
plt.figure(figsize=(20,10))

sns.boxplot(x='Embarked',y='Fare',data=df_train,hue='Survived')
# The best way to fill it would be by most occured value

df_train['Embarked'].fillna(df_train['Embarked'].mode()[0] ,inplace=True)
df_train.Embarked.count() # filled the values with Mode.
#Since Cabin has so many missing value, we will remove that column.
df_train.drop('Cabin',axis=1,inplace=True)
sns.violinplot(x='Embarked' , y='Pclass' , data=df_train , hue='Survived' , palette=['r','g'])
df_train.isnull().sum()
# None of the columns are empty.
sns.countplot(data=df_train,x='SibSp',hue='Survived')
df_train[['SibSp','Survived']].groupby('SibSp').mean()
df_train[['Parch','Survived']].groupby('Parch').mean()
df_train['Alone'] = 0

df_train.loc[(df_train['SibSp']==0) & (df_train['Parch']==0) , 'Alone'] = 1



df_test['Alone'] = 0

df_test.loc[(df_test['SibSp']==0) & (df_test['Parch']==0) , 'Alone'] = 1
df_train.head()
drop_features = ['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket' ]



df_train.drop(drop_features , axis=1, inplace = True)
df_test.info()
# We have a few Null values in Test (Age , Fare) , let's fill it up.
df_test['Fare'].fillna(df_test['Fare'].median() , inplace=True)
df_test.groupby('Pclass').median()
def fill_ages(cols):

    

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 42

        

        elif Pclass == 2:

            return 26.5

        

        else:

            return 24

        

    else:

        

        return Age
df_test['Age'] = df_test[['Age','Pclass']].apply(fill_ages,axis=1)
df_test.info()
drop_featuress = ['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket','Cabin' ]



df_test.drop(drop_featuress , axis=1 , inplace = True)
df_test.info()
def mapping(frame):

    

    frame['Sex'] = frame.Sex.map({'female': 0 ,  'male': 1}).astype(int)

    

    

    frame['Embarked'] = frame.Embarked.map({'S' : 0 , 'C': 1 , 'Q':2}).astype(int)

    

    

    

    frame.loc[frame.Age <= 16 , 'Age'] = 0

    frame.loc[(frame.Age >16) & (frame.Age<=32) , 'Age'] = 1

    frame.loc[(frame.Age >32) & (frame.Age<=48) , 'Age'] = 2

    frame.loc[(frame.Age >48) & (frame.Age<=64) , 'Age'] = 3

    frame.loc[(frame.Age >64) & (frame.Age<=80) , 'Age'] = 4

    

    

    frame.loc[(frame.Fare <= 7.91) , 'Fare'] = 0

    frame.loc[(frame.Fare > 7.91) & (frame.Fare <= 14.454) , 'Fare'] = 1

    frame.loc[(frame.Fare > 14.454) & (frame.Fare <= 31) , 'Fare'] = 2

    frame.loc[(frame.Fare > 31) , 'Fare'] = 3
mapping(df_train)

df_train.head()
mapping(df_test)

df_test.head()
# Importing some algorithms from sklearn.

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
# Splitting data into test and train set.

x_train,x_test,y_train,y_test=train_test_split(df_train.drop('Survived',axis=1),df_train.Survived,test_size=0.20,random_state=66)
models = [LogisticRegression(),RandomForestClassifier(),

        DecisionTreeClassifier()]



model_names=['LogisticRegression','RandomForestClassifier','DecisionTree']



accuracy = []



for model in range(len(models)):

    clf = models[model]

    clf.fit(x_train,y_train)

    pred = clf.predict(x_test)

    accuracy.append(accuracy_score(pred , y_test))

    

compare = pd.DataFrame({'Algorithm' : model_names , 'Accuracy' : accuracy})

compare
params_dict={'criterion':['gini','entropy'],'max_depth':[5.21,5.22,5.23,5.24,5.25,5.26,5.27,5.28,5.29,5.3]}

clf_dt=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params_dict,scoring='accuracy', cv=5)

clf_dt.fit(x_train,y_train)

pred=clf_dt.predict(x_test)

print(accuracy_score(pred,y_test))

print(clf_dt.best_params_)
predio = clf_dt.predict(df_test)



d = {'PassengerId' : df_test_copy.PassengerId , 'Survived' : predio}

answer = pd.DataFrame(d)

# Generate CSV file based on DecisionTree Classifier

answer.to_csv('predio.csv' , index=False)