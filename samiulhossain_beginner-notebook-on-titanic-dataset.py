# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#I created a notebook in Kaggle by clicking on the new notebook option, so the data was already imported in a read-only format

#The data is in a csv format with 1 target variable and 11 feature variables.
# data analysis

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# models

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing, model_selection, metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head()
#Survived is the target variable whilst the others are the feature variables
test_df.head()
print('The number of rows in dataset is - ' , train_df.shape[0])

print('The number of columns in dataset is - ' , train_df.shape[1])
train_df.info()

#Cabin has alot of null values, age has some null values. Null values must be handled.
test_df.info()
train_df.describe()
#this function shows us some details about the numeric variables present in our data
train_df.isnull().sum().sort_values(ascending = True)
#Number of null values in all columns. Cabin has too many null values.
test_df.isnull().sum().sort_values(ascending = True)
train_df.hist(bins = 11, figsize= (12,16)) ;
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Pclass', bins=20)
#Pclass = 3 has the lowest survival rate
g = sns.FacetGrid(train_df, col='Survived')

g.map(sns.distplot, 'Fare', bins=20)
#most of the deaths were from the lower class who had to give low fare.
g1 = sns.FacetGrid(train_df, col='Survived')

g1.map(sns.distplot, 'Age', bins=20)
#Passengers aged 20-40 died the most

#Younger passengers survived more, specially children. This means age is a very important variable
for df in [train_df,test_df]:

    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
#changed the categorical value of sex to 0 and 1
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Sex_binary', bins=20)
train_df["Embarked"] = train_df["Embarked"].fillna("S")
#filled in the missing embarked values
# Explore Embarked vs Survived 

g = sns.factorplot(x="Embarked", y="Survived",  data=train_df,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 

g = sns.factorplot("Pclass", col="Embarked",  data=train_df,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
for df in [train_df,test_df]:

    df['Embarked_binary']=df['Embarked'].map({'S':0,'C':1,'Q':2})
#Changed the categorical value of embarked to 0,1,2
train_df['Family'] = train_df.apply(lambda x: x['SibSp'] + x['Parch'], axis = 1)

test_df['Family'] = test_df.apply(lambda x: x['SibSp'] + x['Parch'], axis = 1)
#using feature engineering, created a new column family from sibsp and parch
#We can see that most first class passengers embark from S C and none from Q. Survival rate is also highest in Pclass 1

#We will now fill the null values of age and fare
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())



dfs = [train_df ,test_df]



for df in dfs:

    df['Age'].fillna(df['Age'].median(), inplace = True)
train_df.tail()
train_df.drop('Sex',axis=1,inplace=True)

train_df.drop('Name',axis=1,inplace=True)

train_df.drop('Cabin',axis=1,inplace=True)

train_df.drop('Ticket',axis=1,inplace=True)

train_df.drop('SibSp',axis=1,inplace=True)

train_df.drop('Parch',axis=1,inplace=True)

train_df.drop('Embarked',axis=1,inplace=True)





test_df.drop('Sex',axis=1,inplace=True)

test_df.drop('Name',axis=1,inplace=True)

test_df.drop('Cabin',axis=1,inplace=True)

test_df.drop('Ticket',axis=1,inplace=True)

test_df.drop('SibSp',axis=1,inplace=True)

test_df.drop('Parch',axis=1,inplace=True)

test_df.drop('Embarked',axis=1,inplace=True)

train_df.tail()
test_df.tail()
#Preparing for model training
features = ['Pclass','Age','Sex_binary','Family','Fare','Embarked_binary']

target = 'Survived'
train_df[features].head()

test_df[features].head()
clfE = ExtraTreesClassifier()

clfE.fit(train_df[features],train_df[target])
clfE_data={}

clfE_data["Train_R2_Score"] = metrics.r2_score(train_df[target],clfE.predict(train_df[features]))

clfE_data
clfR = RandomForestClassifier()

clfR.fit(train_df[features],train_df[target])
clfR_data={}

clfR_data["Train_R2_Score"] = metrics.r2_score(train_df[target],clfR.predict(train_df[features]))

clfR_data
clfD = DecisionTreeClassifier()

clfD.fit(train_df[features],train_df[target])
clfD_data={}

clfD_data["Train_R2_Score"] = metrics.r2_score(train_df[target],clfD.predict(train_df[features]))

clfD_data
#Create classifier object with hyperparameters

#cross validation

param_grid = {

    'n_estimators': [200, 500, 1000],

    'max_features': ['auto'],

    'max_depth': [6, 7, 8],

    'criterion': ['entropy']

}



clf = RandomForestClassifier()

CV = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5)



#Fit our classifier using the training features and the training target values

CV.fit(train_df[features],train_df[target])

CV.best_estimator_

clf = RandomForestClassifier(criterion = 'entropy', max_depth = 8, n_estimators = 200, random_state = 42)

clf.fit(train_df[features], train_df[target])
#Make predictions using the features from the test data set

predictions = clf.predict(test_df[features])



#Display our predictions - they are either 0 or 1 for each training instance 

#depending on whether our algorithm believes the person survived or not.

predictions
clf_data={}

clf_data["Train_R2_Score"] = metrics.r2_score(train_df[target],CV.predict(train_df[features]))
clf_data
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)