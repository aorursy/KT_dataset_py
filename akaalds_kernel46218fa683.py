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
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'

%matplotlib inline



#Importing libraries

import pandas as pd

import numpy as np

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

test['Survived']=np.nan

dataset=train.append(test, ignore_index=True)

dataset.head()
dataset.info()
dataset.describe()
#checking nullvalues in the dataset

dataset.isna().sum()
#checking the other attributes for the passenger where fare is mentioned as null

#and filling the missing value in Fare column

dataset[(dataset['Fare'].isnull())]

missing_fare=dataset[(dataset['Pclass'] == 3) & (dataset['Embarked'] == 'S') & (dataset['Parch'] == 0) & (dataset['Age'] > 50)]['Fare'].mean()

dataset['Fare']=dataset['Fare'].fillna(missing_fare)

dataset.isna().sum()
#checking the other attributes for the passenger where embarked is mentioned as null

dataset[(dataset['Embarked'].isnull())]

#No concrete relationship can be established to fill the missing values hence these values are dropped

dataset.drop(dataset[dataset['Embarked'].isnull()].index, inplace=True)

dataset.shape
#Filling missing values for Age for that we need to check the relationship between Pclass and Age as elderly people usually look for luxurious class

import seaborn as sns

sns.boxplot('Pclass', 'Age', data=dataset)
#writing a function to fill the missing values for Age column deensing upon the box plot as for Pclass=1:age=39, Pclass=2:age=29 and Pclass=3:age=25

def age_approx(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 39

        elif Pclass == 2:

            return 29

        else:

            return 25

    else:

        return Age

dataset['Age']=dataset[['Age', 'Pclass']].apply(age_approx, axis=1)

#Checking for missing values

dataset.isna().sum()
#Converting object 'datatype' column into binary format

dataset['Sex'].value_counts()

dataset['Sex1']=dataset['Sex']

dataset['Sex']=pd.Categorical(dataset['Sex'])

dataset['Sex']=dataset['Sex'].cat.codes



dataset['Embarked'].value_counts()

dataset['Embarked1']=dataset['Embarked']

dataset['Embarked']=pd.Categorical(dataset['Embarked'])

dataset['Embarked']=dataset['Embarked'].cat.codes

dataset.head()
#taking the copy of the original dataset

dataset_org=dataset.copy()



#Moreover Cabin has been dropped as it had a lot of missing values and there doesn't seem  to be any concrete method

#to fillthe missing values for it.

dataset.drop(columns=['Cabin'], inplace=True)



#dividing the dataset based on train & test

train=dataset.loc[(dataset['Survived'].notnull())]

test=dataset.loc[(dataset['Survived'].isnull())]

train.head()

test.head()
#Dropping the columns which are unique and the ones which have been converted

train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Sex1', 'Embarked1'], inplace=True)

test.drop(columns=['Name', 'Ticket', 'Sex1', 'Embarked1','Survived'],inplace=True)

train.head()

test.head()
#Feature correlation

train.corr()['Survived']



#Executing heatmap for correlation comparison and better veiw

ax=sns.heatmap(train.corr())
#Since Pclass and fare are very correlated, further Pclass is better correlated to Target Survived hence we would drop Fare

train.drop(columns=['Fare'], inplace=True)

test.drop(columns=['Fare'], inplace=True)
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y=train_test_split(train.drop(columns=['Survived'], axis=1), train['Survived'], test_size=0.2)
#Creating models and learning the classifiers

from sklearn.ensemble import RandomForestClassifier



#define model

RFC=RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, max_features=3, n_jobs=-1, random_state=40)

RFC.fit(train_x, train_y)

predictions=RFC.predict(test_x)



#Calculating accuracy score and creating confussion matrix

from sklearn.metrics import accuracy_score, confusion_matrix

acc=accuracy_score(test_y, predictions)

print(f'Accuracy Score:', acc)

print(f'Confusion Matrix:')

confusion_matrix(test_y, predictions)
#using logistic regression

from sklearn.linear_model import LogisticRegression



LRC=LogisticRegression(penalty='l2', C=0.1, random_state=40)

LRC.fit(train_x, train_y)

predictions=LRC.predict(test_x)



#Calculating accuracy score

acc=accuracy_score(test_y, predictions)

print(f'Accuracy Score:', acc)

print(f'Confussion matrix:')

confusion_matrix(test_y, predictions)
#USing random forest classifier

test_features=test.drop(columns=['PassengerId']) #to be used for predicting the labels for test data

test_features.head()
#running RFC over the test data for predictions

test_pred=RFC.predict(test_features)



#Creating DataFrame using the predictions 

test_pred_RFC_2=pd.DataFrame(data=test_pred, columns=['Survived'], index=test.index.copy())

dataset_out=pd.merge(test, test_pred_RFC_2, how='left', right_index=True, left_index=True)
#Creating Gender submission file

gender_submission=pd.DataFrame(dataset_out[['PassengerId', 'Survived']])

gender_submission.to_csv('gender_submission.csv', index=False)