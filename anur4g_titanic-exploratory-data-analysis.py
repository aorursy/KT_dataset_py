import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# reading data from csv

data = pd.read_csv('/kaggle/input/titanic/train.csv')

# shape of the dataset

print(data.shape)
test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(test.shape)
data.head()
data.info()
data.describe()
# function of finding NaN Values present in dataset

def find_NaN(data):

    total = data.isnull().sum().sort_values(ascending=False)

    percent_1 = data.isnull().sum()/data.isnull().count()*100

    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

    nan = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

    print('Finding NaN Values present in dataset')

    print(nan.head(4))
find_NaN(data)
# Count values in the column

data['Embarked'].value_counts()
# Fill NaN value with mean in Age

data['Age'] = data['Age'].fillna((data['Age'].mean()))



# Fill NaN value with most common Port

data['Embarked'] = data['Embarked'].fillna('S')
data.columns
# Function Showing DataFrame Side by Side

from IPython.core.display import display, HTML

def display_side_by_side(dfs:list, captions:list):

    output = ""

    combined = dict(zip(captions, dfs))

    for caption, df in combined.items():

        # Use df.head() to show only top 5 rows

        output += df.head().style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()

        output += "\xa0\xa0\xa0"

    display(HTML(output))
# Relationship between Survival of Passenger and Different features

PCLASS = data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

SEX = data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

AGE = data[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)

PARCH = data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

SIBSP = data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

display_side_by_side([PCLASS,SEX,AGE,PARCH,SIBSP], ['TICKET CLASS','SEX','AGE','PARCH','SIBSP'])
# PCLASS

survived = pd.crosstab(index=data.Survived, columns = data.Pclass, margins=True)

survived.columns = ['Upper Class','Middle Class','Lower Class','ColTotal']

survived.index = ['Not Survived','Survived','RowTotal']

# Normalization of PCLASS

survived_per = pd.crosstab(index=data.Survived, columns = data.Pclass, margins=True,normalize=True)

survived_per.columns = ['Upper Class','Middle Class','Lower Class','ColTotal']

survived_per.index = ['Not Survived','Survived','RowTotal']

display_side_by_side([survived, survived_per], ['Survived','Survived_per'])



# Siblings 

survived_sib = pd.crosstab(index=data.Survived, columns = data.SibSp, margins=True,colnames=['Siblings'])

# Normalisation of Siblings

survived_sib_per = pd.crosstab(index=data.Survived, columns = data.SibSp, margins=True,colnames=['Siblings'],normalize=True)

survived_sib_per.index = ['Not Survived','Survived','RowTotal']

survived_sib.index = ['Not Survived','Survived','RowTotal']

display_side_by_side([survived_sib, survived_sib_per], ['Survived_sib','Survived_sib_per'])

# importing libraries for data visualisation

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

sns.set(style='darkgrid')
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(14, 6))

female = data[data['Sex']=='female']

male = data[data['Sex']=='male']



# Chart of Female

ax = sns.distplot(female[female['Survived']==1].Age.dropna(),bins=18, label = survived, ax = axes[0], kde =True)

ax = sns.distplot(female[female['Survived']==0].Age.dropna(),bins=40, label = not_survived, ax = axes[0], kde =True)

ax.legend()

ax.set_title('Female')



# Chart of Male

ax = sns.distplot(male[male['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = True)

ax = sns.distplot(male[male['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = True)

ax.legend()

ax.set_title('Male')
sns.countplot('Pclass', hue='Survived', data=data)

plt.title('PClass Survival Rate')

plt.show()
# Deep Analysis of Pclass

pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True).style.background_gradient(cmap='PuBu')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)

plt.show()
FacetGrid = sns.FacetGrid(data, row='Embarked', height=4.0, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex',palette='deep',  order=None, hue_order=None )

FacetGrid.add_legend()
data = data.drop(['Name','Cabin','PassengerId','Ticket'],axis=1)
# label encoding the data 

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 



data['Sex']= le.fit_transform(data['Sex']) 

data['Embarked'] =le.fit_transform(data['Embarked'])
data.columns
data.head()
plt.figure(figsize=(12,10))

cor = data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train_data = pd.DataFrame()
train_data = data
train_data.head()
train_X = train_data[train_data.columns[1:]].values

train_Y = train_data[train_data.columns[0]]
test_data = pd.DataFrame()

test_data = test

test_data.head()
final_output = pd.DataFrame()

final_output = pd.DataFrame({'PassengerId': test_data['PassengerId']})

final_output.head()
test_data = test_data.drop(['Name','Cabin','PassengerId','Ticket'],axis=1)
# Fill NaN value with mean in Age

test_data['Age'] = test_data['Age'].fillna((test_data['Age'].mean()))



# Fill NaN value with most common Port

test_data['Fare'] = test_data['Fare'].fillna((test_data['Fare'].mean()))
find_NaN(test_data)
# label encoding the data 

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 



test_data['Sex']= le.fit_transform(test_data['Sex']) 

test_data['Embarked'] =le.fit_transform(test_data['Embarked'])
test_data.head()
test_X = test_data[test_data.columns].values
from sklearn.svm import SVC # "Support Vector Classifier" 

svm_clf = SVC(kernel='linear',random_state=45) 

  

# fitting x samples and y classes 

svm_clf.fit(train_X,train_Y) 
test_Y = svm_clf.predict(test_X)
final_output['Survived'] = test_Y
final_output.head()
final_output.to_csv('SVM_output.csv',sep=',',index=False)
# Random Forest is used for unbalanced DataSets

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore")



# Train model

random_clf = RandomForestClassifier(random_state=45,n_estimators=100,min_samples_leaf=50)

random_clf.fit(train_X, train_Y)
random_test_Y = random_clf.predict(test_X)
final_output['Survived'] = random_test_Y
final_output.head()
final_output.to_csv('Random_output.csv',sep=',',index=False)