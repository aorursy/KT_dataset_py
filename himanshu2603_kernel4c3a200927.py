# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_style('dark')

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

    



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data.info()
train_data.info()
train_data['Survived'].value_counts(normalize=True)
import matplotlib.pyplot as plt
left=[1,0]

height=[0.383838,0.616162]

tick_label=['survived','death']

plt.bar(left,height,tick_label=tick_label,width=0.8,color=['red','green'])

plt.title('death & survived percentage')

plt.show()
g=sns.countplot(y=train_data['Survived']).set_title('surviver & death counts')
train_data['Pclass'].value_counts()
train_data.groupby('Pclass').Survived.mean()
fig, axarr = plt.subplots(1,2,figsize=(12,6))

a = sns.countplot(x='Pclass', hue='Survived', data=train_data, 

                  ax=axarr[0]).set_title('Survivors & deads count w.r.t. class')

axarr[1].set_title('Survival rate by class')

b = sns.barplot(x='Pclass', y='Survived', data=train_data, ax=axarr[1]).set_ylabel('Survival rate')
train_data.groupby(['Pclass','Sex']).Survived.mean()
plt.title('survivle by pclass & sex')

g=sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train_data).set_ylabel('survival rate ')
train_data['Embarked'].value_counts()
g=sns.barplot(x=['S','C','Q'],y=[644,168,77] )
train_data.groupby('Embarked').Survived.mean()
g=sns.barplot(x='Embarked',y='Survived',data=train_data)
train_data.groupby(['Embarked','Sex']).Survived.mean()
plt.title('survivle by Embarked and sex')

g=sns.barplot(x='Embarked',y='Survived',hue='Sex',data=train_data).set_ylabel('Survival rate')
train_data.Fare.describe()
fig, hrr = plt.subplots(1,2,figsize=(12,6))

f = sns.distplot(train_data.Fare, color='r', ax=hrr[0]).set_title('Fare distribution')

fare_ranges = pd.qcut(train_data.Fare, 4, labels = ['Low', 'average', 'High', 'Very high'])

hrr[1].set_title('Survival rate by fare category')

g = sns.barplot(x=fare_ranges, y=train_data.Survived, ax=hrr[1]).set_ylabel('Survival rate')

plt.figure(figsize=(12,5))

g=sns.swarmplot(x='Sex',y='Fare',hue='Survived',data=train_data).set_title('plot by fare and sex')
train_data.Age.describe()
fig, hrrr = plt.subplots(1,2,figsize=(12,6))

f = sns.distplot(train_data.Age, color='r', ax=hrrr[0]).set_title('Age distribution')

age_ranges = pd.qcut(train_data.Fare, 4, labels = ['Low', 'average', 'High', 'Very high'])

hrrr[1].set_title('Survival rate by age category')

g = sns.barplot(x=age_ranges, y=train_data.Survived, ax=hrrr[1]).set_ylabel('Survival rate')

train_data['Title'] = train_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

test_data['Title'] = test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
train_data['Title'].value_counts()
test_data['Title'].value_counts()
train_data['Title'].replace(['Major','Col','Capt','Don','Jonkheer','Sir'],'Mr',inplace=True)

test_data['Title'].replace(['Major','Col','Capt','Don','Jonkheer','Sir'],'Mr',inplace=True)

train_data['Title'].replace(['Mlle','Lady','Dona','Mme','Ms','the Countess'],'Miss',inplace=True)

test_data['Title'].replace(['Mlle','Lady','Dona','Mme','Ms','the Countess'],'Miss',inplace=True)
train_data.groupby('Title').Survived.mean()
g=sns.barplot(x='Title' ,y='Survived',data=train_data).set_title('Survival by title')
train_data['SibSp'].value_counts()
g=sns.countplot(train_data['SibSp'])
train_data.groupby('SibSp').Survived.mean()
g=sns.barplot(x='SibSp',y='Survived',data=train_data).set_title('Survival by no of sibling and spouse')
fig, hrr = plt.subplots(1,2,figsize=(12,6))

f = sns.distplot(train_data.Fare, color='r', ax=hrr[0]).set_title('Fare distribution')

fare_ranges = pd.qcut(train_data.Fare, 4, labels = ['Low', 'average', 'High', 'Very high'])

hrr[1].set_title('Survival rate by fare category')

g = sns.barplot(x=fare_ranges, y=train_data.Survived, ax=hrr[1]).set_ylabel('Survival rate')

train_data['Parch'].value_counts()
g=sns.countplot(train_data['Parch'])
train_data.groupby('Parch').Survived.mean()
g=sns.barplot(x='Parch',y='Survived',data=train_data)
train_data['Family_size'] = train_data['SibSp'] + train_data['Parch'] + 1

test_data['Family_size'] = test_data['SibSp'] + test_data['Parch'] + 1
g=sns.barplot(x=train_data.Family_size,y=train_data.Survived)
y = train_data['Survived']

features = ['Pclass', 'Sex', 'Fare', 'Title', 'Embarked', 'Family_size']

X = train_data[features]

X.head()
numerical_cols = ['Fare']

categorical_cols = ['Pclass', 'Sex', 'Title', 'Embarked', 'Family_size']





numerical_transformer = SimpleImputer(strategy='median')





categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())

])





preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])





titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(random_state=0, 

                                                               n_estimators=500, max_depth=5))

                             ])





titanic_pipeline.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X, y, cv=10).mean()))
X_test = test_data[features]

X_test.head()
predictions = titanic_pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print('submitted')