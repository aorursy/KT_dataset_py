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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
data = [train, test]
for dataset in data:

    missing = dataset.isnull().sum()

    print(missing)
gender = {"male": 0, "female": 1}



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(gender)
table = pd.crosstab(train['Survived'],train['Pclass'])

print(table)
table = pd.crosstab(train['Survived'],train['Sex'])

print(table)
table = pd.crosstab(train['Pclass'],train['Sex'])

print(table)
train['Survived'].value_counts().plot.bar( figsize=(10, 10))
train['Survived'].value_counts().plot.bar( figsize=(10, 10))
train['Embarked'].value_counts().plot.pie(figsize=(10, 10))
train['Pclass'].value_counts().plot.pie( figsize=(10, 10))
age_groups = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

train['age_group'] = pd.cut(train.Age, range(0, 81, 10), right=False, labels=age_groups)

train['age_group'].value_counts().plot.pie(figsize=(10, 10))
fare_groups = ['0-49', '50-99', '100-149', '150-199', '200-249', '250-299', '300-349', '350-399','400-449', '450-499', '500-549']

train['fare_group'] = pd.cut(train.Fare, range(0, 600, 50), right=False, labels=fare_groups)

train['fare_group'].value_counts().plot.pie(figsize=(10, 10))
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Professional": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Professional')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)
frames = [train, test]

result = pd.concat(frames, sort=False)

result.head(20)
new_age = result.groupby(['Sex','Title'])['Age'].mean()

print(new_age)
def replace_age_nan(row):

    '''

    function to check if the age is null and replace wth the mean from 

    the mean ages dataframe 

    '''

    if pd.isnull(row['Age']):

        return new_age[row['Sex'],row['Title']]

    else:

        return row['Age']



train['Age'] =train.apply(replace_age_nan, axis=1)

test['Age'] = test.apply(replace_age_nan, axis=1)
for dataset in data:

    missing = dataset.isnull().sum()

    print(missing)
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp','age_group', 'fare_group']



fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cat_features, 1):    

    plt.subplot(3, 3, i)

    sns.countplot(x=feature, hue='Survived', data=train)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
train.corr()

corrMatrix = train.corr()

sns.heatmap(corrMatrix, annot=True)
train.corr()
train['Ticket'].describe()
train['Ticket'].head(20)
X_train = train.drop(columns=['PassengerId','Name','Ticket','Fare','Cabin','Embarked','age_group','fare_group'], axis=1, inplace=True)
X_train = train
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop(["PassengerId",'Name','Ticket','Fare','Cabin','Embarked'], axis=1).copy()

X_train.head()
X_test
forest = RandomForestClassifier(n_estimators=100, random_state =57)

forest.fit(X_train, Y_train)



forest_accuracy = cross_val_score(forest, X_train, Y_train,  cv = 20).mean()



    

forest.score(X_train, Y_train)

Random_forest = round(forest.score(X_train, Y_train) * 100, 3)

print('Model: Random Forest Accuracy: ',Random_forest)
Y_prediction = forest.predict(X_test)

print(Y_prediction)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':Y_prediction})

submission.head()