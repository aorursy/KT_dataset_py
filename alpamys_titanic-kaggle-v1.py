import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pathtr = '/kaggle/input/titanic/train.csv' # path to train.csv

pathte = '/kaggle/input/titanic/test.csv' # path to test.csv

pathsm = '/kaggle/input/titanic/gender_submission.csv' # path to submission sample file



df = pd.read_csv(pathtr) # train dataframe

dff = pd.read_csv(pathte) # test dataframe

submission = pd.read_csv(pathsm) # submission sample
df.shape, dff.shape #check shapes of our train and test tables
df.columns #get column names
df.head() #get a first look at a data by loading head 5 rows of the train DataFrame
df.describe() #describe the dataset to get information about missing values, mean, etc.
df.shape, df.dropna().shape #Let's try to use dropna() and see how much will be lost
df['Pclass'].head() # show first 5 elements
df['Pclass'].isnull().sum() # check if it has missing values
df['Pclass'].value_counts() # get counts of each unique element
df.groupby('Pclass')['Survived'].mean() # get survival rate for each unique element in a feature
df['Name']
df['Name'].head(15)
df['Name'].apply(                             

    lambda x: x.split(', ')[1].split()[0]     

).value_counts()                              
df[df['Name'].apply(lambda x: x.split(', ')[1].split()[0])=='the']
dff['Name'].apply(lambda x: x.split(', ')[1].split()[0]).value_counts() #check if it works on test data
title= {

        'Mr.':'Mr.', 'Sir.':'Mr.', 'Jonkheer.':'Mr.', 'Don.':'Mr.',

        'Mrs.':'Mrs.', 'Mme.':'Mrs.', 'Dona.':'Mrs.', 'the':'Mrs.',

        'Master.':'Master.',

        'Miss.':'Miss.','Ms.':'Miss.', 'Mlle.':'Miss.', 'Lady.':'Miss.',

        'Rev.':'Rev.',

        'Dr.':'Dr.',

        'Col.':'military', 'Major.':'military', 'Capt.':'military'

        }
df['Sex'].head()
df['Sex'].isnull().sum()
df.groupby('Sex')['Survived'].mean()
df['Age'].dropna().hist(bins=20) # show histogram to get a better understanding
df['Age'].isnull().sum()
survival_rate_nan = df[df['Age'].isnull()==True]['Survived'].mean()

survival_rate = df['Survived'].mean()

print(f'Survival rate of NaN-aged is {survival_rate_nan:.3f} and overall is {survival_rate:.3f}')
# let's define mapping function

def new_age(age):

    """

    This function makes categorical feature out of Age. 

    Categories are -> no age, 0-10, 10-20, 20-30, 30-40, 40-50, 50+

    """

    if np.isnan(age)== True:

        return 'no age'

    if age >= 0 and age < 10:

        return '0-10'

    if age >= 10 and age < 20:

        return '10-20'

    if age >= 20 and age < 30:

        return '20-30'

    if age >= 30 and age < 40:

        return '30-40'

    if age >= 40 and age < 50:

        return '40-50'

    return '50+'
for cat in df['Age'].apply(new_age).unique():

    temp_survival = df[df['Age'].apply(new_age)==cat]['Survived'].mean()

    print(f'Age group of |{cat:6}| has {temp_survival:.3f} survival rate')
df['SibSp'].head()
df['SibSp'].isnull().sum()
df['SibSp'].value_counts()
df.groupby('SibSp')['Survived'].mean()
df['SibSp2'] = df['SibSp'].apply(lambda x: x if x<=3 else '3+')
df.groupby('SibSp2')['Survived'].mean()
df['Parch'].head()
df['Parch'].isnull().sum()
df['Parch'].value_counts()
df.groupby('Parch')['Survived'].mean()
df['Parch2'] = df['Parch'].apply(lambda x: x if x<=2 else '2+')
df.groupby('Parch2')['Survived'].mean()
df['isAlone'] = (df['Parch'] + df['SibSp']).apply(lambda x: 1 if x==0 else 0) # SibSp + Parch -> if value is 0, then alone
df.groupby('isAlone')['Survived'].mean() # let's check if it's helpful
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['FamilySize'].hist()
df['FamilySize'].value_counts()
df.groupby('FamilySize')['Survived'].mean()
df['Ticket'].head()
df['Ticket'].isnull().sum()
fun = df['Ticket'].apply(lambda x: len(x.split()))

for i in fun.unique():

    print(i,df[fun==i]['Survived'].mean(), df['Survived'].mean())
df['Fare'].head()
df['Fare'].isnull().sum()
df['Fare'].hist(bins=100)
df['Fare'].describe()
def new_fare(fare):

    """

    This function maps Fare.

    Categories are -> 0-8, 8-15, 15-31, 31+

    """

    if fare <= 8:

        return '0-8'

    if fare > 8 and fare <= 15:

        return '8-15'

    if fare > 15 and fare <=31:

        return '15-31'

    return '31+'
fare_cat = df['Fare'].apply(new_fare)

print(fare_cat.value_counts(), '\n')

for cat in fare_cat.unique():

    cat_survival = df[fare_cat==cat]['Survived'].mean()

    print(f'Fare group |{cat:5}| had {cat_survival:.3f} survival rate')
df['Cabin'].head()
df['Cabin'].isnull().sum()
df[df['Cabin'].isnull()]['Survived'].mean()
df[~df['Cabin'].isnull()]['Survived'].mean()
def new_cabin(cabin):

    """

    This function will map cabin to -> nan and not nan

    """

    if str(cabin)=='nan':

        return 'nan'

    return 'not nan'
df['Cabin'].apply(new_cabin).value_counts()
df['Embarked'].head()
df['Embarked'].isnull().sum()
df.groupby('Embarked')['Survived'].mean()
train_df = pd.read_csv(pathtr) # train dataframe

test_df = pd.read_csv(pathte) # test dataframe
print('TRAIN')

for col in train_df.columns:

    print(f'{col:11} has {train_df[col].isnull().sum():3} missing values')

print('\nTEST')

for col in test_df.columns:

    print(f'{col:11} has {test_df[col].isnull().sum():3} missing values')
test_df[test_df['Fare'].isnull()]
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
def preprocess(train,test):

    result = []

    for df in [train,test]:

        df = df.loc[df['Embarked'].dropna().index]

        df['Title'] = df['Name'].apply(lambda x: x.split(', ')[1].split()[0]).map(title)

        df['AgeGr'] = df['Age'].apply(new_age)

        df['SibSp2'] = df['SibSp'].apply(lambda x: x if x<=3 else '3+')

        df['Parch2'] = df['Parch'].apply(lambda x: x if x<=2 else '2+')

        df['isAlone'] = (df['Parch'] + df['SibSp']).apply(lambda x: 1 if x==0 else 0)

        df['FareGr'] = df['Fare'].apply(new_fare)

        df['Cabin2'] = df['Cabin'].apply(new_cabin)

        df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'Age'], axis=1, inplace=True)

        result.append(df)

    return result
train2_df, test2_df = preprocess(train_df, test_df)
train2_df
test2_df
train2_df.shape, train2_df.dropna().shape
train_X = train2_df.drop('Survived', axis=1)

train_y = train2_df['Survived']
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

def get_cv(X, y, model):

    X1 = pd.get_dummies(X)

    scores = cross_val_score(model, X1, y, cv=5)

    return scores
model_rf = RandomForestClassifier(n_estimators=141, random_state=999)

get_cv(train_X, train_y, model_rf).mean()
model_rf.fit(pd.get_dummies(train_X), train_y)
y_pred = model_rf.predict(pd.get_dummies(test2_df))
submission.head()
submission['Survived'] = y_pred
submission.to_csv('first_try.csv', index=False)