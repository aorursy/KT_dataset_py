# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import Data

titanic_train = pd.read_csv('../input/train.csv',index_col = 'PassengerId')
titanic_test = pd.read_csv('../input/test.csv', index_col = 'PassengerId')

#Pre-process data & Combine test & Train Data

# save the result
train_results = titanic_train['Survived'].copy()
test_results = pd.read_csv('../input/gender_submission.csv', index_col='PassengerId')

titanic_train.drop('Survived', axis=1, inplace=True, errors='ignore')

# combine Train & Test set
titanic = pd.concat([titanic_train, titanic_test])

# Save Train & Test Index
train_index = titanic_train.index
test_index = titanic_test.index

titanic.info()
plt.figure(figsize=(8,4))
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Try to fill the Age with the mean but base on their 'Pclass'

c1_age_mean =round(titanic[titanic['Pclass'] == 1]['Age'].mean())
c2_age_mean =round(titanic[titanic['Pclass'] == 2]['Age'].mean())
c3_age_mean =round(titanic[titanic['Pclass'] == 3]['Age'].mean())

print(c1_age_mean, c2_age_mean, c3_age_mean)
t1 = titanic[titanic['Pclass'] == 1]
t2 = titanic[titanic['Pclass'] == 2]
t3 = titanic[titanic['Pclass'] == 3]

t1['Age'].fillna(c1_age_mean, inplace=True)
t2['Age'].fillna(c2_age_mean, inplace=True)
t3['Age'].fillna(c3_age_mean, inplace=True)
titanic = pd.concat([t1, t2, t3])
plt.figure(figsize=(8,5))
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
nan_total = titanic[titanic['Cabin'].isnull()]['Pclass'].count()
nan_cabin1 = titanic[(titanic['Pclass'] ==1) & (titanic['Cabin'].isnull())]['Pclass'].count()
nan_cabin2 = titanic[(titanic['Pclass'] ==2) & (titanic['Cabin'].isnull())]['Pclass'].count()
nan_cabin3 = titanic[(titanic['Pclass'] ==3) & (titanic['Cabin'].isnull())]['Pclass'].count()

total = titanic['Pclass'].count()
total_cabin1 = titanic[titanic['Pclass'] == 1]['Pclass'].count()
total_cabin2 = titanic[titanic['Pclass'] == 2]['Pclass'].count()
total_cabin3 = titanic[titanic['Pclass'] == 3]['Pclass'].count()

print('Total: nan, total, percentage',nan_total, total, "{:.2%}".format(nan_total/total))
print('class1: nan, total, percentage',nan_cabin1, total_cabin1, "{:.2%}".format(nan_cabin1/total_cabin1))
print('class2: nan, total, percentage',nan_cabin2, total_cabin2, "{:.2%}".format(nan_cabin2/total_cabin2))
print('class3: nan, total, percentage',nan_cabin3, total_cabin3, "{:.2%}".format(nan_cabin3/total_cabin3))

#Make a copy of "cabin" Field, Process it indivdually and combine it back to dataset later.

cabin_only = titanic[['Cabin']].copy()
cabin_only['Cabin'].unique()  # check out what is the values look like
## Get the Deck info

def getdeckinfo(cabin):
    deck =[]
    if ' ' not in str(cabin):             # Process for only one info
        deck = cabin[0]
    else:                                # Process for the field have many different cabin info 
        cabin_list = cabin.split()
        for i in range(len(cabin_list)): 
            deck += cabin_list[i][0]
        deck = ''.join(deck)
    return deck

cabin_only['Deck'] = cabin_only[cabin_only['Cabin'].notnull()]['Cabin'].apply(getdeckinfo)
cabin_only['Deck'].unique()
##  Get the Room info 

#only extract first room no, I actually not very sure whether this can help
cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float") 

cabin_only.head() 
# deal with NaN Value

cabin_only["Deck"] = cabin_only["Deck"].fillna("N")
cabin_only["Room"] = cabin_only["Room"].fillna(cabin_only["Room"].mean())

cabin_only.drop(['Cabin'], axis=1, inplace=True, errors='ignore')
cabin_only.info()
# One Hot Column function (Convert categorical variable into dummy/indicator variables )

def one_hot_column(df, label, drop_col=False):
    '''
    This function will one hot encode the chosen column.
    Args:
        df: Pandas dataframe
        label: Label of the column to encode
        drop_col: boolean to decide if the chosen column should be dropped
    Returns:
        pandas dataframe with the given encoding
    '''
    one_hot = pd.get_dummies(df[label], prefix=label)
    if drop_col:
        df = df.drop(label, axis=1)
    df = df.join(one_hot)
    return df


def one_hot(df, labels, drop_col=False):
    '''
    This function will one hot encode a list of columns.
    Args:
        df: Pandas dataframe
        labels: list of the columns to encode
        drop_col: boolean to decide if the chosen column should be dropped
    Returns:
        pandas dataframe with the given encoding
    '''
    for label in labels:
        df = one_hot_column(df, label, drop_col)
    return df

cabin_only = one_hot(cabin_only, ['Deck'], drop_col=True)
cabin_only.head()
for column in cabin_only.columns:
    titanic[column] = cabin_only[column]
    
titanic.drop('Cabin', axis=1, inplace=True)
titanic.info()
# drop "Name" & "Ticket" First

titanic.drop(['Name', 'Ticket'], inplace=True, axis=1)
# fill the empty field in "Embarked" first

titanic['Embarked'].fillna('N', inplace=True)
# Transfor "Sex" & "Embarked" Fields to Numerical and drop the original fields

titanic = one_hot(titanic, ['Sex', 'Embarked'], drop_col=True ) # re-use the function in defined in "Cabin" Section.
titanic.head()
# Drop "Sex_male" Since it is 100% correlated to "Sex_female" (ask, duplicate features)
titanic.drop(['Sex_male','Embarked_N'] , axis=1, inplace=True)

# Drop "Room" as well, I found it acutally help increase 1% of the performance by removing it
titanic.drop('Room', axis=1, inplace=True)
# final minor fix for "Fare" field

titanic['Fare'].fillna(titanic['Fare'].mean(), inplace=True)
titanic.info()
# Use the original training & Testing Index

X_train = titanic.loc[train_index, :]
y_train = train_results

X_test = titanic.loc[test_index, :]
y_test = test_results
# We can pontentially seperate the dataset by ourself, compare the performance vs. the original one.
# Will do it later
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Build and train the model with Training Data Set
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
# See the performance with Testing Data Set

predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))

