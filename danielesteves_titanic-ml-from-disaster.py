# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()
df_train.info()
df_test.info()
df_train.isnull().sum()
def split_ages(age):

    if(age < 18): return 0

    elif (age <= 35): return 1

    elif (age <= 65): return 2

    else: return 3



def prepare_data(_df, drop_lines=True):

    #Drop Cabin

    output = _df.copy()

    output = output.drop('Cabin', axis=1)



    #Fill NaN Ages with mean of Sex and Pclass

    mean_age = output.groupby(['Sex','Pclass'])['Age'].mean()

    output['Age'] = output.apply(lambda x : mean_age.loc[(x.Sex, x.Pclass)] if np.isnan(x.Age) else x.Age, axis=1)

    

    output['Fare'] = output.apply(lambda x : output.groupby(['Pclass'])['Fare'].mean().loc[x.Pclass] if np.isnan(x.Fare) else x.Fare, axis=1)

    

    output = output.drop(['Name', 'Ticket'], axis=1)



    #Drop rows with Embarked NaN

    if(drop_lines):

        output = output.dropna()

     

    output = output.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}})

    

    #clustering ages

    output['Age_Group'] = output['Age'].map(split_ages)

    

    return output



train = prepare_data(df_train)

test  = prepare_data(df_test, drop_lines=False)

train.head()
train.info()
#sns.heatmap(train.corr())

train.corr()
def family_size(row):

    return row['SibSp'] + row['Parch']



t2 = train.copy()

t2['Family'] = t2.apply(family_size, axis=1)



t2.corr().Survived
test.isnull().sum()
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Age_Group']

X = train.drop(['Survived'], axis=1)[cols]

y = train['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



clf = RandomForestClassifier(n_estimators=10, min_samples_split=5)

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test[cols])

accuracy_score(y_pred, y_test)
X = train.drop(['Survived','PassengerId'], axis=1)[cols]

Y = train['Survived']
clf = RandomForestClassifier(n_estimators=10, min_samples_split=5)

clf = clf.fit(X, Y)
prediction = clf.predict(test.drop(['PassengerId'], axis=1)[cols])
output = test[['PassengerId']]

output['Survived'] = prediction
output.to_csv('gender_submission.csv', index=False)
