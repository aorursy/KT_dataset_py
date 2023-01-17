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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
# Let's start simple and use a random forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score
train.groupby('Survived')['Cabin'].value_counts()
# start with simple features

y = train.Survived

X = train.loc[:, ['Sex', 'Age', 'Fare', 'Pclass', 'Embarked', 'Cabin']]
# This dataset only has 2 values for gender, let's code them as binary

X.Sex = X.Sex.replace({'male': 1, 'female': 0})

# fill in missing ages by median per gender

X.loc[(X.Sex == 1) & (pd.isna(X.Age)), 'Age'] = X.loc[(X.Sex == 1) & (~pd.isna(X.Age)), 'Age'].median()

X.loc[(X.Sex == 0) & (pd.isna(X.Age)), 'Age'] = X.loc[(X.Sex == 0) & (~pd.isna(X.Age)), 'Age'].median()

# fill Embarked as categorical

X.loc[pd.isna(X.Embarked), 'Embarked'] = 0

X.Embarked = X.Embarked.replace({'S': 1, 'C': 2, 'Q': 3})

X.Embarked = X.Embarked.astype('category')

# grab just the letter/hall of cabin

X.loc[~pd.isna(X.Cabin), 'Cabin'] = X.loc[~pd.isna(X.Cabin), 'Cabin'].apply(lambda x: x[0])
dummies = pd.get_dummies(X.Cabin, drop_first=True, dummy_na=True)

X = pd.merge(left=X, right=dummies, left_index=True, right_index=True, how='left')

X.drop('Cabin', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

baseline = y_train.value_counts(normalize=True).values.tolist()[0]
print(f"Baseline Accuracy is: {baseline}")
rfc = RandomForestClassifier(bootstrap=True, n_estimators=250)

rfc.fit(X_train, y_train)

train_pred = rfc.predict(X_train)
# Let's evaluate training performance

print(confusion_matrix(train_pred, y_train))

print(f'Training accuracy is: {accuracy_score(train_pred, y_train)}')
for value, name in sorted(zip(rfc.feature_importances_, X_train.columns), reverse=True):

    print(f'{name}: {value}')
test_pred = rfc.predict(X_test)

print(confusion_matrix(test_pred, y_test))

print(f'Testing accuracy is: {accuracy_score(test_pred, y_test)}')
# This is actually pretty good out of the box and we can probably stop here.

test = pd.read_csv('/kaggle/input/titanic/test.csv')

X_final = test.loc[:, ['PassengerId', 'Sex', 'Age', 'Fare', 'Pclass', 'Embarked', 'Cabin']]
# Repeat same cleaning process

X_final.Sex = X_final.Sex.replace({'male': 1, 'female': 0})

# fill in missing ages by median per gender

X_final.loc[(X_final.Sex == 1) & (pd.isna(X_final.Age)), 'Age'] = X_final.loc[(X_final.Sex == 1) & (~pd.isna(X_final.Age)), 'Age'].median()

X_final.loc[(X_final.Sex == 0) & (pd.isna(X_final.Age)), 'Age'] = X_final.loc[(X_final.Sex == 0) & (~pd.isna(X_final.Age)), 'Age'].median()

X_final.loc[(X_final.Sex == 1) & (pd.isna(X_final.Fare)), 'Fare'] = X_final.loc[(X_final.Sex == 1) &  (~pd.isna(X_final.Fare)), 'Fare'].median()

X_final.loc[(X_final.Sex == 0) & (pd.isna(X_final.Fare)), 'Fare'] = X_final.loc[(X_final.Sex == 0) &  (~pd.isna(X_final.Fare)), 'Fare'].median()

X_final.loc[pd.isna(X_final.Embarked), 'Embarked'] = 0

X_final.Embarked = X_final.Embarked.replace({'S': 1, 'C': 2, 'Q': 3})

X_final.Embarked = X_final.Embarked.astype('category')

X_final.loc[~pd.isna(X_final.Cabin), 'Cabin'] = X_final.loc[~pd.isna(X_final.Cabin), 'Cabin'].apply(lambda x: x[0])

dummies = pd.get_dummies(X_final.Cabin, drop_first=True, dummy_na=True)

X_final = pd.merge(left=X_final, right=dummies, left_index=True, right_index=True, how='left')

X_final.drop('Cabin', axis=1, inplace=True)
final_prediction = rfc.predict(X_final.iloc[:, 1:])
X_final['Survived'] = final_prediction
output = X_final.loc[:, ['PassengerId', 'Survived']].copy()

output.to_csv('final.csv', index=False)