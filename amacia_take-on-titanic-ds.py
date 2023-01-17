# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
t_train_data = pd.read_csv('../input/titanic/train.csv')

t_test_data = pd.read_csv('../input/titanic/test.csv')
t_train_data.info()
t_train_data.describe()
t_train_data.head()
sns.catplot(x='Pclass', y="Age", hue="Sex", data=t_train_data, kind="box")
sns.countplot(x='Pclass', data = t_train_data)
sns.countplot(x='Embarked', hue = 'Pclass', data=t_train_data)
sns.boxplot(y='Fare', x='Embarked', hue='Pclass', data=t_train_data)
def cabin_extract_deck(cabin_string):

    if isinstance(cabin_string, str):

        cabin = cabin_string[0]

        return cabin

    else:

        return 'Unknown'

    

def cabin_extract_number(cabin_string):

    if isinstance(cabin_string, str):

        cabin_string = cabin_string.split()[0]

        if len(cabin_string) > 1:

            cab_num = int(cabin_string[1:])

            return cab_num

        else:

            return None

    else:

        return None
t_train_data['Deck'] = t_train_data['Cabin'].apply(lambda x: cabin_extract_deck(x))

sns.countplot(t_train_data['Deck'], hue=t_train_data['Pclass'])
sns.boxplot(x='Deck', y ='Fare', data=t_train_data, hue='Pclass')
t_train_data['Cab_num'] = t_train_data['Cabin'].apply(lambda x: cabin_extract_number(x))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score
X_m1 = t_train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']]

y = t_train_data['Survived']
X_m1['Age'].fillna(X_m1['Age'].mean(), inplace=True)

X_m1['Sex'] = pd.get_dummies(X_m1['Sex'], drop_first=True)

X_m1
X_train, X_test, y_train, y_test = train_test_split(X_m1, y, random_state=0)
lm_m1 = LogisticRegression()
lm_m1.fit(X_train, y_train)
X_test_predicion = lm_m1.predict(X_test)

print("Model 1")

print('acc',accuracy_score(X_test_predicion, y_test))

print('f1',f1_score(X_test_predicion, y_test))
avg_age_sclass = t_train_data.groupby(['Sex', 'Pclass']).mean()['Age']
def age_imputer(row):

    return avg_age_sclass[row['Sex']][row['Pclass']]
X_m2 = t_train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Sex', 'Fare']]

vals = X_m2[X_m2['Age'].isnull()].apply(lambda x: age_imputer(x), axis=1)

X_m2['Age'].fillna(vals, inplace=True)

X_m2['Sex'] = pd.get_dummies(X_m1['Sex'], drop_first=True)

X_m2.describe()
X_train, X_test, y_train, y_test = train_test_split(X_m2, y, random_state=0)

lm_m2 = LogisticRegression()

lm_m2.fit(X_train, y_train)
X_test_prediction = lm_m2.predict(X_test)

score_lm_m2 = accuracy_score(X_test_prediction, y_test)

print("Model 2")

print('acc',score_lm_m2)

print('f1',f1_score(X_test_prediction, y_test))
from sklearn.preprocessing import OneHotEncoder

ohe_Embarked = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_m3 = t_train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Sex', 'Fare', 'Embarked']]

vals = X_m3[X_m3['Age'].isnull()].apply(lambda x: age_imputer(x), axis=1)

X_m3['Age'].fillna(vals, inplace=True)

X_m3['Sex'] = pd.get_dummies(X_m3['Sex'], drop_first=True)

X_m3['Embarked'].fillna('S', inplace=True)

X_m3['Embarked'].head()
OH_cols = pd.DataFrame(ohe_Embarked.fit_transform(np.array(X_m3['Embarked']).reshape(-1,1)))

#ohe_Embarked.categories_

OH_cols.columns= ['C', 'Q', 'S']

X_m3 = pd.concat([X_m3,OH_cols], axis=1)

X_m3.drop('Embarked', axis=1, inplace=True)
X_m3
X_train, X_test, y_train, y_test = train_test_split(X_m3, y, random_state=0)

lm_m3 = LogisticRegression(max_iter=500)

lm_m3.fit(X_train, y_train)

X_test_prediction = lm_m3.predict(X_test)

score_lm_m3 = accuracy_score(X_test_prediction, y_test)

print("Model 3")

print('ac',score_lm_m3)

print('f1', f1_score(X_test_prediction, y_test))
X_m4 = t_train_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Sex', 'Fare', 'Embarked', 'Deck']]

vals = X_m4[X_m4['Age'].isnull()].apply(lambda x: age_imputer(x), axis=1)

X_m4['Age'].fillna(vals, inplace=True)

X_m4['Sex'] = pd.get_dummies(X_m4['Sex'], drop_first=True)

X_m4['Embarked'].fillna('S', inplace=True)

X_m4
OH_cols = pd.DataFrame(ohe_Embarked.fit_transform(np.array(X_m4['Embarked']).reshape(-1,1)))

#ohe_Embarked.categories_

OH_cols.columns= ['C', 'Q', 'S']

X_m4 = pd.concat([X_m4,OH_cols], axis=1)

X_m4.drop('Embarked', axis=1, inplace=True)
ohe_deck = OneHotEncoder(sparse=False, handle_unknown='ignore')

oh_cols_deck = pd.DataFrame(ohe_deck.fit_transform(np.array(X_m4['Deck']).reshape(-1,1)))

oh_cols_deck.columns = ['DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG','DT', 'DUnknown']



X_m4 = pd.concat([X_m4, oh_cols_deck], axis=1)

X_m4.drop('Deck', axis=1, inplace=True)

X_m4
X_train, X_test, y_train, y_test = train_test_split(X_m4, y, random_state=0)

lm_m4 = LogisticRegression(max_iter=500)

lm_m4.fit(X_train, y_train)

X_test_prediction = lm_m4.predict(X_test)

score_lm_m4 = accuracy_score(X_test_prediction, y_test)

print('ac',score_lm_m4)

print('f1', f1_score(X_test_prediction, y_test))
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier



gnb_c = GaussianNB()

rf_c = RandomForestClassifier(n_estimators=100, max_depth=10)

grad_b_c = GradientBoostingClassifier()
gnb_c.fit(X_train,y_train)

rf_c.fit(X_train, y_train)

grad_b_c.fit(X_train, y_train)
print('ac nb',accuracy_score(gnb_c.predict(X_test), y_test))

print('ac rf',accuracy_score(rf_c.predict(X_test), y_test))

print('ac grb',accuracy_score(grad_b_c.predict(X_test), y_test))

print('f1 nb',f1_score(gnb_c.predict(X_test), y_test))

print('f1 rf',f1_score(rf_c.predict(X_test), y_test))

print('f1 grb',f1_score(grad_b_c.predict(X_test), y_test))

#t_test_data['Deck'] = t_test_data['Cabin'].apply(lambda x: cabin_extract_deck(x))

#kaggle_input = t_test_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Deck']]

#vals = kaggle_input[kaggle_input['Age'].isnull()].apply(lambda x: age_imputer(x), axis=1)

#kaggle_input['Age'].fillna(vals, inplace=True)

#kaggle_input['Sex'] =pd.get_dummies(kaggle_input['Sex'], drop_first=True)

#oh_cols = pd.DataFrame(ohe_Embarked.transform(np.array(kaggle_input['Embarked']).reshape(-1,1)))

#oh_cols.columns=['C', 'Q', 'S']

#oh_cols_deck = pd.DataFrame(ohe_deck.transform(np.array(kaggle_input['Deck']).reshape(-1,1)))

#oh_cols_deck.columns = ['DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG','DT', 'DUnknown']

#kaggle_input = pd.concat((kaggle_input, oh_cols,oh_cols_deck), axis=1)

#kaggle_input.drop(['Embarked', 'Deck'], inplace=True, axis=1)

# Model 1

kaggle_input = t_test_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']]

kaggle_input['Sex'] =pd.get_dummies(kaggle_input['Sex'], drop_first=True)

kaggle_input['Age'].fillna(X_m1['Age'].mean(), inplace=True)

kaggle_input
kaggle_input[kaggle_input.isna().any(axis=1)]
kaggle_input['Fare'].fillna(X_m1[X_m1['Pclass']==3]['Fare'].mean(), inplace=True)
kaggle_ref = lm_m1.predict(kaggle_input)

kaggle_output = lm_m1.predict(kaggle_input)
result = pd.DataFrame({'PassengerId': t_test_data['PassengerId'], 'Survived': kaggle_output})
result.to_csv('result.csv',index=False)
pd.read_csv('../input/titanic/gender_submission.csv')