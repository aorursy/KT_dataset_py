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
data = pd.read_csv('/kaggle/input/titanic/train.csv', encoding='utf-8')

data.head()
data.columns
label = data['Survived']

len(label)
set(data['Cabin'])
features = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].replace('male', 0).replace('female', 1)

features.shape
features.head()
features.Cabin.isna().sum()
features.drop(columns=['Cabin'], inplace=True)

features.tail(5)
features.Pclass.value_counts()
features.Sex.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(12,7))

sns.distplot(features['Fare'])

plt.title('Ticket Fare Dist')

plt.xlabel('Price ($)')
x = features.groupby('Pclass')['Fare'].mean()



plt.figure(figsize=(8,5))

plt.bar(x.index, x, color='plum', edgecolor='k')

plt.title('Pclass and Ticket Fare')
x = data.groupby('Pclass')['Survived'].sum()

y = data.groupby('Pclass')['Survived'].count()

z = pd.merge(x, y, left_index=True, right_index=True)

z['rate'] = z['Survived_x']*100 / z['Survived_y']

z
features.head()
one_hot = pd.get_dummies(features['Pclass'])

enc_features = features[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']].merge(one_hot, left_index=True, right_index=True)

enc_features.head()
import missingno as msno

msno.matrix(enc_features)
enc_features['Age'].isna().sum()
enc_features[enc_features['Age'].isna()]
for_family = features.dropna()[['Age', 'SibSp', 'Parch', 'Fare']]

t = for_family.groupby('SibSp')['Age'].mean()

u = for_family.groupby('Parch')['Age'].mean()

v = for_family.groupby('Age')['Fare'].mean()

# plt.figure(figsize=(11,7))

# plt.bar(t.index, t, color='dodgerblue')

plt.bar(v.index, v, color='plum')
for_family = features.dropna()[['Age', 'SibSp', 'Parch', 'Fare']]

t = for_family.groupby('SibSp')['Age'].mean()

u = for_family.groupby('Parch')['Age'].mean()

v = for_family.groupby('Age')['Fare'].mean()

plt.figure(figsize=(11,7))

plt.bar(t.index, t, color='dodgerblue')

plt.plot(t, color='gray')

# plt.bar(v.index, v, color='plum')

plt.title("SibSp - Age : For Estimation")
age_estimation = pd.DataFrame(t).reset_index()

mer_features = enc_features.merge(age_estimation, how='left', on='SibSp')

mer_features['Age_y'].fillna(10.2, inplace=True)

problem = mer_features[mer_features['Age_y'].isna()].index

mer_features[mer_features.index.isin(problem)]['Age_y'].isna()
mer_features['Age_y'].isna().sum()
mer_features.tail()
dict_temp = dict()



for ind, val in mer_features.iterrows():

    if np.isnan(val['Age_x']):

        dict_temp[ind] = val['Age_y']

    else:

        dict_temp[ind] = val['Age_x']

dict_temp[888]
for_age = pd.DataFrame({'Age': list(dict_temp.values())}, index=list(dict_temp.keys()))
our_features = mer_features.merge(for_age, how='left', left_index=True, right_index=True).drop(columns=['Age_x', 'Age_y'])

our_features
msno.matrix(our_features)
from sklearn.preprocessing import StandardScaler



I = np.array(our_features['Age']).reshape(-1,1)

T = np.array(our_features['Fare']).reshape(-1,1)



scaler1 = StandardScaler()

scaler1.fit(I)

age_scl = scaler1.transform(I)

scaler2 = StandardScaler()

scaler2.fit(T)

fare_scl = scaler2.transform(T)
age_scl.mean(), fare_scl.mean()
final_features = our_features.drop(columns=['Fare', 'Age'])

final_features['Fare'] = fare_scl

final_features['Age'] = age_scl

final_features.head()
from sklearn.preprocessing import MinMaxScaler as Mmscaler



scaler = Mmscaler()

scaler2 = Mmscaler()

x1 = np.array(final_features['SibSp']).reshape(-1,1)

x2 = np.array(final_features['Parch']).reshape(-1,1)



scaler.fit(x1)

scaler2.fit(x2)



sibsp_scl = scaler.transform(x1)

parch_scl = scaler2.transform(x2)
sibsp_scl.mean(), parch_scl.mean()
final_features = final_features.drop(columns=['SibSp', 'Parch'])

final_features['SibSp'] = sibsp_scl

final_features['parch_scl'] = parch_scl

final_features.head()
label[:5]
test = pd.read_csv('/kaggle/input/titanic/test.csv', encoding='utf-8')

test.head()
msno.matrix(test)
age_estimation.head(3)
test.head(3)
test_ = test.merge(age_estimation, how='left', on='SibSp')

test_['Age_y'].fillna(10.2, inplace=True)



dict_temp = dict()



for ind, val in test_.iterrows():

    if np.isnan(val['Age_x']):

        dict_temp[ind] = val['Age_y']

    else:

        dict_temp[ind] = val['Age_x']

        

tpd = pd.DataFrame({'Age': list(dict_temp.values())}, index=list(dict_temp.keys()))



test__ = test_.merge(tpd, how='left', left_index=True, right_index=True).drop(columns=['Age_x', 'Age_y'])

test__.head()
msno.matrix(test__)
test__.isna().sum()
test__.groupby('Pclass')['Fare'].mean()
test__[test__['Fare'].isna()]
test = test.replace(np.nan, 12.459678)

test.isna().sum()
test = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].replace('male', 0).replace('female', 1)

x = pd.get_dummies(test['Pclass'])

x1 = np.array(test['Age']).reshape(-1,1)

x2 = np.array(test['Fare']).reshape(-1,1)

y1 = np.array(test['SibSp']).reshape(-1,1)

y2 = np.array(test['Parch']).reshape(-1,1)



for_x1 = StandardScaler()

for_x2 = StandardScaler()

for_y1 = Mmscaler()

for_y2 = Mmscaler()



for_x1.fit(x1)

for_y1.fit(y1)

for_x2.fit(x2)

for_y2.fit(y2)



xx1 = for_x1.transform(x1)

xx2 = for_x2.transform(x2)

yy1 = for_y1.transform(y1)

yy2 = for_y2.transform(y2)



final_test = test[['PassengerId', 'Sex']].merge(x, how='left', left_index=True, right_index=True)

final_test['Age'] = xx1

final_test['Fare'] = xx2

final_test['SibSp'] = yy1

final_test['Parch'] = yy2



final_test.head()
final_test_features = final_test[['Sex', 1,2,3,'Age','Fare','SibSp','Parch']]

final_ids = final_test[['PassengerId']]

final_test_features.head()
from sklearn.ensemble import GradientBoostingClassifier as GBC

model = GBC(n_estimators=120)

model.fit(final_features, label)
model.feature_importances_
final_features.columns
predictions = model.predict(final_test_features)
predictions
final_ids.head()
submission = final_ids

submission['Survived'] = predictions

submission.sample(5)
pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.shape
submission.head()
submission.to_csv('Titanic_submission.csv')
submission.columns
submission.head()
submission.to_csv('submission_file.csv', index=False)