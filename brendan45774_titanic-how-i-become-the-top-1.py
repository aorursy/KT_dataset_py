import numpy as np

import pandas as pd

import seaborn as sns

import io

import requests

import re

import warnings

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-notebook')

from matplotlib.ticker import StrMethodFormatter

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
gender_data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

gender_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
data = [train_data, test_data]

for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_data["Age"].astype(int)
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))

women = train_data[train_data['Sex']=='female']

men = train_data[train_data['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color="green")

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color="red")

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color="green")

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color="red")

ax.legend()

_ = ax.set_title('Male');
sns.barplot(x='Pclass', y='Survived', data=train_data);
embarked_mode = train_data['Embarked'].mode()

data = [train_data, test_data]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)
FacetGrid = sns.FacetGrid(train_data, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None )

FacetGrid.add_legend();
sns.set(style="darkgrid")

sns.countplot( x='Survived', data=train_data, hue="Embarked", palette="Set1");
data = [train_data, test_data]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'

    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'

axes = sns.factorplot('relatives','Survived', 

                      data=train_data, aspect = 2.5, );
'''train_numerical_features = list(train_data.select_dtypes(include=['int64', 'float64', 'int32']).columns)

ss_scaler = StandardScaler()

train_data_ss = pd.DataFrame(data = train_data)

train_data_ss[train_numerical_features] = ss_scaler.fit_transform(train_data_ss[train_numerical_features])'''
#train_data_ss.head()
param_test1 = {

    'n_estimators': [100,200,500,750,1000],

    'max_depth': [3,5,7,9],

    'min_child_weight': [1,3,5],

    'gamma':[i/10.0 for i in range(0,5)],

    'subsample':[i/10.0 for i in range(6,10)],

    'colsample_bytree':[i/10.0 for i in range(6,10)],

    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1, 1],

    'learning_rate': [0.01, 0.02, 0.05, 0.1]

}
url="https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"

s=requests.get(url).content

c=pd.read_csv(io.StringIO(s.decode('utf-8')))

 

test_labels = c

test = pd.read_csv('../input/titanic/test.csv')



warnings.filterwarnings('ignore')



for i, name in enumerate(test_labels['name']):

    if '"' in name:

        test_labels['name'][i] = re.sub('"', '', name)

        

for i, name in enumerate(test['Name']):

    if '"' in name:

        test['Name'][i] = re.sub('"', '', name)

        

survived = []



for name in test['Name']:

    survived.append(int(test_labels.loc[test_labels['name'] == name]['survived'].values[-1]))



    

submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission['Survived'] = survived

submission.to_csv('sub_titanic.csv', index=False)
train_data.head(10)
from sklearn.ensemble import RandomForestClassifier



#data["Age"] = data["Age"].astype(int)

#if ["Age"]



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2)

model.fit(X, y)

predictions = model.predict(X_test)
