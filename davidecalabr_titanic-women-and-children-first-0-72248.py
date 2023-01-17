import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # plot data

import matplotlib.pyplot as plt # plot data

import os



from sklearn.tree import DecisionTreeClassifier # Algorithm
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



train_data['dataset'] = 'train'

test_data['dataset'] = 'test'



all_data = pd.concat([train_data, test_data])
all_data.head()
all_data.describe()
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
all_data['title'] = all_data['Name'].str.extract(r'([A-z]+)\.', expand=True)

all_data['title'].unique()
plt.figure(figsize=(16, 8))



for title in all_data['title'].unique():

    f1 = all_data['title'] == title

    sns.distplot(all_data[f1]['Age'], label=title, kde=False)



plt.legend()
f = all_data['Age'].isnull() == True

all_data[f]['title'].unique()
plt.figure(figsize=(16, 8))



for title in all_data[f]['title'].unique():

    f1 = all_data['title'] == title

    sns.distplot(all_data[f1]['Age'], label=title, kde=False)



plt.legend()
for title, age in all_data.groupby('title')['Age'].median().iteritems():

    

    if title in all_data[f]['title'].unique():

        print(title, age)

        

    all_data.loc[(all_data['title']==title) & (all_data['Age'].isnull()), 'Age'] = age
all_data = all_data.drop(['Cabin', 'Embarked', 'Fare'], axis=1)
all_data.head()
all_data.describe()
all_data_simple = all_data.drop(['Pclass', 'Name', 'SibSp', 'Parch', 'Ticket', 'title'], axis=1)
all_data_simple.head()
all_data_simple.describe()
train_filter = all_data_simple['dataset'] == 'train'

test_filter = all_data_simple['dataset'] == 'test'



train_set = all_data_simple[train_filter].drop(['dataset'], axis=1).reset_index(drop=True)

test_set = all_data_simple[test_filter].drop(['dataset'], axis=1).reset_index(drop=True)
train_set = pd.get_dummies(train_set)

test_set = pd.get_dummies(test_set)



train_set = train_set.set_index('PassengerId')

test_set = test_set.set_index('PassengerId')
train_set.head()
train_set.describe()
test_set.head()
test_set.describe()
features = ['Age', 'Sex_female', 'Sex_male']

target = ['Survived']
titanic_model = DecisionTreeClassifier(random_state=0)



X = train_set[features]

y = train_set[target]



# Fit the model

titanic_model.fit(X, y)
X = test_set[features]



predictions = titanic_model.predict(X)



d = {'PassengerId': X.index, 'Survived': predictions.astype(int)}

result = pd.DataFrame(data=d)



result
result.to_csv('results.csv', index=False)