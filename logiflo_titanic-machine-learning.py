# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_path = '../input/titanic/test.csv'

data_test_titanic = pd.read_csv(test_path)





train_path = '../input/titanic/train.csv'

data_train_titanic = pd.read_csv(train_path)

data_train_titanic.head()
data_train_titanic.info()
data_train_titanic.isnull().sum()
data_train_titanic['Age'].fillna((data_train_titanic['Age'].mean()), inplace=True)

data_train_titanic['Embarked'].fillna((data_train_titanic['Embarked'].mode()[0]), inplace=True)
data_train_titanic['With_cabin'] = data_train_titanic['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
data_train_titanic.isnull().sum()
print(data_train_titanic.Sex.unique())

print(data_train_titanic.Embarked.unique())
cleanup_nums = {'Sex':     {'male': 0, 'female': 1},

                'Embarked': {'S': 0, 'C': 1, 'Q': 2}}
data_train_titanic.replace(cleanup_nums, inplace=True)

print(data_train_titanic.Sex.unique())

print(data_train_titanic.Embarked.unique())
data_test_titanic.info()
data_test_titanic.isnull().sum()
data_test_titanic['Age'].fillna((data_test_titanic['Age'].mean()), inplace=True)

data_test_titanic['Fare'].fillna((data_test_titanic['Fare'].mean()), inplace=True)

data_test_titanic['With_cabin'] = data_test_titanic['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
data_test_titanic.isnull().sum()
data_test_titanic.replace(cleanup_nums, inplace=True)

print(data_test_titanic.Sex.unique())

print(data_test_titanic.Embarked.unique())
plt.rcParams['figure.figsize'] = (18, 7)



plt.subplot(1, 2, 1)

sns.distplot(data_train_titanic.Age)

plt.title('Age distribution', fontsize = 20)



plt.subplot(1, 2, 2)

sns.countplot(data_train_titanic.Sex)

plt.title('Number of men and women', fontsize = 20)



plt.show()


grid = sns.FacetGrid(data_train_titanic, row="Sex", col="Survived", margin_titles=True, size=4)

grid.map(plt.hist, "Age", bins=np.linspace(0, 40, 15));

plt.rcParams['figure.figsize'] = (7, 7)

sns.countplot(x = "Pclass", hue='Survived', data = data_train_titanic)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
data_train_titanic.info()
y = data_train_titanic.Survived

# Create X

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'With_cabin']

X = data_train_titanic[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = RandomForestRegressor(random_state=1)



rf_model_on_full_data.fit(X, y)
test_X = data_test_titanic[features]



test_preds = rf_model_on_full_data.predict(test_X)



output = pd.DataFrame({'PassengerId': data_test_titanic.PassengerId,

                       'Survived': test_preds})

output.to_csv('submission.csv', index=False)