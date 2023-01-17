# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# change sex

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

# combine train and test data.

combine = pd.concat([train_data,test_data],sort=False)

combine.head()
combine.describe()
train_data.describe()
test_data.describe()
test_data.dtypes
train_data.dtypes
test_data.columns
train_data.columns
# for hitogram

histogram_features = ["Survived", "Age"]

plot_features = train_data[histogram_features]

#features = ["Pclass", "Sex", "SibSp", "Parch"]

#X = pd.get_dummies(train_data[features])

plot_features.head()
# plot histogram

plt.hist(plot_features["Age"], bins = 20)
sns.scatterplot(x=train_data['Age'], y=train_data['Survived'])
women = train_data.loc[train_data.Sex == 'female'] ["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
# show null values in train data

combine.isnull().sum()
combine.notnull().sum()
#train dataset show null values in all columns

print(train_data.isnull().sum())
# frequency of values in embarked feature

combine['Embarked'].value_counts()
# Q18: homework2. fill in missing values in embarked in train dataset with most common occurenece

values = {'Embarked': 'S'}

train_data.fillna(value=values,  inplace=True)

print('After filling NaN values in Embarked')

print(train_data.isnull().sum())

combine['Pclass'].value_counts()
combine['Sex'].value_counts()
combine['Survived'].value_counts()
# convert 'Sex' column to values with 1 and 0

gender = pd.get_dummies(combine['Sex'])

gender.head()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")

