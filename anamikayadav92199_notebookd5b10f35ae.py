# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
data.corr()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.heatmap(data.corr())

plt.show()
sns.set_style('dark')

sns.set_palette('RdBu')

sns.set_context('poster')

sns.catplot(x = 'Survived',data=data, kind='count')

plt.show()
sns.set_palette(['Blue','Black'])

sns.catplot(x = 'Pclass',data=data, kind='count',hue='Survived')

plt.show()
# Count of Survived people of each Sex

sns.set_palette(['Blue','Black'])

sns.catplot(x = 'Sex', data = data, kind='count', hue='Survived')

plt.show()
# Distribution of Age among the Survived people

sns.set_context('notebook')

sns.catplot(x = 'Survived', y='Age', data=data,kind='box')

plt.show()
# Relation between the survived people and their fare

sns.catplot(x = 'Survived', y='Fare', data=data, kind='bar')

plt.show()
# Count of Survived people from each Embarking

sns.catplot(x = 'Embarked', data = data, kind='count', hue='Survived')

plt.show()
data.isnull().sum()
data['Age'] = data['Age'].fillna(data['Age'].mean())

data['Cabin'] = data['Cabin'].fillna('Missing')

data = data.dropna()
data.isnull().sum()
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
data_test.isnull().sum()
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())

data_test['Cabin'] = data_test['Cabin'].fillna('Missing')

data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].mean())
data_test.isnull().sum()
data = data.drop(columns=['Name'],axis=1)



data = data.drop(columns=['Ticket'],axis=1)



data = data.drop(columns=['Cabin'], axis=1)

sex_map = {

        'male':0,

    'female':1

}

data.loc[: ,'Sex'] = data['Sex'].map(sex_map)

data_test.loc[: , 'Sex'] = data_test['Sex'].map(sex_map)
data_test = data_test.drop(columns=['Name'],axis=1)



data_test = data_test.drop(columns=['Ticket'],axis=1)



data_test = data_test.drop(columns=['Cabin'], axis=1)
data.head()
# Adding dummified columns

data['Embarked:C'] = pd.get_dummies(data['Embarked'])['C']

data['Embarked:S'] = pd.get_dummies(data['Embarked'])['S']

# Same but to test data

data_test['Embarked:C'] = pd.get_dummies(data_test['Embarked'])['C']

data_test['Embarked:S'] = pd.get_dummies(data_test['Embarked'])['S']
data.head()
columns_to_drop = ['SibSp', 'Parch', 'Embarked']

data = data.drop(columns = columns_to_drop)

data_test = data_test.drop(columns = columns_to_drop)
data.head()
data_test.head()
from sklearn.model_selection import train_test_split

X = data.drop(columns = ['Survived'])

y = data['Survived']



# Train/Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
# Fitting Logistic Regression

log_reg_model = LogisticRegression()

log_reg_model.fit(X_train, y_train)


# Scoring

train_prediction = log_reg_model.predict(X_train)

test_prediction = log_reg_model.predict(X_test)

accuracy_train = accuracy_score(train_prediction, y_train)

accuracy_test = accuracy_score(test_prediction, y_test)



print(f"Score on training set: {accuracy_train}")

print(f"Score on test set: {accuracy_test}")
from sklearn.tree import DecisionTreeClassifier



# lists for scoring

training_scores = []

test_scores = []



# finding optimal depth

for i in range(1,20):

    decision_tree = DecisionTreeClassifier(max_depth=i)

    decision_tree.fit(X_train, y_train)

    training_scores.append(decision_tree.score(X_train,y_train)*100)

    test_scores.append(decision_tree.score(X_test,y_test)*100)

# Plotting training and test scores against max_depth = i

plt.figure()



plt.plot(training_scores, label = 'Training Scores')

plt.plot(test_scores, label = 'Test Scores')

plt.xticks(range(1,15,1))

plt.title('Relationship between training score and test score vs max_depth')

plt.xlabel('Maximum Depth')

plt.ylabel('Score')

plt.legend()



# Print best test score:

highest_score = max(test_scores)

highest_index = test_scores.index(highest_score)



print(f"Best Accuracy Score: Max Depth of {highest_index + 1}")

print(f"Training accuracy of {round(training_scores[highest_index], 2)}% (max_depth = {highest_index + 1})")

print(f"Test accuracy of {round(test_scores[highest_index],2)}% (max_depth = {highest_index + 1})")

    
decision_tree_3 = DecisionTreeClassifier(max_depth=3)

decision_tree_3.fit(X_train, y_train)
pred = decision_tree_3.predict(data_test)

submission_dict = {'PassengerId':data_test['PassengerId']  , "Survived": pred }

submission = pd.DataFrame(submission_dict)

submission

submission.to_csv("titanic_submission.csv", index = False)