# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics



# import the train CSV into notebook



dataset = pd.read_csv('../input/train.csv')

print(dataset.shape)

dataset.head()

print(dataset.groupby('Survived').count())
dataset['Family_size'] = dataset['SibSp'].values + dataset['Parch'].values

dataset.tail()
# Some of the columns not much value added in this i.e., Name, Fare, Ticket etc., so we will remove them off the dataset



dataset = dataset.drop(['Name', 'Ticket', 'Fare', 'Embarked', 'Cabin'], axis = 1)

print(dataset.shape)

dataset.head()
# move the Survived Column to the end



dataset = dataset[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_size', 'Survived']]

dataset.head()
# Check for any missing values from the dataset



dataset.isna().any()

dataset['Age'].isna().value_counts()
dataset.Age = dataset['Age'].fillna(method = 'ffill')
# convert the categorical features to numeric using Labelencoder



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



dataset['Sex'] = le.fit_transform(dataset['Sex'])

dataset.head()

dataset['Age'].value_counts()
a = map(lambda x : x//15, dataset['Age'])

a = list(a)

print(len(a))
dataset['Age'] = pd.Series(a)

dataset['Age']
# Split the data as X, Y 



X = dataset.iloc[:, 0:7]

Y = dataset['Survived']



print(X.shape, Y.shape)

X.head()
from sklearn.feature_selection import chi2



chi, p = chi2(X, Y)

chi
# now split the data into train and CV 



from sklearn.model_selection import train_test_split



x_train, x_cv, y_train, y_cv = train_test_split(X, Y, train_size = 0.8)

print(x_train.shape, x_cv.shape)     # Size of Train and CV datasets

print(y_train.shape, y_cv.shape) 
# Now use the Decission Tree to make the prediction



dtree = DecisionTreeClassifier(criterion = 'gini',random_state = 42, min_samples_split = 2)

dtree
from sklearn.model_selection import GridSearchCV



depth = {'max_depth': [3, 4, 5, 6, 7]}



grid = GridSearchCV(estimator = dtree, param_grid = depth, scoring = 'accuracy', cv = 5, error_score = np.NaN)

grid
# train the model with the CV for finding best fit hyperparameters



grid.fit(x_train, y_train)



print(grid.best_score_)

print(grid.best_params_)
# predict the CV error



dt = DecisionTreeClassifier(max_depth = grid.best_params_['max_depth'], random_state = 42, min_samples_split = 2)



dt.fit(x_train, y_train)



pred_cv = dt.predict(x_cv)



accuracy = metrics.accuracy_score(y_cv, pred_cv)

print('The CV datset prediction accuracy is', accuracy)

print(dt.feature_importances_)
dict(zip(x_train.columns, dt.feature_importances_))
# Testing it in test dataset



test_df = pd.read_csv('../input/test.csv')

print(test_df.shape)

test_df.head()

# Applying those preprocessing steps on the test data



test_df['Family_size'] = test_df['SibSp'].values + test_df['Parch'].values



test_df = test_df.drop(['Name', 'Ticket', 'Fare', 'Embarked', 'Cabin'], axis = 1)
test_df.isna().any()
test_df.Age = test_df['Age'].fillna(method = 'ffill')

test_df.head()
test_df['Sex'] = le.fit_transform(test_df['Sex'])



b = map(lambda x : x//15, test_df['Age'])

b = pd.Series(a)

test_df['Age'] = b



test_df.head()
print(test_df.shape)
# Now our test dataset is ready and we can predict its Survival status from the above model



pred_test = dt.predict(test_df)

pred_test.shape

print(type(pred_test))
c = dict(zip(test_df['PassengerId'], pred_test))

c.items()
gender_submission1 = pd.DataFrame(data = list(c.items()), columns = ['PassengerId', 'Survived'])

gender_submission1.head()
print(os.listdir("../input"))
gender_submission1.to_csv('submission.csv',encoding = 'utf-8')