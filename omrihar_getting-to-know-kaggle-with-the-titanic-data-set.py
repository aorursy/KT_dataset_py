# Some important imports



import pandas as pd

import tensorflow as tf

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from IPython.display import display

import matplotlib.pyplot as plt

%matplotlib inline
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(5)
# Prepare the data for the logistic regression



data = train[['Age', 'SibSp', 'Parch', 'Fare']].copy()

data['Sex'] = train.Sex == 'female'

data = data.join(pd.get_dummies(train.Pclass, prefix='Pclass'))

data = data.join(pd.get_dummies(train.Embarked, prefix='Embarked'))



# Handle NaNs

#display(data.isnull().sum())

# There are 177 NaN values in the age columns (out of 890 rows)

# As a first attempt, we simply drop those rows

not_null_ind = data.index[~data.Age.isnull()]

data = data.ix[not_null_ind,:]



Y = train.Survived[not_null_ind]



# Split this data to train and test datasets so I can estimate how well I'm doing

X_train, X_test, Y_train, Y_test = train_test_split(data, Y)
# First attempt, feed the data to scikit-learn's LogisticRegression algorithm

# without tweaking any of the parameters

logistic = LogisticRegression().fit(X_train, Y_train)

display("Score of the 'naive' logistic regression: {:.3f}".format( logistic.score(X_test, Y_test)))



# To see which factors are most important here, we plot coefficients with the data columns

display(pd.DataFrame.from_dict(dict(Factors=X_train.columns, Coefficients=logistic.coef_[0])))



# The result of this is around 0.8, which if we compare with the leaderboard is not very good.

# The best people get is 100% on the test data set with many above 0.88
# To proceed now, I will plot some "learning curves" to see where I should spend my

# time in improving the learning algorithm

# Since scikit-learn doesn't give you access to the cost function directly, we have to use a circumspect route

# (or program it myself)

from sklearn.linear_model.logistic import _logistic_loss

cost = _logistic_loss(logistic.coef_, X_test, Y_test, 1, 1/logistic.C)

display(cost)
logistic.coef_