### Necessary Imports

import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
samp = pd.read_csv('../input/predicting-bank-telemarketing/samp_submission.csv')

train = pd.read_csv('../input/predicting-bank-telemarketing/bank-train.csv')

test = pd.read_csv('../input/predicting-bank-telemarketing/bank-test.csv')

train.drop(columns = 'duration')

train.head()
test.head()
### This gives us the probability of each occurance

train['y'].value_counts(1)
### In the sample data (which only has the client id), we randomly assign success values based off the probabilities shown above

samp.Predicted = np.random.choice(range(2), size = samp.shape[0], p = [train['y'].value_counts(1)[0], train['y'].value_counts(1)[1]])
samp.to_csv('first_test.csv', index = False)
print(train.columns)

train.dropna()
train = pd.get_dummies(train)

test = pd.get_dummies(test)

train.head()
X = train.drop(columns = 'y')

X = X.drop(columns = 'id')

X = X.drop(columns = 'duration')

Y = train['y']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)



test = test.drop(columns = 'id')

test = test.drop(columns = 'duration')

test['default_yes'] = 0
## Fitting the tree to our testing data 

tree = DecisionTreeClassifier(max_depth = 5)

tree.fit(X_train,Y_train)
## Running the tree on our training and testing data

print("Training accuracy:", tree.score(X_train, Y_train))

print("Testing accuracy:", tree.score(X_test, Y_test))
pd.DataFrame({'Gain': tree.feature_importances_}, index = X_train.columns).sort_values('Gain', ascending = False)
## Running bagging classifier on our original decision tree

bag_model = BaggingClassifier(base_estimator=tree, n_estimators=100,bootstrap=True)

bag_model = bag_model.fit(X_train,Y_train)

y_pred = bag_model.predict(X_test)

print("Training accuracy: ", bag_model.score(X_train,Y_train))

print("Testing accuracy: ", bag_model.score(X_test,Y_test))



feature_importances = np.mean([

    tree.feature_importances_ for tree in bag_model.estimators_

], axis=0)



pd.DataFrame({'Gain': tree.feature_importances_}, index = X_train.columns).sort_values('Gain', ascending = False)
predictions = pd.DataFrame(bag_model.predict(test))

samp['Predicted'] = predictions

samp.to_csv('second_test.csv', index=False)