# importing packages required for the analysis

import pandas as pd
# get the data from train file

train = pd.read_csv("../input/train.csv")
# Cabin column - too many NULL values, don't think it can help to predict the Survival

train.drop(['Cabin'], 1, inplace=True) # No need to assign as inplace Flag is set
list(train) # Get the list of columns of dataframe
len(train) # Number of records in a dataframe
train = train.dropna() # Removing null values from data frame
len(train) # Number of records in a dataframe
y = train.Survived # target value - outcome to train for
y.head() # Get the few elements and column summarize info
y.to_csv("clean_training_outcome.csv", index=False)
X = train.drop(['Survived'], 1) # We do not need "Survived" columns anymore in our features
X.head() # top 5 elements of the dataframe
# We do not need PassengerId, Name, Ticket as they do not suggest anything

X.drop(['PassengerId', 'Name', 'Ticket'], 1, inplace=True) # 1 is used for the axis, in this case it is column
X.head()
# We will create dummies (hotcode patching) for non-numerical values

X = pd.get_dummies(X)
X.head() # you will find all different string values become ENUM here, and now we get boolean value if they are present
X.to_csv("clean_train.csv", index=False) # Storing into the file, so we can use it for some other model
# Preprocessing is done - It is time to create models
from sklearn import tree # We will use Decision Tree - a naive but powerful model
dtc = tree.DecisionTreeClassifier()

dtc.fit(X, y) # Here we are creating a model, where X - input features and Y - target outcome
# Now we will pre-process test data, to be used for the prediction

test = pd.read_csv("../input/test.csv")
# Submission file requires id and the predicted survival value, creating a dataframe for IDs.

ids = test[['PassengerId']]
ids.head()
# Dropping columns which are not required anymore in the training data, but before doing that, let us check the data

test.head()
# PassengerId, Name, Ticket are not Needed

# Cabin - does contain bunch of NaN

test.drop(['PassengerId', 'Name', 'Ticket'], 1, inplace=True)
test.drop(['Cabin'], 1, inplace=True)
test.head()
# Now there will be some NA values, we will replace that with an arbitratory number (2)

test.fillna(2, inplace=True)
test.head()
# Now we need to change non-numerical values into ENUM i.e. convert them into numerical values

test = pd.get_dummies(test)
test.head()
test.to_csv("clean_test.csv", index=False)
# A model will try to predict the test data.

predictions = dtc.predict(test)
predictions[:5]
len(predictions)
len(ids)
final_result = ids.assign(Survived=predictions) # predictions for Survival for different IDs
final_result.head()
len(final_result)
# Convert the outcome into CSV file

# We do not want index column, otherwise it will create a new column

final_result.to_csv("dt_results.csv", index=False) 
# Now we will do another prediction using logistic regressions

X = pd.read_csv("clean_train.csv")

y = pd.read_csv("clean_training_outcome.csv", header=None)
X.head()
y.head()
len(X), len(y)
# Next model - Logistic Regression

from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
logisticRegr.fit(X, y)
test = pd.read_csv("clean_test.csv")
test.head()
len(test)
predictions = logisticRegr.predict(test)
predictions[:5]
len(predictions)
len(ids)
final_result = ids.assign(Survived=predictions)
final_result.head()
len(final_result)
final_result.to_csv("logistic_regression_results.csv", index=False) 
list(X)