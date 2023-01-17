import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
# Lets see the row and column count 

train_df.shape
train_df["Sex"] = train_df["Sex"].apply(lambda Sex: 0 if Sex == 'male' else 1)
# Now lets check. now we can numerical data

train_df.head()
y = targets = labels = train_df["Survived"].values  # output 

columns = ["Fare", "Pclass", "Sex", "Age", "SibSp"] # Input features
features = train_df[list(columns)].values
features
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # Lets fill the missing values using imputer
X = imp.fit_transform(features)
X
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one = my_tree_one.fit(X, y)
#The feature_importances_ attribute make it simple to interpret the significance of the predictors you include
print(my_tree_one.feature_importances_) 
print(my_tree_one.score(X, y))
test_df.head()
# Even here we convert categorical to numerical values
test_df["Sex"] = test_df["Sex"].apply(lambda Sex: 0 if Sex == 'male' else 1)

features_test = test_df[list(columns)].values
imp_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_test = imp_test.fit_transform(features_test)
X_test
pred = my_tree_one.predict(X_test) # Lets predict the output using the input features
#Print Confusion matrix 
pred = my_tree_one.predict(X)
df_confusion = metrics.confusion_matrix(y, pred)
df_confusion
# Lets see the accuracy of our model.
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, pred)
print(accuracy)
#Setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(X, y)

#Print the score of the new decison tree
print(my_tree_two.score(X, y))
pred = my_tree_two.predict(X)
df_confusion = metrics.confusion_matrix(y, pred)
df_confusion
# Lets see the accuracy of our model.
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, pred)
print(accuracy)
# Now finally lets apply predict on the test data
test_predictions = my_tree_two.predict(X_test)
print(test_predictions)
# We will create new dataframe where we will have just the passenger id and the survivor predictions that we made.
test_ids = test_df["PassengerId"]
submission_df = {"PassengerId": test_ids,
                 "Survived": test_predictions}
submission = pd.DataFrame(submission_df)