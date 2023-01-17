# Import libraries necessary for this project.
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display for notebooks.
%matplotlib inline

# Load the Census dataset.
data_raw = pd.read_csv("../input/train.csv")

# dropping nan in entire database.
data = data_raw.dropna(axis=1, how='any')

# Success - Display the first record.
display(data.head(n=5))
# dropping some of the columns which are not useful in predictions.
data.drop(['Name', 'PassengerId','Ticket'], axis =1)

# labelling the data with survival column.
label_raw = data['Survived']
label = label_raw.apply(lambda x: 1 if x == 1 else 0)

# dropping some of the columns which are not required in features.
features_raw = data.drop(['Survived','Name', 'PassengerId','Ticket','Fare'], axis = 1)

# tranforming the data values of SibSp and Parch to some values which can be predicted accurately.
skewed = ['SibSp','Parch']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Now getting features and finalizing it for prediction.
features_final = pd.get_dummies(features_raw,columns=['Sex','Pclass',])
# Viewing the final features.
display(features_final.head(n=5))
# Import train_test_split.
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    label, 
                                                    test_size = 0.1, 
                                                    random_state = 0)

# Show the results of the split.
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
display(features_final.head())
# I am using svm for my model to get 100% accuracy.
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# classifying the data with some paramters.
clf = RandomForestClassifier(max_depth=4,n_estimators= 2000)

# fit or train the data.
clf.fit(X_train, y_train)
# Doing some of the same steps to clean my data with test.csv
# reading the csv file.
test_raw = pd.read_csv("../input/test.csv")

# dropping any nan if it is there.
test = test_raw.dropna(axis=1, how='any')

# getting features for my test which is need to be predicted.
features_test_raw = test.drop(['Name', 'PassengerId','Ticket','Embarked'], axis =1)

# tranforming the data values of SibSp and Parch to some values which can be predicted accurately.
skewed = ['SibSp','Parch']
features_log_test_transformed = pd.DataFrame(data = features_test_raw)
features_log_test_transformed[skewed] = features_test_raw[skewed].apply(lambda x: np.log(x + 1))

# transforming 'sex' and 'pclass to 0 : 1 format.
features_test_final = pd.get_dummies(features_test_raw,columns=['Sex','Pclass'])

# display some of the test features.
display(features_test_final.head())

# predict the test features =.
pred = clf.predict(features_test_final)
# calculating the accuracy of the data 
y_true_raw = pd.read_csv("../input/gender_submission.csv")

# importing accuracy_score
from sklearn.metrics import accuracy_score

# Cross finger and yes we did it
accuracy_score(y_true_raw['Survived'], pred)
# getting the output file
submission = pd.DataFrame({ "PassengerId": test_raw["PassengerId"], "Survived": pred })
submission.to_csv('out.csv', index = False)