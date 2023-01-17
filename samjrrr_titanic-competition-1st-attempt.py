#Import modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #Used to check project directory
import matplotlib.pyplot as plt #Used for data analysis, plots removed in current version
import seaborn as sns #Used for data analysis, plots removed in current version
from sklearn.model_selection import train_test_split #Used to format training data
from sklearn.ensemble import RandomForestClassifier #Model used for prediction
from sklearn.metrics import make_scorer, accuracy_score #Functions for determining model accuracy
from sklearn.model_selection import GridSearchCV #Funcitoin for optimising model parameters

#Print contents of project directory
print(os.listdir("../input"))
#Read training and test data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#Check contents of data
train.head()
#Describe data
print(train.describe())
#Set feature and target arrays for training data
X = train.drop(["Survived"], axis = 1)
y = train.Survived

#Set ID for later competion entry
ID = test.PassengerId
#Missing Values

print("\n Initial missing train Values")
print(X.isnull().sum())

print("\n Initial missing test Values")
print(test.isnull().sum())

#For now will use mean value for age. Next version will look at how age can be predicted from other parameters
#Only 3 total values missing for Fare and Embarked so will fill with most common value
#Large amount of data missing for cabin, will make assumption that these were people who did not have a cabin and make new catogory X to describe.
#Cabin next steps: Check relationship between fare and Cabin, if assumption is true would assume fare of people in cabin X is low. 
#Cabin next steps: Use other parameters to build model. Look into cabin number 

#Function to fill missing values in data
def missing_values_1(df):
    #Use mean average for age
    df["Age"] = df["Age"].fillna(X["Age"].mean())
    #Fill fare and embarked with mode value
    for col in ["Fare", "Embarked"]:
        df[col] = df[col].fillna(X[col].mode()[0])        
    #Fill cabin with X to represent no cabin
    df.Cabin = df.Cabin.fillna("X")
    df["Cabin_letter"] = df["Cabin"].str[0]
    return df

#Function to drop unwanted columns
def drop_data_1(df, columns_to_drop):
    for col in columns_to_drop:
        del df[col]
    return df

#Function to make sure catogorical features are saved as strings and then encode
#Encoder in function doesn't seem to work so have added this function later
def encode_data_1(df, cat_feature_list):
    #Ensure all values are saved as strings
    for col in cat_feature_list:
        df[col] = df[col].astype(str)
    df_encode = pd.get_dummies(df)
    return df_encode

#Function to transform data which combines the missing value, drop, and encode functions
def data_transform_1(df, columns_to_drop, cat_feature_list):
    missing_values_1(df)
    drop_data_1(df, columns_to_drop)
    encode_data_1(df, cat_feature_list)
    #df_encode = pd.get_dummies(df)
    return df

#NAME - Next: Use to extract title as new feature. Future: Certain names may provide information on class
#Ticket - Next: explore variable to see if there apears to be any useful information
#Cabin - Cabin floor has already been extracted. Next: look into cabin number
#PassengerId - ID is identical to index so uneeded for predictions
columns_to_drop = ["Name", "Ticket", "Cabin", "PassengerId"]
cat_feature_list = ["Pclass", "Sex", "Embarked", "Cabin_letter"]

data_transform_1(X, columns_to_drop, cat_feature_list)
data_transform_1(test, columns_to_drop, cat_feature_list)
X = pd.get_dummies(X)
test = pd.get_dummies(test)

#Quick fix as test has no cabin letter T
test["Cabin_letter_T"] = 0

print("\n Missing train Values post transform")
print(X.isnull().sum())

print("\n Missing test Values post transform")
print(test.isnull().sum())
#Format training data
#Will fit model using train_X/y and test with test_X/y, all taken from train dataset 

train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 0, test_size = 0.2)
#Modeling

#Random Forest Classifier used as model
clf = RandomForestClassifier()

#Fit data using defualy parameters
clf.fit(train_X, train_y)
pred_X = clf.predict(test_X)

print("Accuracy of non-optimised model: {:.2f}%".format(accuracy_score(test_y, pred_X)))
#Optimise randomforrest
#Vary n_estimators and min samples split
#Next: Determine which args of the clf are most import. Further optimsiation 

clf = RandomForestClassifier()

#Parameters to vary in model
parameters = {'n_estimators': range(5, 205, 10), 'min_samples_split': [2, 3, 5]}

#Create scorer used to evaluate performance in parameter tuning
acc_scorer = make_scorer(accuracy_score)

# Optimisation by grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(train_X, train_y)

# Set the clf to the optimised parameters
clf = grid_obj.best_estimator_

# Fit data using optimised algorithm 
clf.fit(train_X, train_y)

#Predict using test set from training data
pred_X = clf.predict(test_X)

print("Accuracy of optimised model: {:.2f}%".format(accuracy_score(test_y, pred_X)))
#Competition entry

#Refit model using all training data for improved accuracy
clf = RandomForestClassifier()
parameters = {'n_estimators': range(5, 205, 10), 'min_samples_split': [2, 3, 5]}
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(train_X, train_y)
clf = grid_obj.best_estimator_
clf.fit(X, y)

#Use new model to predict survivial rate for test data
pred_y = clf.predict(test)

#Produce survivial data for competition entry 
submission = pd.DataFrame({'PassengerId': ID, 'Survived': pred_y})
submission.to_csv('submission.csv', index=False)

print("Finished!")