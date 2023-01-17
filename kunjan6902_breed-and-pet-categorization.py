import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

train = pd.read_csv('/kaggle/input/hackerearth-adopt-pet-challenge/train.csv')

train.head()
# Checking Problem Category wise Data frequency 

print('Breed Category frequency :')

print(train['breed_category'].value_counts(normalize=True))

print(' ')

print('Breed Category count :')

print(train['breed_category'].value_counts())



print('========================================')



# Checking Problem Category wise Data frequency 

print('Pet Category frequency :')

print(train['pet_category'].value_counts(normalize=True))

print(' ')

print('Pet Category count :')

print(train['pet_category'].value_counts())
train.dtypes
train.isnull().sum()
#Column - condition



#Based on Dataset evaluation, all NaN values of "condition" column is for Breed_category "2". <br>

#So, We will impute condition as "3" (a specific number in short which is not already a category) for all NaN values. <br>



train['condition'] = train['condition'].fillna('3')

train['condition'] = train['condition'].astype(int)

train.isnull().sum()
#Column - condition



#Categories => One Hot Encoding



#Prefix parameter is used to add 'condition' word as prefix to the column names that are going to be generated

condition_ohe = pd.get_dummies(train['condition'], prefix='condition') 



#Appending generated One Hot Encoding to dataframe & Dropping original 'condition column' as it is redundant now

train = pd.concat([train, condition_ohe], axis=1)

train = train.drop('condition', axis = 1)



train.head()
#Column - color_type



#As 'color_type' column contains lots of unique categories, We have two approaches

#https://datascience.stackexchange.com/questions/48875/how-to-handle-columns-with-categorical-data-and-many-unique-values



#One Hot Encoding

#Word Embedding



print(train['color_type'].nunique())

print(train['color_type'].unique())
color_type_ohe = pd.get_dummies(train['color_type']) 



#Appending generated One Hot Encoding to dataframe & Dropping original 'color_type column' as it is redundant now

train = pd.concat([train, color_type_ohe], axis=1)

train = train.drop('color_type', axis = 1)



train.head()
#Column - issue_date + listing_date



#To Convert Datetime columns issue_date and listing_date, We will find Day wise difference between these two columns in 'time_difference_D' column

import datetime

import numpy as np

from datetime import timedelta



train['issue_date_2'] =  pd.to_datetime(train['issue_date'], format='%Y-%m-%d')

train['listing_date_2'] =  pd.to_datetime(train['listing_date'], format='%Y-%m-%d')



train['time_difference_D'] = (train['listing_date_2'] - train['issue_date_2']) / np.timedelta64(1, 'D')



Labled_Columns = ['pet_id', 'issue_date', 'listing_date', 'issue_date_2', 'listing_date_2']

train = train.drop(Labled_Columns, axis = 1)
train.head()
temp = train.copy()



Labled_Columns = ['breed_category', 'pet_category']

X = temp.drop(Labled_Columns, axis = 1)

print(len(X.columns))

print(X.columns)



y = temp[["breed_category", "pet_category"]]

print(len(y.columns))

print(y.columns)
from sklearn.model_selection import train_test_split



# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
# using classifier chains

from skmultilearn.problem_transform import ClassifierChain

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# initialize classifier chains multi-label classifier

classifier = ClassifierChain(LogisticRegression())



# Training logistic regression model on train data

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



# Transforming scipy.sparse.csc.csc_matrix to dataframe

y_pred = pd.DataFrame(predictions.toarray())
# accuracy

print("Accuracy = ", accuracy_score(y_true=y_test, y_pred=y_pred))

print("\n")
#Breaking y_train and y_test into two separate columns - Breed and Pet



#X_train and X_test will be used for predicting breed_category



y_train_breed = y_train["breed_category"]

y_train_pet = y_train["pet_category"]



y_test_breed = y_test["breed_category"]

y_test_pet = y_test["pet_category"]



#Following X_train_for_PetCategory & X_test_for_PetCategory will be used for predicting pet_category

X_train_for_PetCategory = pd.concat([X_train, y_train_breed], axis=1)

X_test_for_PetCategory = pd.concat([X_test, y_test_breed], axis=1)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train_breed)



predictions = logisticRegr.predict(X_test)

score = logisticRegr.score(X_test, y_test_breed)

print('logistic Regression score for Breed Category Prediction is : {:.2f} %'.format(score * 100))



logisticRegr.fit(X_train_for_PetCategory, y_train_pet)



predictions = logisticRegr.predict(X_test_for_PetCategory)

score = logisticRegr.score(X_test_for_PetCategory, y_test_pet)

print('logistic Regression score for Pet Category Prediction is : {:.2f} %'.format(score * 100))
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



#'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.

#'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

#'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
dtree_model = DecisionTreeClassifier().fit(X_train, y_train_breed)

dtree_pred_Breed = dtree_model.predict(X_test)



print('DecisionTreeClassifier + Breed')

print('Breed_precision : {:.2f} %'.format(precision_score(y_test_breed, dtree_pred_Breed, average="weighted") * 100))

print('Breed_recall : {:.2f} %'.format(recall_score(y_test_breed, dtree_pred_Breed, average="weighted") * 100))



print('=======================================')



dtree_model = DecisionTreeClassifier().fit(X_train_for_PetCategory, y_train_pet)

dtree_pred_Pet = dtree_model.predict(X_test_for_PetCategory)



print('DecisionTreeClassifier + Pet')

print('Pet_precision : {:.2f} %'.format(precision_score(y_test_pet, dtree_pred_Pet, average="weighted") * 100))

print('Pet_recall : {:.2f} %'.format(recall_score(y_test_pet, dtree_pred_Pet, average="weighted") * 100))
rf_clf_b = RandomForestClassifier(random_state=0, n_jobs=-1)   

rf_clf_b.fit(X_train, y_train_breed)

rf_pred_Breed = rf_clf_b.predict(X_test)



print('RandomForestClassifier + Breed')

print('Breed_precision : {:.2f} %'.format(precision_score(y_test_breed, rf_pred_Breed, average="weighted") * 100))

print('Breed_recall : {:.2f} %'.format(recall_score(y_test_breed, rf_pred_Breed, average="weighted") * 100))



print('=======================================')



rf_clf_p = RandomForestClassifier(random_state=0, n_jobs=-1)   

rf_clf_p.fit(X_train_for_PetCategory, y_train_pet)

rf_pred_Pet = rf_clf_p.predict(X_test_for_PetCategory)



print('RandomForestClassifier + Pet')

print('Pet_precision : {:.2f} %'.format(precision_score(y_test_pet, rf_pred_Pet, average="weighted") * 100))

print('Pet_recall : {:.2f} %'.format(recall_score(y_test_pet, rf_pred_Pet, average="weighted") * 100))
#XGB Classifier

xg_cl_b = xgb.XGBClassifier(objective= "multi:softprob", n_estimators=1000, learning_rate=0.05, seed=123)

xg_cl_b.fit(X_train, y_train_breed)

xg_pred_Breed = xg_cl_b.predict(X_test)



print('XGBClassifier + Breed')

print('Breed_precision : {:.2f} %'.format(precision_score(y_test_breed, xg_pred_Breed, average="weighted") * 100))

print('Breed_recall : {:.2f} %'.format(recall_score(y_test_breed, xg_pred_Breed, average="weighted") * 100))



print('=======================================')



xg_cl_p = xgb.XGBClassifier(objective= "multi:softprob", n_estimators=1000, learning_rate=0.05, seed=123)

xg_cl_p.fit(X_train_for_PetCategory, y_train_pet)

xg_pred_Pet = xg_cl_p.predict(X_test_for_PetCategory)



print('XGBClassifier + Pet')

print('Pet_precision : {:.2f} %'.format(precision_score(y_test_pet, xg_pred_Pet, average="weighted") * 100))

print('Pet_recall : {:.2f} %'.format(recall_score(y_test_pet, xg_pred_Pet, average="weighted") * 100))
import pandas as pd

test = pd.read_csv('/kaggle/input/hackerearth-adopt-pet-challenge/test.csv')



print(test.columns)



#Condition Column

test['condition'] = test['condition'].fillna('3')

test['condition'] = test['condition'].astype(int)



condition_ohe = pd.get_dummies(test['condition'], prefix='condition') 

test = pd.concat([test, condition_ohe], axis=1)

test = test.drop('condition', axis = 1)



#Color_type Column

color_type_ohe = pd.get_dummies(test['color_type']) 

test = pd.concat([test, color_type_ohe], axis=1)

test = test.drop('color_type', axis = 1)



#Issue_date and Listing_date Column

test['issue_date_2'] =  pd.to_datetime(test['issue_date'], format='%Y-%m-%d')

test['listing_date_2'] =  pd.to_datetime(test['listing_date'], format='%Y-%m-%d')

test['time_difference_D'] = (test['listing_date_2'] - test['issue_date_2']) / np.timedelta64(1, 'D')

Labled_Columns = ['pet_id', 'issue_date', 'listing_date', 'issue_date_2', 'listing_date_2']

test = test.drop(Labled_Columns, axis = 1)
#Expected Brown Tiger, Black Tiger in input data



#Test data does not contain these two values as color_type, and hence Model trained on Training_Data is not

#working on Test data



test.insert(14, 'Black Tiger', 0)

test.insert(27, 'Brown Tiger', 0)
#XGB Classifier

pred_Breed = xg_cl_b.predict(test)

#Appending Predicted Breed to X_test_for_PetCategory data for ClassifierChain

test['breed_category'] = pred_Breed



pred_Pet = xg_cl_p.predict(test)

test['pet_category'] = pred_Pet
test.to_csv("prediction_results.csv")