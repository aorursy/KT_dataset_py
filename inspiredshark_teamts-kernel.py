# General

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
def display_all(df):

    with pd.option_context("display.max_rows", 100, "display.max_columns", 100): 

        display(df)
# read the train data set and set the index to the PassengerId

train = pd.read_csv('../input/application_train.csv', index_col = 'SK_ID_CURR')

app_train = train.copy()
# Check the head of the data set

app_train.head()
# Check the size of the data set

app_train.shape
# list all columns

app_train.columns.tolist
app_train.describe()
# Check the data types of the data set

app_train.dtypes

app_train.dtypes.value_counts()
# Check for missing values

train.isnull().sum()
#Check for unique features

app_train.nunique()
# Reading submission data set

submission = pd.read_csv('../input/sample_submission.csv')

display_all(submission)
# Check the Test data set

app_test = pd.read_csv('../input/application_test.csv', index_col = 'SK_ID_CURR')

display_all(app_test)
# Check the shape of the test data set

app_test.shape
# List all features of the test data set

app_test.columns.tolist
# Looking into the statistics of the data set

app_test.describe()
# Checking the data types of the data set features

app_test.dtypes

app_test.dtypes.value_counts()
# Checking for the missing values in the data set

app_test.isnull().sum()
# combine the train and test datab sets

app_train["dataset"] = "train"

app_test["dataset"] = "test"

combined_df = app_train.append(app_test, sort=True)

combined_df.shape
# Looking into the combined data sets

display_all(combined_df)
# Checking for outliers in the TARGET features

combined_df['TARGET'].value_counts()
# Visualization plot for the TARGET features

combined_df['TARGET'].astype(float).plot.hist()
# Checking the data types of the combined set

combined_df.dtypes.value_counts()
# Checking the number of unique values in each object data types

combined_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Encoding the object data types with 2 or less unique features 

for c in [c for c in combined_df.columns if c!= "dataset"]:

    

    if combined_df[c].dtypes == 'object':

        

        if len(list(combined_df[c].unique())) <= 2:

            combined_df[c] = combined_df[c].astype('category')

            combined_df[c] = combined_df[c].cat.codes   
# Check for the changes in the data types

combined_df.dtypes.value_counts()
combined_df.head()
# Splitting the combined_data into test and train data set

dftrain = combined_df[combined_df.dataset == 'train'].drop('dataset', axis = 1)

dftest = combined_df[combined_df.dataset == 'test'].drop(['dataset', 'TARGET'], axis = 1)

dftest.shape, dftrain.shape
# Encoding the features with more than two unique values

dftrain = pd.get_dummies(dftrain)

dftest = pd.get_dummies(dftest)

dftrain.head()
# Check for correlation between TARGET and other features with values > 0.01

#corr = train.corr()['TARGET'].sort_values()

corr = dftrain.corr().TARGET

listOne = corr[np.abs(corr)>10**(-2)].index

dftrain = dftrain[listOne]

corr
# New Shape of train data set

dftrain.shape
# List of features with missing values > 100,000

# listTwo = train[train.isnull().sum()>10**(5)].

listTwo = (dftrain.isnull().sum()[dftrain.isnull().sum()>100000]).index

listTwo
# Droping features with missing values from the train data set 

dftrain = dftrain.drop(listTwo, axis = 1)

dftrain.shape
# Adding List of features with correlation > 0.01 to test data set and dropping TARGET

listTest = listOne.drop("TARGET")  

dftest = dftest[listTest]

dftest.head()
# Droping features with missing values from the test data set 

dftest = dftest.drop(listTwo, axis = 1)

dftest.shape
# split the train data set into train and validation enable shuffling

X = dftrain.drop('TARGET', axis = 1)

y = dftrain['TARGET']

X_train, X_valid, y_train, y_valid = train_test_split(

         X, y, test_size=0.2, random_state=42, shuffle=False)
# get the shapes of the data sets 

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
# impute the missing numerical valuese with median of the feature

imp = SimpleImputer(strategy="mean")

imp.fit(X_train)

X_train = pd.DataFrame(imp.transform(X_train),columns = X_train.columns)

X_valid = pd.DataFrame(imp.transform(X_valid),columns = X_valid.columns)
# build the model, fit the train and get the score for train and 

# validation

m = RandomForestClassifier(n_estimators=11,

                          max_features=.4,

                          random_state=42,

                          min_samples_leaf=10,

                          bootstrap=False,

                          n_jobs=-1)

m.fit(X_train, y_train)

m.score(X_train, y_train),m.score(X_valid, y_valid)
# get the confusion matrix

y_pred = m.predict(X_valid)

confusion_matrix(y_valid,y_pred)
# check the precision_recall_fscore_support

precision_recall_fscore_support(y_valid,y_pred)
#Interpreting the predictions

feat_importance = pd.DataFrame({"features": X_train.columns, 

                              "importance": m.feature_importances_}).sort_values("importance", ascending=False)

feat_importance
# Plot a graph to show all features

plot = feat_importance.plot('features', 'importance', 'barh', figsize=(5,40), legend=False)

fig = plot.get_figure()
#List all the features

feat_importance['features'].tolist()
# get rid of some features

X_train2 = X_train.drop(['ORGANIZATION_TYPE_Construction',

 'FLAG_DOCUMENT_6',

 'OCCUPATION_TYPE_Security staff',

 'FLAG_EMP_PHONE',

 'OCCUPATION_TYPE_Managers',

 'OCCUPATION_TYPE_Cooking staff',

 'NAME_EDUCATION_TYPE_Lower secondary',

 'NAME_FAMILY_STATUS_Widow',

 'NAME_INCOME_TYPE_State servant',

 'NAME_HOUSING_TYPE_Rented apartment',

 'OCCUPATION_TYPE_Low-skill Laborers',

 'FONDKAPREMONT_MODE_reg oper spec account',

 'ORGANIZATION_TYPE_XNA',

 'OCCUPATION_TYPE_High skill tech staff',

 'ORGANIZATION_TYPE_Medicine',

 'ORGANIZATION_TYPE_Transport: type 3',

 'ORGANIZATION_TYPE_School',

 'OCCUPATION_TYPE_Accountants',

 'FONDKAPREMONT_MODE_org spec account',

 'ORGANIZATION_TYPE_Restaurant',

 'ORGANIZATION_TYPE_Military',

 'FLAG_DOCUMENT_16',

 'FLAG_DOCUMENT_13'],axis=1)





X_valid2 = X_valid.drop(['ORGANIZATION_TYPE_Construction',

 'FLAG_DOCUMENT_6',

 'OCCUPATION_TYPE_Security staff',

 'FLAG_EMP_PHONE',

 'OCCUPATION_TYPE_Managers',

 'OCCUPATION_TYPE_Cooking staff',

 'NAME_EDUCATION_TYPE_Lower secondary',

 'NAME_FAMILY_STATUS_Widow',

 'NAME_INCOME_TYPE_State servant',

 'NAME_HOUSING_TYPE_Rented apartment',

 'OCCUPATION_TYPE_Low-skill Laborers',

 'FONDKAPREMONT_MODE_reg oper spec account',

 'ORGANIZATION_TYPE_XNA',

 'OCCUPATION_TYPE_High skill tech staff',

 'ORGANIZATION_TYPE_Medicine',

 'ORGANIZATION_TYPE_Transport: type 3',

 'ORGANIZATION_TYPE_School',

 'OCCUPATION_TYPE_Accountants',

 'FONDKAPREMONT_MODE_org spec account',

 'ORGANIZATION_TYPE_Restaurant',

 'ORGANIZATION_TYPE_Military',

 'FLAG_DOCUMENT_16',

 'FLAG_DOCUMENT_13'],axis=1)
#Show the plot of features with high importance 

X_valid2 = feat_importance.plot('features', 'importance', 'barh', figsize=(5,30), legend=False)

X_train2 = feat_importance.plot('features', 'importance', 'barh', figsize=(5,30), legend=False)

fig = X_train2.get_figure()

fig = X_valid2.get_figure()


#Display shape of train and validation data

#There is a problem in this line of code, the goal is to see the list of features for X_train2 and X_valid2

# X_train2['features'].tolist(), X_valid2['features'].tolist()

# feat_importance['features'].tolist()
# impute the missing values in the test data set with the medians 

#from the train data set

test_imp = pd.DataFrame(imp.transform(dftest),columns=dftest.columns,index=dftest.index)
# predict for the test data set

test_pred = m.predict(test_imp)
# look into your test data set

test_imp.head()
# creat the submission data frame and look into it

submission = pd.DataFrame({"SK_ID_CURR": test_imp.index, "TARGET":test_pred})
# change the data type of the prediction from float to integer

#submission.TARGET = submission.TARGET.apply(lambda x : float(x))
# check the submission data set 

submission.head(10)

