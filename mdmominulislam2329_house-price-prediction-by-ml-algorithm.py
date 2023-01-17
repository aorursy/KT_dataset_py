# Start Python Imports

# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
#Select the loaction of party

#Load the train and test data set 

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# example of what a submission should look like

df_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv') 
#view the training data

df_train.head()
#only to see columns

df_train.columns
#analysing the dependent variable like sale price

df_train['SalePrice'].describe()

#descriptive statistical summary
#After having a summary, lets look at some figure to have clear visualization

#histogram plot using seaborn

sns.distplot(df_train['SalePrice']);
# View the test data (same columns as the training data)

df_test.head() # head = view first 5 lines
# View the example submisison dataframe to have an idea that what to submit: 

df_submission.head()
#analysing the train data

df_train.describe()

#descriptive statistical summary
#What missing values are there

#Now We will deal about missing data can imply a reduction of the sample size. 

#This can prevent us from proceeding with the analysis. Moreover, from a substantive perspective, 

#we need to ensure that the missing data process is not biased and hidding an inconvenient truth.

#Where are the holes in our data

# Plot graphic of missing values

missingno.matrix(df_train, figsize = (30,10))
# Alternatively, you can see the number of missing values like this

df_train.isnull().sum()
df_bin = pd.DataFrame() # for discretised continuous variables

df_con = pd.DataFrame() # for continuous variables
# Different data types in the dataset

df_train.dtypes
#We'll go through each column iteratively and see which 

#ones to use in our first models. Some may need more preprocessing than others to get ready.

#view the training data

df_train.head()
#we want our machine learning model to predict Sale Price based off all the others.

fig = plt.figure(figsize=(20,10))

sns.countplot(y='SalePrice', data=df_train);

print(df_train.SalePrice.value_counts());
df_train['SalePrice'].value_counts().head(30).plot(kind='barh',figsize=(20,10));
# Let's add this to our subset dataframes

df_bin['SalePrice'] = df_train['SalePrice']

df_con['SalePrice'] = df_train['SalePrice']

df_bin.head()
df_con.head()
from matplotlib import rcParams



# figure size in inches

#to make the chart bigger

rcParams['figure.figsize'] = 11.7,8.27

sns.distplot(df_train.LotArea);

#Here the values are numerical
# How many missing variables does Pclass have?

df_train.LotArea.isnull().sum()
df_bin.head()
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

df_train["MSZoning_coded"] = lb_make.fit_transform(df_train["MSZoning"])

df_train[["MSZoning", "MSZoning_coded"]].head(15)
# Let's add this to our subset dataframes

df_bin['MSZoning_coded'] = df_train['MSZoning_coded']

df_con['MSZoning_coded'] = df_train['MSZoning_coded']
df_bin.head()
# How many missing variables does YrSold?

df_train.YrSold.isnull().sum()
# Let's add this to our subset dataframes

df_bin['YrSold'] = df_train['YrSold']

df_con['YrSold'] = df_train['YrSold']
# How many missing variables does YrSold?

df_train.MSSubClass.isnull().sum()
# Let's add this to our subset dataframes

df_bin['MSSubClass'] = df_train['MSSubClass']

df_con['MSSubClass'] = df_train['MSSubClass']
# One-hot encode binned variables

one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('SalePrice')

df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)



df_bin_enc.head()
df_con.head(10)
# Seclect the dataframe we want to use first for predictions

selected_df = df_con
# Split the dataframe into data and labels

X_train = selected_df.drop('SalePrice', axis=1) # data

y_train = selected_df.SalePrice # labels
# Shape of the data (without labels)

X_train.shape
X_train.head()
# Shape of the labels

y_train.shape
# Function that runs the requested algorithm and returns the accuracy metrics

def fit_ml_algo(algo, X_train, y_train, cv):

    

    # One Pass

    model = algo.fit(X_train, y_train)

    acc = round(model.score(X_train, y_train) * 100, 2)

    

    # Cross Validation 

    train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

    # Cross-validation accuracy metric

    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    

    return train_pred, acc, acc_cv
# Logistic Regression

start_time = time.time()

train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 

                                                               X_train, 

                                                               y_train, 

                                                                    10)

log_time = (time.time() - start_time)

print("Accuracy: %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)

print("Running Time: %s" % datetime.timedelta(seconds=log_time))
# k-Nearest Neighbours

start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 

                                                  X_train, 

                                                  y_train, 

                                                  10)

knn_time = (time.time() - start_time)

print("Accuracy: %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)

print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
# Gaussian Naive Bayes

start_time = time.time()

train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 

                                                                      X_train, 

                                                                      y_train, 

                                                                           10)

gaussian_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gaussian)

print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)

print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes'],

    'Score': [

        acc_knn, 

        acc_log,  

        acc_gaussian, 

    ]})

print("---Reuglar Accuracy Scores---")

models.sort_values(by='Score', ascending=True)


# We need our test dataframe to look like this one

X_train.head()
# Our test dataframe has some columns our model hasn't been trained on

df_test.head()
# Create a list of columns to be used for the predictions

wanted_test_columns = X_train.columns

wanted_test_columns
# Make a prediction using the ML algorithm on the wanted columns

predictions = train_pred_gaussian.predict(df_test[wanted_test_columns])
# Create a submisison dataframe and append the relevant columns

submission = pd.DataFrame()

submission['Id'] = test['Id']

submission['SalePrice'] = predictions # our model predictions on the test dataset

submission.head()