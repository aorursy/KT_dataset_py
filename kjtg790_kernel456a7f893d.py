#Check the python and seaborn version

from platform import python_version

import seaborn as sns



print(python_version())

print(sns.__version__)
# Import required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import os

import glob

import itertools

from sklearn.preprocessing import MinMaxScaler

from matplotlib import *

import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_mldata

from sklearn.feature_selection import RFE

from sklearn import metrics

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn.metrics import explained_variance_score

import statsmodels.api as sm

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,roc_auc_score ,roc_curve,auc

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split,StratifiedKFold

from scipy.stats import uniform, randint

import xgboost as xgb

from sklearn.pipeline import make_pipeline
# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
#Print the plot within the page

%matplotlib inline
# set seaborn theme if you prefer

sns.set_style('whitegrid')
# read data

df_titanic_train_data =  pd.read_csv("../input/titanic/train.csv")

#Look at the data frame to review top rows

df_titanic_train_data.head()
#Understand the rows and columns of the dataframe

df_titanic_train_data.shape
#Lets check the datatypes of the dataframe

df_titanic_train_data.info()
#Lookup up the summary

df_titanic_train_data.describe()
#Load the test data and check the top few rows

df_titanic_test_data = pd.read_csv("../input/titanic/test.csv")

df_titanic_test_data.head()
#Check for the rows and columns

df_titanic_test_data.shape
# Inspecting the percentages of Null values of the train dataframe

round(100*(df_titanic_train_data.isnull().sum()/len(df_titanic_train_data.index)), 2)
# Inspecting the percentages of Null values for the test dataframe

round(100*(df_titanic_test_data.isnull().sum()/len(df_titanic_test_data.index)), 2)
#Check for columns in the dataframe

df_titanic_train_data.columns
#Check for columns in the dataframe for the test dataframe

df_titanic_test_data.columns
#get the Survived value count

Survived = df_titanic_train_data["Survived"].value_counts()

#create dataframe for the Survived variable

df_Survived = pd.DataFrame({'Survived?': Survived.index,

                   'values': Survived.values

                  })

#plot the graph

plt.figure(figsize = (6,6))

plt.title('Survived Counts')

sns.set_color_codes('dark')

#Let draw using bar chart to compare the values

sns.barplot(x = 'Survived?', y='values', data=df_Survived)

locs, labels = plt.xticks()

plt.show()
#Let us impute the missing value for the Age variable with the mean value

df_titanic_train_data['Age'].fillna((round(df_titanic_train_data['Age'].mean(),0)),inplace=True)

df_titanic_train_data.head()
#Perform the same imputation for the test dataframe

df_titanic_test_data['Age'].fillna((round(df_titanic_test_data['Age'].mean(),0)), inplace=True)

df_titanic_test_data.head()
#remove the variable which has more than 70% missing values

pct_null = df_titanic_train_data.isnull().sum() / len(df_titanic_train_data)

missing_features = pct_null[pct_null > 0.70].index

df_titanic_train_data.drop(missing_features, axis=1, inplace=True)
#remove the variable which has more than 70% missing values

pct_null = df_titanic_test_data.isnull().sum() / len(df_titanic_test_data)

missing_features = pct_null[pct_null > 0.70].index

df_titanic_test_data.drop(missing_features, axis=1, inplace=True)
# Inspecting the percentages of Null values for the training dataframe

round(100*(df_titanic_train_data.isnull().sum()/len(df_titanic_train_data.index)), 2)
# Inspecting the percentages of Null values for the test dataframe

round(100*(df_titanic_test_data.isnull().sum()/len(df_titanic_test_data.index)), 2)
#Impute the missing Fare value with the mean

df_titanic_train_data['Fare'].fillna((round(df_titanic_train_data['Fare'].mean(),0)),inplace=True)

df_titanic_train_data.head()
#Impute the missing Fare value with the mean for the test dataframe

df_titanic_test_data['Fare'].fillna((round(df_titanic_test_data['Fare'].mean(),0)),inplace=True)

df_titanic_test_data.head()
# Inspecting the percentages of Null values

round(100*(df_titanic_train_data.isnull().sum()/len(df_titanic_train_data.index)), 2)
# Inspecting the percentages of Null values for the test data

round(100*(df_titanic_test_data.isnull().sum()/len(df_titanic_test_data.index)), 2)
#Check the unique values of the column

df_titanic_train_data['Survived'].unique()
#Check the unique values of the column

df_titanic_train_data['Pclass'].unique()
#Check the unique values of the column. We need to convert this into numberic values

df_titanic_train_data['Sex'].unique()
#Check the unique values of the column

df_titanic_train_data['Age'].unique()
#Check the unique values of the column

df_titanic_train_data['SibSp'].unique()
#Plot the Survived vs Pclass variables

sns.countplot(x='Survived',data=df_titanic_train_data,hue='SibSp',palette='RdBu_r')
#Check the unique values of the column

df_titanic_train_data['Parch'].unique()
#Plot the Survived vs Pclass variables

sns.countplot(x='Survived',data=df_titanic_train_data,hue='Parch',palette='RdBu_r')
#Check the unique values of the column

df_titanic_train_data['Ticket'].unique()
#Check the unique values of the column

df_titanic_train_data['Fare'].unique()


#Check the unique values of the column. We need to convert this into dummies (0s and 1s)

df_titanic_train_data['Embarked'].unique()
#plot the Survived vs Embarked

sns.countplot(x='Survived',data=df_titanic_train_data,hue='Embarked',palette='RdBu_r')
#Plot the Survived vs Pclass variables

sns.countplot(x='Survived',data=df_titanic_train_data,hue='Pclass',palette='RdBu_r')
#Lets check Survived vs Sex

sns.countplot(x='Survived',data=df_titanic_train_data,hue='Sex',palette='RdBu_r')
#Lets check the Age distribution

sns.distplot(df_titanic_train_data['Age'],bins=30,kde=False)
# Extract age and Survived columns in to new data frame

df_age_with_target = df_titanic_train_data.loc[:, ['Survived', 'Age']]



#Set the Age range

df_age_with_target['Age'] = pd.cut(df_age_with_target['Age'], bins=np.linspace(0, 70, num=11))

#Top 15 rows

df_age_with_target.head(15)
#Lets check the age distribution against Survived

plt.figure(figsize=(14,7))

sns.countplot('Age', data=df_age_with_target, hue='Survived')
#Let us find out the data types of the columns from the leads data frame

df_titanic_train_data.dtypes.value_counts()
df_titanic_train_data.shape
#Copy original dataframe for later use

df_titanic_train_data_original = df_titanic_train_data.copy()
#Check the column types again

df_titanic_train_data.info()
#Remove colums that are not required for prediction

remove_cols = ['Name','Ticket']

df_titanic_train_data = df_titanic_train_data.drop(remove_cols, axis=1)

#Check the column types again

df_titanic_train_data.info()
# Defining the map function

def binary_map(x):

    return x.map({"male": 1, "female": 0})
#Modify the column which has only two unique values

var_list = ['Sex']

df_titanic_train_data[var_list] = df_titanic_train_data[var_list].apply(binary_map)

# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df_titanic_train_data[['Embarked']],drop_first=True)



# Adding the results to the main dataframe

df_titanic_train_data = pd.concat([df_titanic_train_data, dummy1], axis=1)
#Checking the datatypes count

df_titanic_train_data.dtypes.value_counts()
#Dropping original variables as they are transformed

df_titanic_train_data = df_titanic_train_data.drop(['Embarked'], axis = 1)
#Now checking the data types again. Now all of the columns are having types that are helpful for the prediction (numberic)

df_titanic_train_data.dtypes.value_counts()
#Confirm that we do not have any object or non-numberic type

df_titanic_train_data.info()
#Create X and y

from sklearn.model_selection import train_test_split

X = df_titanic_train_data.drop(['PassengerId','Survived'], axis=1)

y = df_titanic_train_data['Survived']
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')

rfecv.fit(X, y)
# Create standardizer

standardizer = StandardScaler()
# Create logistic regression

logit = LogisticRegression()
# Pipeline to standardize and execute logistic regression

pipeline = make_pipeline(standardizer, logit)
#Let us now create k-Fold cross-validation

kf = KFold(n_splits=10, shuffle=True, random_state=1)
#Perform k-fold cross-validation

cv_results = cross_val_score(pipeline, # Pipeline created above

                             X, # Feature matrix for the features

                             y, # Target variable

                             cv=kf, # Using the Cross-validation technique, perform the validation

                             scoring="accuracy", # Loss function to find accuracy

                             n_jobs=-1)
#Mean value of the results

cv_results.mean()
#Let us check KNN classifier algorithm and it's performance

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X, y)

y_pred = knn.predict(X)

metrics.accuracy_score(y, y_pred)
# Let us split dataset of 25 observations into 5 folds

kf = KFold(25, shuffle=False)
#Perform KNN cross validation

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=5)
# Use cross_val_score function

# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the dat

# cv=10 for 10 folds

# scoring='accuracy' for evaluation metric - althought they are many

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

print(scores)
# use average accuracy as an estimate of out-of-sample accuracy

# numpy array has a method mean()

print(scores.mean())
# search for an optimal value of K for KNN



# range of k we want to try

k_range = range(1, 31)

# empty list to store scores

k_scores = []



# 1. we will loop through reasonable values of k

for k in k_range:

    # 2. run KNeighborsClassifier with k neighbours

    knn = KNeighborsClassifier(n_neighbors=k)

    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours

    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

    # 4. append mean of scores for k neighbors to k_scores list

    k_scores.append(scores.mean())





print(k_scores)
# in essence, this is basically running the k-fold cross-validation method 30 times because we want to run through K values from 1 to 30

# we should have 30 scores here

print('Length of list', len(k_scores))

print('Max of list', max(k_scores))
# plot how accuracy changes as we vary k



# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

# plt.plot(x_axis, y_axis)

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-validated accuracy')
# 10-fold cross-validation with the best KNN model

knn = KNeighborsClassifier(n_neighbors=20)
# Instead of saving 10 scores in object named score and calculating mean

# We're just calculating the mean directly on the results

print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')

rfecv.fit(X, y)



print("Optimal number of features: %d" % rfecv.n_features_)

print('Selected features: %s' % list(X.columns[rfecv.support_]))



# Plot number of features VS. cross-validation scores

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
# Splitting the data into train and validation set

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
#Checking the X dataframe

X.head(10)
#Using the standard scaler to standarize the variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['Pclass','Age','SibSp','Parch']] = scaler.fit_transform(X_train[['Pclass','Age','SibSp','Parch']])



X_train.head()
y_train.head()
#Check the correlation

plt.figure(figsize = (25,15))        # Size of the figure

sns.heatmap(X_train.corr(),annot = True)

plt.show()
corr_matrix = X_train.corr().abs()



#check correlation between independent variables

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

print(sol)
# Using the Logistic regression model to fit and find the summary of the model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
# Let us now use the RFE

logreg = LogisticRegression()

rfe = RFE(logreg, 20)

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
#Perform the logistic regression again with fit and summarize the model

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
#Dropping variables from the model

col1 = col.drop(['Parch','Fare'],1)
#New list of variables for re-fitting the model

col1
#Re-fit the model using the new variable set

X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# VIF is important to find the independent variables and it's impact to the model 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# We can now check the VIF using a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let us now check the predicted values of the training set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
#Array conversion with reshaping

y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
#Make the y_train data into dataframe with converted and probability of conversion

y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_prob':y_train_pred})

y_train_pred_final['PassengerId'] = y_train.index

y_train_pred_final.head()
#Let us use the standard cut-off of 0.5 to the logistic model

y_train_pred_final['predicted'] = y_train_pred_final.Survived_prob.map(lambda x: 1 if x > 0.5 else 0)



#Check the dataframe with predicted variable against prospects

y_train_pred_final.head()
# Define the Confusion matrix  and print the same

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )

print(confusion)
# Based on the predictions, we will check the accuracy of the model and print the same

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))
# VIF is important to find the independent variables and it's impact to the model 

from statsmodels.stats.outliers_influence import variance_inflation_factor
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
#Let's see the sensitivity of our logistic regression model

#0.5

TP / float(TP+FN)
# Let us calculate specificity

#0.5

TN / float(TN+FP)
# Let us find false postive rate - predicting Converted when customer does not have Converted

print(FP/ float(TN+FP))
# Let's print the positive predicting value 

print (TP / float(TP+FP))
# Let us now find the Negative predicting value

print (TN / float(TN+ FN))
#Define the ROC curve function

def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
#define metrics for the roc curve

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Survived_prob, drop_intermediate = False )
#Plot the ROC Curve

draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survived_prob)
# We will now Check probablity using new columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Survived_prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Using the accuracy, sensitivity and specificity , we will find various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# We will now plot a graph based on accuracy,sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
#From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

#We will set the cut-off to 0.3 so that we will have high sensitivity to get more leads to be identified for sales people



y_train_pred_final['final_predicted'] = y_train_pred_final.Survived_prob.map( lambda x: 1 if x > 0.4 else 0)



y_train_pred_final.head()
#Let us find the Converted counts from the training set

y_train_pred_final['Survived'].value_counts()
#Let us now find the predicted variable which can tell us how close it is to the Converted variable

y_train_pred_final['final_predicted'].value_counts()
#Here is the precision calculation

TP / TP + FP



confusion[1,1]/(confusion[0,1]+confusion[1,1])
# We can now check the accuracy of the final model

metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_predicted)



confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_predicted )

confusion2



TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
#Here is the sensitivity of the model

#0.4

TP / float(TP+FN)
#Here is Specificity of the model

#0.4

TN / float(TN+FP)
# Let us now calculate false postive rate - predicting Converted when prospect does not have converted

print(FP/ float(TN+FP))
# Here is the positive predictive value 

print (TP / float(TP+FP))
# Here is the negative predictive value

print (TN / float(TN+ FN))
#Let us now check the confusion matrix based on the above



confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )

confusion
#Recall calculation based on the above calculations

TP / TP + FN



confusion[1,1]/(confusion[1,0]+confusion[1,1])
from sklearn.metrics import precision_score, recall_score
#Let us find the precision score

precision_score(y_train_pred_final.Survived , y_train_pred_final.predicted)
#Let us now calculate recall score

recall_score(y_train_pred_final.Survived, y_train_pred_final.predicted)
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Survived, y_train_pred_final.predicted
#p,r and thresholds

p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survived_prob)
#Threshold plot

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
#Let us check the columns

X_validation.columns
#scale and transform the variables

X_validation[['Pclass','Age','SibSp']] = scaler.fit_transform(X_validation[['Pclass','Age','SibSp']])



X_validation.head()
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')

rfecv.fit(X, y)



print("Optimal number of features: %d" % rfecv.n_features_)

print('Selected features: %s' % list(X.columns[rfecv.support_]))



# Plot number of features VS. cross-validation scores

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
#Based on final variables, we will be performing prediction on the test set

X_validation = X_validation[col1]

X_validation.head()
#Add constant using test dataset

X_validation_sm = sm.add_constant(X_validation)
#Predict the Conversion based on the model built earlier

y_validation_pred = res.predict(X_validation_sm)
y_validation_pred[:10]
#Let us create a dataframe using the test pred

y_pred_1 = pd.DataFrame(y_validation_pred)
#Here is the created dataframe

y_pred_1.head()
#Making the dataframe using the test set

y_validation_df = pd.DataFrame(y_validation)
#Adding Prospect ID to the dataframe

y_validation_df['PassengerId '] = y_validation_df.index
#Resetting the index

y_pred_1.reset_index(drop=True, inplace=True)

y_validation_df.reset_index(drop=True, inplace=True)
#Now concat the dataframes

y_pred_final = pd.concat([y_validation_df, y_pred_1],axis=1)
y_pred_final.head()
#Let us now rename the column name

y_pred_final= y_pred_final.rename(columns={ 0 : 'Survived_prob'})
y_pred_final.head()
#Let us now try with the cut-off of 0.3 as we did for training data

y_pred_final['final_predicted'] = y_pred_final.Survived_prob.map(lambda x: 1 if x > 0.4 else 0)
#Examine the dataframe

y_pred_final.head()
#Find the confusion matrix

confusion2 = metrics.confusion_matrix(y_pred_final.Survived.round(), y_pred_final.final_predicted.round() )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
#Sensitivity of the model we built

#0.4 Validation

TP / float(TP+FN)
#Specificity of the model we built

#0.4 Validation

TN / float(TN+FP)
from sklearn.metrics import precision_score, recall_score
#Finally we will check the precision of the model

precision_score(y_pred_final.Survived , y_pred_final.final_predicted)
df_titanic_test_data.head()
X_test = df_titanic_test_data.drop(['PassengerId','Name','Parch','Ticket','Fare'], axis=1)
# Defining the map function

def binary_map(x):

    return x.map({"male": 1, "female": 0})
var_list = ['Sex']

X_test[var_list] = X_test[var_list].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(X_test[['Embarked']],drop_first=True)
# Adding the results to the Leads dataframe

X_test = pd.concat([X_test, dummy1], axis=1)
#Dropping original variables as they are transformed

X_test = X_test.drop(['Embarked'], axis = 1)
#Using the standard scaler to standarize the variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_test[['Pclass','Age','SibSp','Sex']] = scaler.fit_transform(X_test[['Pclass','Age','SibSp','Sex']])
X_test.head()
#Add constant using test dataset

X_test_sm = sm.add_constant(X_test)
#Predict the Conversion based on the model built earlier

y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
#Let us create a dataframe using the test pred

y_pred_1 = pd.DataFrame(y_test_pred)

#Here is the created dataframe

y_pred_1.head()
#Adding Prospect ID to the dataframe

y_pred_1['PassengerId'] = df_titanic_test_data['PassengerId']

y_pred_1['Name'] = df_titanic_test_data['Name']

y_pred_1['Sex'] = df_titanic_test_data['Sex']
y_pred_1.reset_index(drop=True, inplace=True)
y_pred_final= y_pred_1.rename(columns={ 0 : 'Survived_prob'})
y_pred_final['Predicted'] = y_pred_final.Survived_prob.map(lambda x: 1 if x > 0.4 else 0)
y_pred_final.head()
df_final_data = y_pred_final.copy()

to_cols = ['PassengerId','Predicted']

df_final_data = df_final_data[to_cols]

df_final_data.rename(columns={'Predicted': 'Survived'}, inplace=True)

df_final_data.head()
#write the result to the CSV file

df_final_data.to_csv('submission.csv',index=False)