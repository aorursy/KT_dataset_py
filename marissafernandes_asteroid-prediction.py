import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import data to dataframe

data = pd.read_csv('/kaggle/input/asteroid-dataset/dataset.csv')
pd.set_option('display.max_columns', 500)

data.head()
data.columns
data.describe()
data.shape
#1. id and spkid

print(data['id'].nunique())

print(data['spkid'].nunique())

print(data['full_name'].nunique())

print(data['pdes'].nunique())
# Potentially hazardous asteroids

data['pha'].value_counts(normalize=True)
# Near Earth Object

data['neo'].value_counts(normalize=True)
# Asteroid orbit ID

print(data['orbit_id'].unique())

print(data['orbit_id'].nunique())
# Comet Designation prefix

print(data['prefix'].unique())

print(data['prefix'].nunique())
# Equinox reference

print(data['equinox'].unique())

print(data['equinox'].nunique())
# Orbit classification

print(data['class'].unique())

print(data['class'].nunique())
data1 = data.drop(['id', 'pdes', 'name', 'prefix', 'equinox'], axis='columns', inplace=False)
asteroid_df = data1[data1['pha'].notna()]

asteroid_df = asteroid_df.drop(['diameter', 'albedo', 'diameter_sigma'], axis= 'columns')
asteroid_df = asteroid_df[asteroid_df['H'].notna()]
asteroid_df = asteroid_df[asteroid_df['sigma_ad'].notna()]

asteroid_df = asteroid_df[asteroid_df['ma'].notna()] # Remove row with the one missing value for 'ma'
asteroid_df['neo'] = asteroid_df['neo'].astype('category')

asteroid_df['pha'] = asteroid_df['pha'].astype('category')

asteroid_df['class'] = asteroid_df['class'].astype('category')
# What percent of asteroids are near earth objects?



asteroid_df['neo'].value_counts(normalize=True)*100
# Of the near earth objects, what percent of them are potentially hazardous asteroids?



asteroid_df[asteroid_df['neo']=='Y']['pha'].value_counts(normalize=True)*100
# How many asteroids of the dataset are potentially hazardous asteroids?



asteroid_df['pha'].value_counts(normalize=True)*100
# Of the potentially hazardous asteroids, what percent of them are near earth objects?



asteroid_df[asteroid_df['pha']=='Y']['neo'].value_counts(normalize=True)*100
# What is the distribution of the orbit classification?



asteroid_df['class'].value_counts(normalize=True)*100
# How many orbit IDs exist?



asteroid_df['orbit_id'].nunique()
# Number of orbit_id that have less than 10 occurances

orbits = asteroid_df['orbit_id'].value_counts().loc[lambda x: x<10].index.to_list()
len(orbits)
asteroid_df.loc[asteroid_df['orbit_id'].isin(orbits), 'orbit_id'] = 'other'
# Reset the index

asteroid_df = asteroid_df.reset_index(drop=True)
# Create a subset of only numerical columns to scale

subset_df = asteroid_df[asteroid_df.columns[~asteroid_df.columns.isin(['spkid', 'full_name', 'neo', 'pha', 'orbit_id', 'class'])]]
from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()

scaled_df = scaler.fit_transform(subset_df)

scaled_df = pd.DataFrame(scaled_df, columns=subset_df.columns)

asteroid_df = pd.concat([asteroid_df[['spkid', 'full_name', 'neo', 'pha', 'orbit_id', 'class']],scaled_df], axis=1)

scaled_df.head()
# 1. Create one-hot encoding columns using get_dummies

asteroid_df1 = pd.get_dummies(asteroid_df, columns=['neo', 'class', 'orbit_id'])

asteroid_df1.head()
from sklearn.model_selection import train_test_split



X = asteroid_df1.drop(['spkid', 'full_name', 'pha'], axis=1)

y = asteroid_df1.iloc[:]['pha']



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1501)
print("Before OverSampling, counts of label 'N': {}".format(sum(y_train == 'N'))) 

print("Before OverSampling, counts of label 'Y': {} \n".format(sum(y_train == 'Y'))) 

  

# import SMOTE module from imblearn library 

from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 12) 

x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel()) 

  

print("After OverSampling, counts of label 'N': {}".format(sum(y_train_res == 'N'))) 

print("After OverSampling, counts of label 'Y': {}".format(sum(y_train_res == 'Y'))) 
def metricCalculation(y_test, pred):

    

    precision_metric = metrics.precision_score(y_test, pred, average = "macro")

    recall_metric = metrics.recall_score(y_test, pred, average = "macro")

    accuracy_metric = metrics.accuracy_score(y_test, pred)

    f1_metric = metrics.f1_score(y_test, pred, average = "macro")

    print('Precision metric:',round(precision_metric, 2))

    print('Recall Metric:',round(recall_metric, 2))

    print('Accuracy Metric:',round(accuracy_metric, 4))

    print('F1 score:',round(f1_metric, 2))
# Import the model

from sklearn.linear_model import LogisticRegression



# Instantiate the model

logisticRegr = LogisticRegression(max_iter= 10000) # create object for the class



# Fit to train model with features and labels

logisticRegr.fit(x_train_res, y_train_res)



# Predict for test set

lr_pred = logisticRegr.predict(x_test)
# Calculate metrics

metricCalculation(y_test, lr_pred)
# Print confusion matrix

print(metrics.confusion_matrix(y_test, lr_pred))
# Import the model

from sklearn.ensemble import RandomForestClassifier



# Instantiate model with 150 decision trees

rf = RandomForestClassifier(n_estimators = 150, random_state = 1551)



# Train the model on training data

rf.fit(x_train_res, y_train_res)



# Predict for test set

rf_pred = rf.predict(x_test)
# Calculate metrics

metricCalculation(y_test, rf_pred)
# Confusion matrix

print(metrics.confusion_matrix(y_test, rf_pred))
feature_imp = pd.DataFrame(rf.feature_importances_,index=x_train_res.columns, columns = ['Importance']).sort_values(by='Importance', ascending=False)
# Top 10 important variables

feature_imp[0:10]
# 10 least important features

feature_imp[-10:]
feature_imp[-50:].index
asteroid_df2 = pd.get_dummies(asteroid_df, columns=['neo', 'class'])

asteroid_df2.drop(['orbit_id','sigma_ma', 'sigma_tp'], axis='columns', inplace=True)
# Create train test splits 



X1 = asteroid_df2.drop(['spkid', 'full_name', 'pha'], axis=1)

y1 = asteroid_df2.iloc[:]['pha']



x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.30, random_state=1501)
# Create equal balance of classes using SMOTE



sm = SMOTE(random_state = 12) 

x_train_res1, y_train_res1 = sm.fit_sample(x_train1, y_train1.ravel()) 

  

print("After OverSampling, counts of label 'N': {}".format(sum(y_train_res1 == 'N'))) 

print("After OverSampling, counts of label 'Y': {}".format(sum(y_train_res1 == 'Y'))) 
# Instantiate model with 150 decision trees

rf = RandomForestClassifier(n_estimators = 150, random_state = 1551)



# Train the model on training data

rf.fit(x_train_res1, y_train_res1)



# Predict for test set

rf_pred1 = rf.predict(x_test1)
# Calculate metrics

metricCalculation(y_test, rf_pred1)
print(metrics.confusion_matrix(y_test, rf_pred))
# Duplicate the training data sets for the label

y_train_res_2 = y_train_res
# Encode labels



for n,i in enumerate(y_train_res_2):

    if i=='Y':

      y_train_res_2[n] = 1

    else:

        y_train_res_2[n] = 0
# Use label encoding to encode test labels 

y_test_2 = y_test.cat.codes
# Load the training dataset along with the label to LGBM

import lightgbm as lgb 



train_data=lgb.Dataset(x_train_res,label=y_train_res_2)
#setting parameters for lightgbm



param = {'num_leaves': 150, # number of leaves per tree

         'nrounds': 350,

         'max_depth': 25, # depth of tree

         'learning_rate': 0.01, # learning rate

         'max_bin': 500 # max number of bins to bucket the feature values.

        }
# Train the model 



lgbm = lgb.train(param, train_data)

lgbm_pred = lgbm.predict(x_test)



# Convert the predicted probabilities to 0 or 1

for i in range(0,len(y_test_2)):

    if lgbm_pred[i]>=.5:       # setting threshold to .5

       lgbm_pred[i]=1

    else:  

       lgbm_pred[i]=0
# Calculate metrics

metricCalculation(y_test_2, lgbm_pred)
# Confusion Matrix

print(metrics.confusion_matrix(y_test_2, lgbm_pred))
# Duplicate the training data sets for the label

y_train_res_3 = y_train_res1



# Encode labels



for n,i in enumerate(y_train_res_3):

    if i=='Y':

      y_train_res_3[n] = 1

    else:

        y_train_res_3[n] = 0

        

# Use label encoding to encode test labels 

y_test_3 = y_test1.cat.codes
# Load the training dataset along with the label to LGBM

import lightgbm as lgb 



train_data_1=lgb.Dataset(x_train_res1,label=y_train_res_3)
# Train the model 



lgbm_1 = lgb.train(param, train_data_1)

lgbm_pred_1 = lgbm_1.predict(x_test1)



# Convert the predicted probabilities to 0 or 1

for i in range(0,len(y_test_3)):

    if lgbm_pred_1[i]>=.5:       # setting threshold to .5

       lgbm_pred_1[i]=1

    else:  

       lgbm_pred_1[i]=0
# Calculate metrics

metricCalculation(y_test_3, lgbm_pred_1)
# Confusion Matrix

print(metrics.confusion_matrix(y_test_3, lgbm_pred_1))