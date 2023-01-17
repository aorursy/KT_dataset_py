# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_values = pd.read_csv("/kaggle/input/train_values.csv")

train_labels = pd.read_csv("/kaggle/input/train_labels.csv")

test_values = pd.read_csv("/kaggle/input/test_values.csv")

submission_format = pd.read_csv("/kaggle/input/submission_format.csv")

# Inspecting data

train_labels.head()
test_values.head()
# Save the patient_id in a separate dataframe

train_patient_id = train_labels['patient_id']

test_patient_id = test_values['patient_id']



# Delete the patient_id columns in our data sets because its not really useful for regression/machine learning

train_labels.drop(columns='patient_id',inplace=True)

train_values.drop(columns='patient_id',inplace=True)

test_values.drop(columns='patient_id',inplace=True)
# Combine the train_values and train_labels(contains results)

train = pd.concat([train_values,train_labels],axis=1)



# Append the test_values set nderneath the train sets

df = train.append(test_values,ignore_index=True)



df.info()
df['thal'] = df['thal'].astype('category')
df['thal'].value_counts()
# Use describe just to inspect the values from each numeric columns. So I know whether column is boolean or not, whats the mean and std

df.describe()
binary=[]; binary_num=[]

categ=[];categ_num=[]

cont=[];cont_num=[]

for col in df.columns:

    if df[col].nunique() == 2:

        binary.append(col)

        binary_num.append(df[col].nunique())

    elif (df[col].nunique() > 2) & (df[col].nunique()<10):

        categ.append(col)

        categ_num.append(df[col].nunique())

    else:

        cont.append(col)

        cont_num.append(df[col].nunique())



listBinary = pd.Series(dict(zip(binary,binary_num)))

print('BINARY FEATURES(num unique):\n',listBinary)

listCateg = pd.Series(dict(zip(categ,categ_num)))

print('\n\nMULTI CATEGORY FEATURES(num unique):\n',listCateg) # \n is to skip 1 line

listCont = pd.Series(dict(zip(cont,cont_num)))

print('\n\nCONTINOUS FEATURES(num unique):\n',listCont)
# Plot distribution of Age

plt.figure(figsize=(15,8))

plt.subplot(2,1,1)

sns.countplot(listCont.index[0],data=df)

plt.subplot(2,1,2)

sns.countplot(listCont.index[0],data=df, hue='heart_disease_present')

plt.show()
# Plot distribution of max_heart_rate_achieved

plt.figure(figsize=(15,8))

plt.subplot(2,1,1)

plt.hist(df['max_heart_rate_achieved'],bins=20)

# sns.countplot(listCont.index[1],data=df)

plt.subplot(2,1,2)

sns.countplot(listCont.index[1],data=df, hue='heart_disease_present')

plt.show()



# # Supplementary.. just looking if theres correlation between age & max_heart_rate with hue on heart_disease

# sns.scatterplot(df['max_heart_rate_achieved'],df['age'],hue =df['heart_disease_present'])
# Check Skewness 

from scipy.stats import skew

print(skew(df['max_heart_rate_achieved']))



# Sweet, its about -0.52 which lies between -1 & +1. So no need to transform
# Plot distribution of oldpeak_eq_st_depression

plt.figure(figsize=(15,8))

plt.subplot(2,1,1)

# plt.hist(df['oldpeak_eq_st_depression'],bins=20)

sns.countplot(listCont.index[2],data=df)

plt.subplot(2,1,2)

sns.countplot(listCont.index[2],data=df, hue='heart_disease_present')

plt.show()
bins = [-1, 0, 1, 2, 3, 4, 5, 10]

names = [0, 1, 2, 3, 4, 5, 6]



df['OldPeakRange'] = pd.cut(df['oldpeak_eq_st_depression'], bins, labels=names)



# print(df.OldPeakRange.value_counts()) # Double check this by looking at the value from the first subplot above
# Plot distribution of oldpeak_eq_st_depression

plt.figure(figsize=(15,8))

plt.subplot(2,1,1)

# plt.hist(df['oldpeak_eq_st_depression'],bins=20)

sns.countplot('OldPeakRange',data=df,hue='heart_disease_present')
# Plot distribution of resting_blood_pressure

plt.figure(figsize=(15,8))

plt.subplot(2,1,1)

sns.countplot(listCont.index[3],data=df)

plt.subplot(2,1,2)

sns.countplot(listCont.index[3],data=df, hue='heart_disease_present')

plt.show()
# Plot distribution of serum_cholesterol_mg_per_dl

plt.figure(figsize=(15,8))

plt.subplot(2,1,1)

sns.countplot(listCont.index[4],data=df)

plt.subplot(2,1,2)

sns.countplot(listCont.index[4],data=df,hue='heart_disease_present')

plt.show()
bins = [0, 200, 239, 270, 1000]

names = ['normal','borderline','high','veryhigh']



df['SerumCholestrolRange'] = pd.cut(df['serum_cholesterol_mg_per_dl'], bins, labels=names)



print(df.SerumCholestrolRange.value_counts()) # Double check this by looking at the value from the first subplot above
# Plot distribution of serum_cholesterol_mg_per_dl

plt.figure(figsize=(15,4))

sns.countplot('SerumCholestrolRange',data=df,hue='heart_disease_present')
cat_cols = ['OldPeakRange', 'SerumCholestrolRange']



for col in cat_cols:

    df[col]=df[col].astype('category')

    

df.info()
df.drop(columns = ['oldpeak_eq_st_depression', 'serum_cholesterol_mg_per_dl'],inplace=True)
df.info()
for col in listCateg.index:

    df[col]=df[col].astype('category')

print(df.info())
uniqueVal = []

for col in df.columns:

    if df[col].dtype not in ('int64','float64'):

        print(col,':',df[col].nunique())

        uniqueVal.append(df[col].nunique())



print('Total Unique values from categorical data: ',sum(uniqueVal))
# Uising pd.get_dummies to one-hot encode the categorical data

new_df = pd.get_dummies(df, drop_first=True)



new_df.info()
# Make a countplot with hue on the heart disease for each variables

plt.figure(figsize=(15,12))

plt.subplot(3,1,1)

sns.countplot('exercise_induced_angina',data=new_df,hue='heart_disease_present')

plt.subplot(3,1,2)

sns.countplot('fasting_blood_sugar_gt_120_mg_per_dl',data=new_df,hue='heart_disease_present')

plt.subplot(3,1,3)

sns.countplot('sex',data=new_df,hue='heart_disease_present')



plt.show()
# Store heart_disease_present in y

y = new_df[['heart_disease_present']]

y = y.loc[:179,:]

y.shape
# Make a new copy of new_df and remove heart_disease_present



readySet = new_df.copy()



readySet.drop(columns = 'heart_disease_present',inplace=True)



readySet.info()
# Make a list of int64 data columns

listIntCol=[]

for col in readySet.columns:

    if readySet[col].dtype not in ['uint8']:

        listIntCol.append(col)

     

        

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler



# Make a Column transformer where I only want to Standard Scale int64 columns

ct = ColumnTransformer([

        ('somename', StandardScaler(), listIntCol)

    ], remainder='passthrough')



readySet_trans = ct.fit_transform(readySet)
trainSet = readySet_trans[:180] # Include index 180

submissionSet = readySet_trans[180:] # Does not include index 180

print(trainSet.shape)

print(submissionSet.shape)



# # If you want to understand about numpy slicing or get confused about it, play around with this example

# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# h = x[:3]

# print(h)

# j = x[3:]

# print(j)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(trainSet,y,stratify=y, test_size=0.3, random_state=123)
# Check the percentage of each distinct values in y_train & y_test if they are the same



print(pd.value_counts(y_train.values.flatten())/len(y_train))

print(pd.value_counts(y_test.values.flatten())/len(y_test))



# yup same
# Import necesary packages

from xgboost import XGBClassifier 

from sklearn.metrics import recall_score, log_loss





xgb = XGBClassifier(seed=123)



xgb.fit(X_train,y_train)



preds = xgb.predict(X_test)



print('Score : {:.4f}'.format(xgb.score(X_test,y_test)))



print('Log Loss : {:.2f}'.format(log_loss(y_test,preds)))
from sklearn.linear_model import LogisticRegression

import warnings

from sklearn.exceptions import DataConversionWarning



warnings.filterwarnings(action='ignore', category=DataConversionWarning)



lr = LogisticRegression(solver='liblinear')



lr.fit(X_train,y_train)



preds_lr = lr.predict(X_test)



print('Score : {:.4f}'.format(lr.score(X_test,y_test)))



print('Log Loss : {:.2f}'.format(log_loss(y_test,preds_lr)))



lr.get_params()
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import log_loss



xgb_param_grid = {

    'learning_rate': np.arange(0.01, 0.2, 0.01),

    'max_depth': np.arange(3,10,1),

    'n_estimators': np.arange(50, 200, 50),

    'colsample_bytree':np.arange(0.5,1,0.1),

    'min_child_weight': np.arange(0,2,0.5)

}



randomized_xgb = RandomizedSearchCV(estimator=xgb,

                        param_distributions = xgb_param_grid,

                        cv=10,n_iter=40, scoring="recall",verbose=1,random_state=12) # why 12? Aaron Rodgers



# Fit the estimator

randomized_xgb.fit(X_train,y_train)



preds = randomized_xgb.predict(X_test)



# Compute metrics

print('Best score: ',randomized_xgb.best_score_)

print('\nBest estimator: ',randomized_xgb.best_estimator_)

print('\n Log Loss : {:.4f}'.format(log_loss(y_test,preds)))

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss



# Ignore Warnings

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)



lr_param_grid = {

    'penalty': ['l1','l2'],

    'C': np.arange(0.01,1.1,0.1)

}



grid_lr = GridSearchCV(estimator=lr,

                        param_grid = lr_param_grid,

                        cv=5, scoring="recall",verbose=1) # why 12? Aaron Rodgers



# Fit the estimator

grid_lr.fit(X_train,y_train)



preds_rand_lr = grid_lr.predict(X_test)



# Compute metrics

print('Best score: ',grid_lr.best_score_)

print('\nBest estimator: ',grid_lr.best_estimator_)

print('\n Log Loss : {:.4f}'.format(log_loss(y_test,preds_rand_lr)))
predictions = grid_lr.predict(submissionSet)



# recall I have submission_format file



test = pd.read_csv("/kaggle/input/test_values.csv")

PatientId = test['patient_id']



submission = pd.DataFrame({ 'patient_id': PatientId,

                            'heart_disease_present': predictions })

submission.to_csv(path_or_buf ="HeartDisease_Submission.csv", index=False)

print("Submission file is formed")