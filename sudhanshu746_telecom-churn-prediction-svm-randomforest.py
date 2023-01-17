# We are predicting percentage of subscribers to a service who discontinue their subscriptions to the service within a given time period

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.ticker as mtic
import matplotlib.pyplot as plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
telecomDf = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
telecomDf.head()
# Lets examine variables for feature selection
telecomDf.columns.values
# Checking the data types of all the columns
telecomDf.dtypes
# Now lets explore if is there any missing or null values 
telecomDf.TotalCharges = pd.to_numeric(telecomDf.TotalCharges, errors='coerce')
telecomDf.isna().any() # All False confirm there is no missing values

# Preprocessing
telecomDf.isnull().sum()
# There are 11 missing value for Total Charges, lets remove these 11 values having missing data from dataset
# Remove NA values 
telecomDf.dropna(inplace = True)
# Lets remove customerId from dataset, which is not required for model
telecomDf4dummy = telecomDf.iloc[:,1:]
# Converting Label variable i'e Churn to binary Numerical  
telecomDf4dummy['Churn'].replace(to_replace='No',value=0,inplace=True)
telecomDf4dummy['Churn'].replace(to_replace='Yes',value=1,inplace=True)

# Convert categorical variable into dummy/indicator variables
# pd.get_dummies creates a new dataframe which consists of zeros and ones.
dummiesDf = pd.get_dummies(telecomDf4dummy)
dummiesDf.head(20)
# Feature Selection 

# Now Lets check correlation of Churn with other variables
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
dummiesDf.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
'''
Now, we have below predictor/independent variables
Contact Month-To-Month
Tenure
Total Charges
Online Security
Tech Support_No
Internet_service_FiberOptics

plt.figure(figsize=(20,10))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
plt.figure(figsize=(15,10))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
'''


#  Conclusion:  As per correlation, Month to month contracts, absence of online security and tech support seem to be positively correlated with churn.
#  While, tenure, two year contracts and Internet Service seem to be negatively correlated with churn.
# services such as Online security, streaming TV, online backup, tech support, Device protection, Partner and Streaming movies without internet connection seem to be negatively related to churn.

Y = dummiesDf['Churn'].values
#Accuracy 79.95
X = dummiesDf.drop(columns = ['Churn'])
# Accuracy 78.31%
#selected_features = ['Contract_Month-to-month','tenure','TotalCharges']
#Accuracy 79.31%
selected_features =['Contract_Month-to-month','tenure','TotalCharges','OnlineSecurity_No','TechSupport_No','InternetService_Fiber optic','PaymentMethod_Electronic check','MonthlyCharges','Contract_Two year','InternetService_DSL']
#Accuracy 76.46%
#selected_features=['Contract_Month-to-month','OnlineSecurity_No','TechSupport_No','tenure','Contract_Two year']
#Accuracy 79.53%
#selected_features=X.drop(columns=['PhoneService_Yes','gender_Female','gender_Male','PhoneService_No']).columns.values
X_select = X[selected_features]
# Lets scale all the variables from a range of 0 to 1
# Transforms features by scaling each feature to a given range.
#This estimator scales and translates each feature individually such that it is in the given range on the training set (0,1).
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Selected features
scaler.fit(X_select)
X_select = pd.DataFrame(scaler.transform(X_select))
X_select.columns=selected_features

X_select.head(20)
'''
1. Let's use Random forest classifier to approach Telecom churn data
Why Random forest ?
Random forest classifier is trademark term for an ensemble of decision tree
ensemble models combines several decision trees to produce better predictive performance than utilizing a single decision tree.
train_test_split: Split arrays/matrices into random train and test subsets, we are taking 20% data  as test. Random_states is seed value used by the random number generator
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=99)
randomForestModel = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,random_state =50, max_features = "auto",max_leaf_nodes = 30)
randomForestModel.fit(x_train,y_train)
testPrediction =  randomForestModel.predict(x_test)
print(metrics.accuracy_score(y_test,testPrediction))
importances = randomForestModel.feature_importances_
weights = pd.Series(importances,index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')
'''
2.1 Let's use Random forest classifier to approach Telecom churn data on selected features
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(X_select, Y, test_size=0.2, random_state=99)
randomForestModel = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,random_state =50, max_features = "auto",max_leaf_nodes = 30)
randomForestModel.fit(x_train,y_train)
testPrediction =  randomForestModel.predict(x_test)
print(metrics.accuracy_score(y_test,testPrediction))
# Confusion Matrix Validation 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,testPrediction))  
# 3. Lets check performance with SVM ( Support Vecor Machine) Model
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=99)
from sklearn.svm import SVC

modelSVM = SVC(kernel='linear') 
modelSVM.fit(X_train,y_train)
preds = modelSVM.predict(X_test)
metrics.accuracy_score(y_test, preds)


# Create the Confusion matrix for SVM
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,preds))  