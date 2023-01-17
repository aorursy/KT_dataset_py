# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Training Data.csv')
df.shape
df.head()
# Drop patient_id column
df = df.drop(['patient_id'],axis=1)
# Convert Categorical Variables to numerical 
df['Gender'].replace('F' , 0 ,inplace=True)
df['Gender'].replace('M', 1 ,inplace=True)
df['Adherence'].replace('Yes', 1 ,inplace=True)
df['Adherence'].replace('No', 0 ,inplace=True)
df.head()
sns.countplot(df['Sms_Reminder'], color='red')
sns.set(rc={'figure.figsize':(15,12)})
sns.countplot(x="Age", hue="Adherence", data=df)
# heatmap to observe correlations between features and with the output variable 
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# Normalizing Age and Prescription_period colums to 0 and 1
df['Age'] = df['Age'] / 113
df['Prescription_period'] = df['Prescription_period'] / 120
df.head()
# Split the data into training and testing sets
y = df['Adherence']
X = df.drop('Adherence',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 165)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', X_test.shape)
print('Testing Features Shape:', y_train.shape)
print('Testing Labels Shape:', y_test.shape)
# Importing sklearn and xgboost libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import  RandomForestClassifier as RFC
from xgboost import XGBClassifier as XGBC
from sklearn.metrics import  accuracy_score
model_score = {}
# 1. decision tree
dtc = DTC(random_state=1)
dtc.fit(X_train, y_train)
prediction_dtc = dtc.predict(X_test)
model_score[dtc] = accuracy_score(y_test, prediction_dtc)
# 2. Random Forest
rfc = RFC(random_state=1 ,n_estimators=100)
rfc.fit(X_train, y_train)
prediction_rfc = rfc.predict(X_test)
model_score[rfc] = accuracy_score(y_test, prediction_rfc)
# 3. Logistic Regression
lgr = LogisticRegression(penalty='l1')
lgr.fit(X_train, y_train)
prediction_lgr = lgr.predict(X_test)
model_score[lgr] = accuracy_score(y_test, prediction_lgr)
model_score
# 4. XGBoost
params = {'eta': 0.5, 'max_depth':6, 'gamma': 0.1, 'subsample': 1, 'reg_alpha': 1, 
          'n_jobs': -1, 'random_state': 1, 'n_estimators': 1000}
xgbc = XGBC(**params)
xgbc.fit(X_train, y_train, early_stopping_rounds=3, eval_set=[[X_test, y_test]])
prediction_xgbc = xgbc.predict(X_test)
score = accuracy_score(y_test, prediction_xgbc)
score
# Confusion matrix for XGBoost
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, prediction_xgbc)
cm
# 5. ANN using keras
import keras
from keras.models import Sequential  # Required to initialize the neural network
from keras.layers import Dense  # Required to build the layers in the neural network
classifier = Sequential()
# Adding hidden layer
classifier.add(Dense(output_dim = 5 , init = 'uniform' , activation = 'relu' , input_dim = 9))
classifier.add(Dense(output_dim = 5 , init = 'uniform' , activation = 'relu')) # creating 2nd hidden layer.
classifier.add(Dense(output_dim = 1 , init = 'uniform' , activation = 'sigmoid')) # creating output layer
classifier.compile(optimizer='adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])
# fitting the model to ANN
classifier.fit(X_train, y_train, batch_size = 50 , nb_epoch = 100)
y_prob = classifier.predict(X_test)
y_pred = (y_prob > 0.5)
y_pred
# Confusion matrix for ANN
from sklearn.metrics import confusion_matrix
cm_1  = confusion_matrix(y_test, y_pred)
cm_1
# Predicting the results for test_data
df_test1 = pd.read_csv('../input/Test Data.csv')
df_test1.shape
# Similar modifications done to test_data
df_test = df_test1.drop(['patient_id'],axis=1)
df_test['Gender'].replace('F' , 0 ,inplace=True)
df_test['Gender'].replace('M', 1 ,inplace=True)
df_test['Age'] = df_test['Age'] / 113
df_test['Prescription_period'] = df_test['Prescription_period'] / 120
final_result_ann = classifier.predict(df_test) # prediction using ANN
final_result_xgboost = xgbc.predict_proba(df_test) # prediction using XGBoost
final_result_ann = final_result_ann.reshape(119788)
output_dataframe = pd.DataFrame({'patient_id': df_test1['patient_id'] ,'probability_score_knn': final_result_ann, 'probability_score_xgboost': final_result_xgboost[:,1] })
output_dataframe['probability_score'] = (output_dataframe['probability_score_knn'] + output_dataframe['probability_score_xgboost'])/2
output_dataframe.head()
output_dataframe['Adherence'] = (output_dataframe['probability_score'] > 0.5)
output_dataframe.drop('probability_score_knn', axis=1, inplace= True)
output_dataframe.drop('probability_score_xgboost', axis=1, inplace= True)
output_dataframe.head()
final_output = output_dataframe[['patient_id' , 'Adherence', 'probability_score']]
final_output['Adherence'].replace(True , 'Yes' ,inplace=True)
final_output['Adherence'].replace(False, 'No' ,inplace=True)
final_output.head()
final_output.to_csv('output.csv', index=False)