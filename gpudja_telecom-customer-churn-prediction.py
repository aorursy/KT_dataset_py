import pandas as pd

import numpy as np



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn import metrics

from sklearn.metrics import classification_report
# importing datasets

churn = pd.read_csv('../input/logisticregression-telecomcustomer-churmprediction/churn_data.csv')

customer = pd.read_csv('../input/logisticregression-telecomcustomer-churmprediction/customer_data.csv')

internet = pd.read_csv('../input/logisticregression-telecomcustomer-churmprediction/internet_data.csv')



# merging churn and customer dataframe on customerID

df_1 = pd.merge(churn, customer, how='inner', on='customerID')



# merging df_1 and internet dataframe on customerID

data = pd.merge(df_1,internet, how='inner', on = 'customerID')
data.head()
data.describe()
data.info()
# Checking for null values

data.isnull().sum()
# TotalCharges is an object and not float!!!

# We don´t have null values but from error we can see that column 'TotalCharges' contains whitespace = ' '



# data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
# How many whitespace = ' ' we have in column 'TotalCharges'

data['TotalCharges'].str.isspace().value_counts()
data['TotalCharges'].isnull().sum()
# Replacing whitespace to NAN values and converting to numeric data (float)

data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
# How many NAN values is in column

data['TotalCharges'].isnull().sum()
# Replacing NAN values with mean value from all data in column 'TotalCharges'



#new_value = data['TotalCharges'].astype('float').mean(axis=0)



new_value = (data['TotalCharges']/data['MonthlyCharges']).mean()*data['MonthlyCharges']

data['TotalCharges'].replace(np.nan, new_value, inplace=True)
# How many NAN values is in column 'TotalCharges' after replacing NAN with mean 

data['TotalCharges'].isnull().sum()
# Checking for null values

data.isnull().sum()
sns.pairplot(data=data)
sns.countplot(x = 'Contract', data=data)
plt.figure(figsize=(10,6))

sns.countplot(x = 'PaymentMethod', data=data)
sns.countplot(x = 'Churn', data=data)
sns.countplot(x = 'gender', data=data)
pd.set_option('display.max_columns', 500)

data.head()
# Making list for columns for One Hot Encoding

lista = ['PhoneService','PaperlessBilling', 'Churn', 'Partner', 'Dependents']



# With .map method and lambda function turning Yes/No into 1/0

data[lista] = data[lista].apply(lambda x:x.map({'Yes': 1, "No": 0}))

data.head()
# checking other data npr. 'StreamingMovies'

data['StreamingMovies'].value_counts()
# Making dummy variables for categorical data with more inputs



data_dummy = pd.get_dummies(data[['Contract', 'PaymentMethod', 'gender', 'MultipleLines', 'InternetService', 

                                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                                    'TechSupport', 'StreamingTV', 'StreamingMovies']], drop_first=True)

data_dummy.head()
# Merging original data frame with 'dummy' dataframe

data = pd.concat([data,data_dummy], axis=1)

data.head()
data.columns
# Dropping attributes for which we made dummy variables



data = data.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 

                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                        'TechSupport', 'StreamingTV', 'StreamingMovies'], axis=1)
# setting Independent variable (X) and Dependent variable (y)

X = data.drop(['Churn','customerID'], axis=1)

y = data['Churn']



# spliting data into train and test samples

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
# since data are within long-range (0 - 8684) it is necessary to perform data standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# just checking X_train before standardization

X_train.head()
# standardization on X_train

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()
# Ordinary Least Squares: sm.OLS(y, X)

mod1 = sm.OLS(y_train,X_train,data=data)
results1 = mod1.fit()

print(results1.summary())
# LogisticRegression object as lr

lr = LogisticRegression()
#  RFE - Feature ranking with recursive feature elimination.

rfe = RFE(estimator=lr, n_features_to_select=20, step=1)    

rfe = rfe.fit(X_train, y_train)
# print summaries for the selection of attributes

print(rfe.support_)

print(rfe.ranking_)
# making list and dataframe to see what attributes was selected

list_for_df = list(zip(X_train.columns, rfe.support_, rfe.ranking_))

df = pd.DataFrame(list_for_df, columns = ['X_train.columns', 'rfe.support_', 'rfe.ranking_'])

df.head()
# the list of attributes that are selected

sel_att = X_train.columns[rfe.support_]

sel_att
#Adding a constant

X_train_const = sm.add_constant(X_train[sel_att])
# Ordinary Least Squares: sm.OLS(y, X)

mod2 = sm.OLS(y_train,X_train_const,data=data)

results2 = mod2.fit()

print(results2.summary())
# Getting the predicted values on the train set

y_predicted_train = results2.predict(X_train_const)

y_predicted_train.head()
# making dataframe for train values and predicted values with 'customerID'as index

final_y_predicted_df = pd.DataFrame(index= y_train.index, columns=('Churn','Churn_Predicted_Initial'))

final_y_predicted_df = pd.DataFrame({'Churn':y_train.values, 'Churn_Predicted_Initial':y_predicted_train})

final_y_predicted_df.index.name = 'customerID'

final_y_predicted_df.head()
# TRAIN DATA & PREDICTED ON TRAIN DATA

#Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

final_y_predicted_df['Churn_Predicted_Final'] = final_y_predicted_df.Churn_Predicted_Initial.map(lambda x: 1 if x > 0.5 else 0)

final_y_predicted_df.head()
# Confusion matrix for train data

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(final_y_predicted_df['Churn'], final_y_predicted_df['Churn_Predicted_Final'])

print(confusion_matrix)
# Accuracy_score for train data

from sklearn.metrics import classification_report

print(classification_report(final_y_predicted_df['Churn'], final_y_predicted_df['Churn_Predicted_Final']))
# Overall accuracy.

metrics.accuracy_score(final_y_predicted_df['Churn'], final_y_predicted_df['Churn_Predicted_Final'])
# standardization on X_test

X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_test[['tenure','MonthlyCharges','TotalCharges']])



#Adding a constant

X_test_const = sm.add_constant(X_test[sel_att])
# Getting the predicted values on the test set

y_predicted_test = results2.predict(X_test_const)

y_predicted_test.head()
# making dataframe for test values and predicted values with 'customerID'as index

final_y_predicted_train = pd.DataFrame(index= y_test.index, columns=('Churn','Churn_Predicted_Initial'))

final_y_predicted_train = pd.DataFrame({'Churn':y_test.values, 'Churn_Predicted_Initial':y_predicted_test})

final_y_predicted_train.index.name = 'customerID'

final_y_predicted_train.head()
# TEST DATA & PREDICTED ON TEST DATA

#Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

final_y_predicted_train['Churn_Predicted_Final'] = final_y_predicted_train.Churn_Predicted_Initial.map(lambda x: 1 if x > 0.5 else 0)

final_y_predicted_train.head()
# Confusion matrix for test data

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(final_y_predicted_train['Churn'], final_y_predicted_train['Churn_Predicted_Final'])

print(confusion_matrix)
# Accuracy_score for train data

from sklearn.metrics import classification_report

print(classification_report(final_y_predicted_train['Churn'], final_y_predicted_train['Churn_Predicted_Final']))
# Overall accuracy.

metrics.accuracy_score(final_y_predicted_train['Churn'], final_y_predicted_train['Churn_Predicted_Final'])
lr.fit(X_train,y_train)
predictions_lr = lr.predict(X_test)
print(classification_report(y_test,predictions_lr))