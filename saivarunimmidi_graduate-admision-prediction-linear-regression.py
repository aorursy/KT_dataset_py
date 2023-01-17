# importing necessary libraries for avoiding warnings

import warnings

warnings.filterwarnings('ignore')
# import necessary libraries for importing and understanding the data

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# importing the necessary libraries for model building

import sklearn

import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Reading the data into dataframes

admission_dataframe =  pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
# analyzing the data

admission_dataframe.head()
# Checking the shape of the dataframe

admission_dataframe.shape
# checking the info about the columns of the dataframe

admission_dataframe.info()
# analyzing the summary statistics of the numerical columns of the dataframe

admission_dataframe[['GRE Score', 'TOEFL Score','CGPA', 'SOP', 'LOR ', 'University Rating' ]].describe()
# checking the columns present in the dataframe

admission_dataframe.columns
# analyzing the array of the values of the data

admission_dataframe.values
# analyzing the distributions of numerical columns of the dataframe

plt.figure(figsize = (10,5)) #(Width, height) in figsize

plt.subplot(2,3,1)

sns.distplot(admission_dataframe['GRE Score'])

plt.subplot(2,3,2)

sns.distplot(admission_dataframe['TOEFL Score'])

plt.subplot(2,3,3)

sns.distplot(admission_dataframe['CGPA'])

plt.subplot(2,3,4)

sns.distplot(admission_dataframe['SOP'])

plt.subplot(2,3,5)

sns.distplot(admission_dataframe['University Rating'])

plt.subplot(2,3,6)

sns.distplot(admission_dataframe['LOR '])

plt.tight_layout()

plt.show()
# analyzing the pair plot distibutions of the numerical variables

sns.pairplot(admission_dataframe[['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR ', 'University Rating' ,'Chance of Admit ']])
# plotting the heat map between the numerical variables

matrix = admission_dataframe[['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR ', 'University Rating' ,'Chance of Admit ']].corr()

# plotting the heatmap using seaborn

sns.heatmap(matrix, annot = True, linecolor= 'white', linewidths= 1)
# analyzing the variation between categorical variables in relation with target variables

plt.figure(figsize=(7,5))

sns.boxplot(admission_dataframe['Research'], admission_dataframe['Chance of Admit '])

plt.tight_layout()

plt.show()
# analyzing the dataframe again

admission_dataframe.head()
# Performing Train Test split

df_train, df_test = train_test_split(admission_dataframe, train_size = 0.70, test_size = 0.30, random_state = 100)
# checking the shapes of train and test dataframe

print(df_train.shape, df_test.shape)
# Dropping the column serial number as it will be no use in modelling

df_train.drop('Serial No.', axis = 1, inplace = True)

df_test.drop('Serial No.', axis =1, inplace = True)
# checking the df_train

df_train.head()
# Scaling the features of the dataframe



# initiating the scaler object

scaler = MinMaxScaler()



# fittting and transforming the data on top of the scaler object for the training data set

df_train[::] = scaler.fit_transform(df_train)
# checking the dataframe after scaling the variables

df_train.head()
# checking the describe of the dataframe to check whether scaling has been done or not

df_train.describe()
# creating X_train and y_train

y_train = df_train.pop('Chance of Admit ')

X_train =  df_train
# checking X_train 

X_train.head()
# checking y_train

y_train.head()
# adding constant to X_train as statsmodel in built doesn't add constant

X_train_sm = sm.add_constant(X_train)



# creating the model object

lr_model_1 = sm.OLS(y_train, X_train_sm)



# fitting the model on top of the data

lr_model_1 = lr_model_1.fit()



# checking the summary statistics of the model

lr_model_1.summary()
# checking the VIF values



# creating the dataframe of the VIF values

VIF = pd.DataFrame()



# adding column to the VIF dataframe for columns of the X_train

VIF['features'] = X_train.columns



# adding column for the VIF values of the features

VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



# sorting the dataframe based on VIF_value

VIF.sort_values('VIF_value', ascending= False, inplace = True)



# checking the dataframe

VIF
# dropping SOP from X_train

X_train.drop('SOP', axis = 1, inplace = True)
# adding constant to X_train as statsmodel in built doesn't add constant

X_train_sm = sm.add_constant(X_train)



# creating the model object

lr_model_2 = sm.OLS(y_train, X_train_sm)



# fitting the model on top of the data

lr_model_2 = lr_model_2.fit()



# checking the summary statistics of the model

lr_model_2.summary()
# checking the VIF values



# creating the dataframe of the VIF values

VIF = pd.DataFrame()



# adding column to the VIF dataframe for columns of the X_train

VIF['features'] = X_train.columns



# adding column for the VIF values of the features

VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



# sorting the dataframe based on VIF_value

VIF.sort_values('VIF_value', ascending= False, inplace = True)



# checking the dataframe

VIF
# Dropping Univeristy Rating from X_train

X_train.drop('University Rating', axis = 1, inplace = True)
# adding constant to X_train as statsmodel in built doesn't add constant

X_train_sm = sm.add_constant(X_train)



# creating the model object

lr_model_3 = sm.OLS(y_train, X_train_sm)



# fitting the model on top of the data

lr_model_3 = lr_model_3.fit()



# checking the summary statistics of the model

lr_model_3.summary()
# checking the VIF values



# creating the dataframe of the VIF values

VIF = pd.DataFrame()



# adding column to the VIF dataframe for columns of the X_train

VIF['features'] = X_train.columns



# adding column for the VIF values of the features

VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



# sorting the dataframe based on VIF_value

VIF.sort_values('VIF_value', ascending= False, inplace = True)



# checking the dataframe

VIF
# Dropping CGPA from X_train

X_train.drop('GRE Score', axis = 1, inplace = True)
# adding constant to X_train as statsmodel in built doesn't add constant

X_train_sm = sm.add_constant(X_train)



# creating the model object

lr_model_4 = sm.OLS(y_train, X_train_sm)



# fitting the model on top of the data

lr_model_4 = lr_model_4.fit()



# checking the summary statistics of the model

lr_model_4.summary()
# checking the VIF values



# creating the dataframe of the VIF values

VIF = pd.DataFrame()



# adding column to the VIF dataframe for columns of the X_train

VIF['features'] = X_train.columns



# adding column for the VIF values of the features

VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



# sorting the dataframe based on VIF_value

VIF.sort_values('VIF_value', ascending= False, inplace = True)



# checking the dataframe

VIF
# Dropping TOEFL Score from X_train

X_train.drop('CGPA', axis = 1, inplace = True)
# adding constant to X_train as statsmodel in built doesn't add constant

X_train_sm = sm.add_constant(X_train)



# creating the model object

lr_model_5 = sm.OLS(y_train, X_train_sm)



# fitting the model on top of the data

lr_model_5 = lr_model_5.fit()



# checking the summary statistics of the model

lr_model_5.summary()
# checking the VIF values



# creating the dataframe of the VIF values

VIF = pd.DataFrame()



# adding column to the VIF dataframe for columns of the X_train

VIF['features'] = X_train.columns



# adding column for the VIF values of the features

VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



# sorting the dataframe based on VIF_value

VIF.sort_values('VIF_value', ascending= False, inplace = True)



# checking the dataframe

VIF
# Dropping Research from X_train

X_train.drop('TOEFL Score', axis = 1, inplace = True)
# adding constant to X_train as statsmodel in built doesn't add constant

X_train_sm = sm.add_constant(X_train)



# creating the model object

lr_model_6 = sm.OLS(y_train, X_train_sm)



# fitting the model on top of the data

lr_model_6 = lr_model_6.fit()



# checking the summary statistics of the model

lr_model_6.summary()
# checking the VIF values



# creating the dataframe of the VIF values

VIF = pd.DataFrame()



# adding column to the VIF dataframe for columns of the X_train

VIF['features'] = X_train.columns



# adding column for the VIF values of the features

VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]



# sorting the dataframe based on VIF_value

VIF.sort_values('VIF_value', ascending= False, inplace = True)



# checking the dataframe

VIF
# Making predictions on the train dataset

y_train_pred = lr_model_6.predict(X_train_sm)
# analyzing the y_train_pred

y_train_pred.head()
# checking the distribution of the error terms

res = y_train - y_train_pred

sns.distplot(res)
# checking for the assumption whether there exists any pattern in the error terms

plt.figure()

sns.scatterplot(y_train, res)

plt.show()
# analyzing the df_test

df_test.head()
# scaling the test data using the scaler object defined for scaling of train dataset

df_test[::] = scaler.transform(df_test)
# analyzing the df_test after scaling

df_test.head()
# creating X_test and y_test

y_test =  df_test.pop('Chance of Admit ')

X_test = df_test
# analyzing X_test

X_test.head()
# analyzing y_test

y_test.head()
# modifying the X_test using the features of the model build

X_test = X_test[X_train.columns]
# adding constant to X_test

X_test_sm = sm.add_constant(X_test)
# making predictions using model

y_test_pred = lr_model_6.predict(X_test_sm)
# checking the r2_score on the predictions made on the test set

r2_score_test = r2_score(y_test, y_test_pred)

r2_score_test
# checking the r2_score on the train set again

r2_score_train = r2_score(y_train, y_train_pred)

r2_score_train
# checking the MSE and RMSE as well on the test data set

mean_squared_error_test =  mean_squared_error(y_test, y_test_pred)

print('The MSE value on test dataset {}'.format(mean_squared_error_test))

RMSE_test = np.sqrt(mean_squared_error_test)

print('The RMSE value on test dataset {}'.format(RMSE_test))
# checking the MSE and RMSE as well on the train data set

mean_squared_error_train =  mean_squared_error(y_train, y_train_pred)

print('The MSE value on train dataset {}'.format(mean_squared_error_train))

RMSE_train = np.sqrt(mean_squared_error_train)

print('The RMSE value on train dataset {}'.format(RMSE_train))
# analyzing the error between y_test and y_test_pred

error = y_test - y_test_pred

error.head()
# creating a dataframe having y_last and the error terms

last_error_datafarme = pd.DataFrame()



# adding column for y_last

last_error_datafarme['True values'] = y_test



# adding error terms as a column

last_error_datafarme['Error values'] = error



# restricting the observation to be last 100 of the validation set

last_error_datafarme = last_error_datafarme.tail(100)



# checking the dataframe

last_error_datafarme.head()
# checking the shape of the last_error_dataframe

last_error_datafarme.shape
# checking the dataframe for which last 100 observations error has been defined

last_error_datafarme.head()
# adding a column for computing RMSE for each data point

last_error_datafarme['RMSE'] = last_error_datafarme['Error values'].apply(lambda x: np.sqrt(x**2))
# checking the dataframe after addition of column

last_error_datafarme.head()