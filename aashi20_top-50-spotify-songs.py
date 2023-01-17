# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt

import seaborn as sns
# Loading dataset and priniting top 5 rows

top50 = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding='cp1252')

top50.head()

# Checking the statistics of dataset for numerical variables

top50.describe()
# Checking the statistics of dataset for categorical variables

top50.describe(include='O')
# Droping the Unnamed fiels as it adds no value to our analysis

top50=top50.drop('Unnamed: 0', axis=1)

top50.head()
#Checking If Null Values Exist

top50.isna().sum()
#DataAnalysis

#Checking Most Popular Genre



top50['Genre'].value_counts()
# Checking Histogram of Genre  

plt.figure(figsize=(15,10))

sns.countplot(top50['Genre'])

plt.title('Distribution by Genre')

plt.show()
# Combine related Genres as many variations of each genre are present in data set

# Selecting rows where Genre contains the word "pop"

sns.countplot(top50[top50['Genre'].str.contains('pop')]['Genre'])

plt.title('Distribution by Pop Genre')

plt.show()
top50.loc[top50['Genre'].str.contains('pop', case=False), 'Genre'] = 'Pop'

top50
top50['Genre'].value_counts()
top50.loc[top50['Genre'].str.contains('hip', case=False), 'Genre'] = 'Hip Hop'

top50
top50.loc[top50['Genre'].str.contains('rap', case=False), 'Genre'] = 'Rap'
top50.loc[top50['Genre'].str.contains('reggaeton', case=False), 'Genre'] = 'Reggaeton'
top50['Genre'].value_counts()
top50.head(10)
top50.head()
# Checking pairplots of all variables first. 

plt.figure(figsize=(20,10))

sns.pairplot(top50, hue='Genre')

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (20, 10))

sns.heatmap(top50.corr(), annot = True, cmap="YlGnBu")

plt.show()
top50.drop('Track.Name', axis=1, inplace=True)
top50.drop('Artist.Name', axis=1, inplace=True)
# importing library for label encoding the Genre Data. 

from sklearn.preprocessing import LabelEncoder

# creating object for label encoding. 

le = LabelEncoder()
# Encoding the Genre column. 

top50.Genre = le.fit_transform(top50.Genre)
top50.head(15)
# Check the distribution of target variable. 

plt.figure(figsize=(20,10))

sns.distplot(top50.Popularity)

plt.show()
# Creating the Features and Targets datasets. 

X = top50[['Genre', 'Beats.Per.Minute', 'Energy', 'Danceability',

       'Loudness..dB..', 'Liveness', 'Valence.', 'Length.', 'Acousticness..',

       'Speechiness.']]



y = top50.Popularity
# Importing library for Train Test split. 

from sklearn.model_selection import train_test_split

# Creating the splits. 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=50)
# Importing library for standard scaling

from sklearn.preprocessing import StandardScaler

# Creating the scaler object

scaler = StandardScaler()
# Scaling the Training and Testing Data. 

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Checking the dimensions of the training and testing sets. 

print("Training Feature data : ", X_train.shape)

print("Training Feature data : ", X_test.shape)

print("Training Feature data : ", y_train.shape)

print("Testing Target data : ", y_test.shape)
# Checking the training data. 

X_train
# Import the libraries .

from sklearn.linear_model import LinearRegression

# Creating the object

regressor = LinearRegression()
# Fit the model. 

regressor.fit(X_train, y_train)
# Predicting the test results. 

y_pred = regressor.predict(X_test)

# Checking the predictions. 

y_pred
# Checking the actuals

y_test
# Creating dataframe of features and coefficients. 

output = {'Features': X.columns, 'Coefficient': regressor.coef_}

output_df = pd.DataFrame(output)

output_df.sort_values('Coefficient')
# Checking RMSE



# Import libraries. 

from sklearn.metrics import mean_squared_error



# Checking the RMSE

mean_squared_error(y_pred, y_test)

# Checking the intercept

regressor.intercept_
# Importing the RFE Library. 

from sklearn.feature_selection import RFE
# Running RFE with the output number of the variable equal to 5

# We select 5 as we have total 10 variables, hence 5 looks to be a good number,

# Considering we do not loose much information from the functional perspective as well .. !! 

rfe = RFE(regressor, 5) # running RFE

rfe = rfe.fit(X_train, y_train) # Fitting the training data
# Getting the columns with RFE

list(zip(X.columns,rfe.support_,rfe.ranking_))
# Creating new set of features. 

X_new = X[['Genre', 'Energy', 'Valence.', 'Length.', 'Speechiness.']]
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Scaling the Training and Testing Data. 

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Creating the object

regressor_rfe = LinearRegression()
# Fitting the model. 

regressor_rfe.fit(X_train, y_train)
# Getting the predictions. 

y_pred_rfe = regressor_rfe.predict(X_test)

# Checking the RMSE

mean_squared_error(y_pred_rfe, y_test)
# Checking the intercept

regressor_rfe.intercept_
# Creating dataframe of features and coefficients. 

output_rfe = {'Features': X_new.columns, 'Coefficient': regressor_rfe.coef_}

output_df_rfe = pd.DataFrame(output_rfe)

output_df_rfe
# We will compare the differences as well, if possible. 

prediction_diff = y_pred_rfe - y_test

check_predictions = {'Predictions': y_pred_rfe, 'Actuals': y_test, 'Difference': prediction_diff}

check_predictions_df = pd.DataFrame(check_predictions)

check_predictions_df.sort_values('Difference')

# Plotting y_test and y_pred to understand the spread.

fig = plt.figure(figsize=(20,10))

plt.scatter(y_test,y_pred)

fig.suptitle('Actuals v/s Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Actuals', fontsize=18)                          # X-label

plt.ylabel('Predicted', fontsize=16)  