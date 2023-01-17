# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# reading csv file into dataframe.

rain_df = pd.read_csv('../input/weatherAUS.csv')
# print the head of the dataframe.

rain_df.head()
# print basic information about each column in the dataframe.

rain_df.info()
# print information like mean, min, max etc for numeric columns in the dataframe. 

rain_df.describe()
# the output here shows that column 'RainTomorrow' has more 'No' values and less 'Yes' values. This may impact the performance of our model.

rain_df['RainTomorrow'].value_counts()
# prints the number of NULL values in each column.

rain_df.isnull().sum()
# handling NULL values in the column.

rain_df['MaxTemp'].fillna(rain_df['MaxTemp'].mean(),inplace=True)

rain_df['MinTemp'].fillna(rain_df['MinTemp'].mean(),inplace=True)

rain_df['Temp9am'].fillna(rain_df['Temp9am'].mean(),inplace=True)

rain_df['Temp3pm'].fillna(rain_df['Temp3pm'].mean(),inplace=True)

rain_df['WindSpeed9am'].fillna(rain_df['WindSpeed9am'].mean(),inplace=True)

rain_df['WindSpeed3pm'].fillna(rain_df['WindSpeed3pm'].mean(),inplace=True)

rain_df['Humidity9am'].fillna(rain_df['Humidity9am'].mean(),inplace=True)

rain_df['Humidity3pm'].fillna(rain_df['Humidity3pm'].mean(),inplace=True)

rain_df['WindGustSpeed'].fillna(rain_df['WindGustSpeed'].mean(),inplace=True)

rain_df['Pressure9am'].fillna(rain_df['Pressure9am'].mean(),inplace=True)

rain_df['Pressure3pm'].fillna(rain_df['Pressure3pm'].mean(),inplace=True)
# handling NULL values in the column.

rain_df['Evaporation'].fillna(rain_df['Evaporation'].mean(),inplace=True)

rain_df['Sunshine'].fillna(rain_df['Sunshine'].mean(),inplace=True)

rain_df['Cloud9am'].fillna(rain_df['Cloud9am'].mean(),inplace=True)

rain_df['Pressure3pm'].fillna(rain_df['Pressure3pm'].mean(),inplace=True)

rain_df['Cloud3pm'].fillna(rain_df['Cloud3pm'].mean(),inplace=True)

rain_df['WindGustDir'].fillna('Unknown',inplace=True)

rain_df['WindDir9am'].fillna('Not Available',inplace=True)

rain_df['WindDir3pm'].fillna('Not known',inplace=True)
# dropping the remaining rows with NULL values.

rain_df.dropna(inplace=True)
# dropping columns which are not useful for prediction.

rain_df.drop(columns=['Date','Rainfall','RISK_MM'],inplace=True)
# printing the number of rows and columns of the dataframe.

rain_df.shape
# converting categorical columns to numerical.

rain_dummies = pd.get_dummies(rain_df,drop_first=True)

rain_df = pd.concat([rain_df,rain_dummies],axis=1)
# printing the number of rows and columns of the dataframe.

rain_df.shape
# dropping categorical columns. 

rain_df.drop(columns=['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow'],inplace=True)
# printing the number of rows and columns of the dataframe.

rain_df.shape
# importing the model and train_test_split.

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
# printing column names.

rain_df.columns
# seperating target column from the other columns.

X = rain_df.drop(columns='RainTomorrow_Yes')

y = rain_df['RainTomorrow_Yes']
# splitting data into training and test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=102)
# printing the number of rows and columns of the dataframes.

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# import StandardScaler for scaling the features.

from sklearn.preprocessing import StandardScaler
# creating an instance of StandardScaler.

scaler = StandardScaler()
# scaling down the training data.

X_train_scaled = scaler.fit_transform(X_train)
# creating an instance of the model.

lr_model = LogisticRegression(solver='newton-cg', max_iter=10000)
# train the model.

lr_model.fit(X_train_scaled,y_train)
# scaling down the test data.

X_test_scaled = scaler.transform(X_test)
# performing predictions.

predictions = lr_model.predict(X_test_scaled)
# import metrics to evaluate the model.

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
# print classification report.

print(classification_report(y_test,predictions))
# print confusion matrix.

print(confusion_matrix(y_test,predictions))
# print accuracy score.

print(accuracy_score(y_test,predictions))