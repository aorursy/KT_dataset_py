# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import the library
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
df = pd.read_csv('..//input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head() #to load first 5 rows
#Get the number of rows and columns in the data set.
df.shape
#Show all of the column names
df.columns.values
#Check for na or missing data
df.isna().sum()
#Show statistics on the current data 
df.describe()
#Get the number of customers that churned
df['Churn'].value_counts()
#Visualize the count of customer churn
sns.countplot(df['Churn'])
#What percentage of customers are leaving ?
retained = df[df.Churn == 'No']
churned = df[df.Churn == 'Yes']
num_retained = retained.shape[0]
num_churned = churned.shape[0]
#Print the percentage of customers that stayed and left
print( num_retained / (num_retained + num_churned) * 100 , "% of customers stayed with the company.")
#Print the percentage of customers that stayed and left
print( num_churned / (num_retained + num_churned) * 100,"% of customers left the company.")
#Visualize the churn count for both Males and Females
sns.countplot(x='gender', hue='Churn',data = df)
#Visualize the churn count for the internet service
sns.countplot(x='InternetService', hue='Churn', data = df)
numerical_features = ['tenure', 'MonthlyCharges']
fig, ax = plt.subplots(1, 2, figsize=(28, 8))
df[df.Churn == 'No'][numerical_features].hist(bins=20, color="blue", alpha=0.5, ax=ax)
df[df.Churn == 'Yes'][numerical_features].hist(bins=20, color="orange", alpha=0.5, ax=ax)
#Remove the unnecessary column customerID
cleaned_df = df = df.drop('customerID', axis=1)

#Look at the number of rows and cols in the new data set
cleaned_df.shape
#Convert all the non-numeric columns to numerical data types
for column in cleaned_df.columns:
   if cleaned_df[column].dtype == np.number:
      continue
   cleaned_df[column] = LabelEncoder().fit_transform(cleaned_df[column])
#Check the new data set data types
cleaned_df.dtypes
#Scale the cleaned data
X = cleaned_df.drop('Churn', axis = 1) 
y = cleaned_df['Churn']
#Standardizing/scaling the features
X = StandardScaler().fit_transform(X)
#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Create the model
model = LogisticRegression()
#Train the model
model.fit(x_train, y_train)
predictions = model.predict(x_test)
#printing the predictions
print(predictions)
#Check precision, recall, f1-score
print( classification_report(y_test, predictions) )
