# Importing the libraries

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import sklearn

import sys
df= pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df.columns # Tells us which column names contains spaces hence need to be renamed
# rename the columns



df.rename(columns = {'Chance of Admit ':'Chance of Admit', 'LOR ': 'LOR'}, inplace=True)

df.columns # Check if fixed
# Exploratory Data Analysis



df.isnull().sum()



# There are no missing values

# Eliminate the irrelevant data : input X (features matrix) is everything but serial No

# Output Y is : 'Chance of Admit' Column
df.drop('Serial No.', axis=1, inplace= True)  

# Got rid of the column and modified df
df.head()
# Figure out the correlation strength between input values and output

fig, ax= plt.subplots(figsize=(8,5))

sns.heatmap(df.corr(), annot=True, cmap='Reds')
# The strongest contributers are CGPA, GRE Score, TOEFL Score
sns.distplot(df['GRE Score'])
sns.distplot(df['CGPA'])  # distribution is normal std is 0.6 and mean value is about 8.5
sns.distplot(df['TOEFL Score'])
sns.lmplot(x="CGPA",y="Chance of Admit", data=df) 

# Alternatively : sns.regplot(x=df["CGPA"],y=df["Chance of Admit"], data=df)
sns.lmplot(x="GRE Score",y="Chance of Admit", data=df) 

# Further Analysis of Data 



fig, ax = plt.subplots(figsize=(8,6))

sns.countplot(x= 'Research', data=df)

# or sns.countplot(df['Research'])

plt.title('Research Experience')

plt.ylabel('Number of Applicants')

ax.set_xticklabels(['No Research Experience', 'Has Research Experience'])
fig, ax = plt.subplots(figsize=(8,6))

sns.countplot(df['University Rating'])

plt.title('University Rating')

plt.ylabel('Number of Applicants')
# Preparing Data Before Model Selection

# X is the features matrix : Input parameters for the model

X= df.drop('Chance of Admit', axis =1)

X.head()
# y is the target value : output

y = df['Chance of Admit']
X.shape, y.shape  # for matrix multiplication to work number of rows must be the same y= x*b
# split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)
# Choose the Algorithms 
# 1. Linear Regression

from sklearn.linear_model import LinearRegression  # Choose the Model from class



linear = LinearRegression() # instantiate the object

linear.fit(X_train,y_train)



y_predict =linear.predict(X_test)



linear.score(X_test, y_test)
# Repeat the same using another Model

# 2. Decision Trees

from sklearn.tree import DecisionTreeRegressor



decision = DecisionTreeRegressor(random_state = 5, max_depth = 4) # tried different depths and 4 worked the best

decision.fit(X_train, y_train)

y_predict = decision.predict(X_test)

decision.score(X_test,y_test)
# 3. Random Forests

from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(random_state= 3, max_depth=3)



forest.fit(X_train, y_train)



y_predict = forest.predict(X_test)



forest.score(X_test, y_test)
# The best models seems to be linear regression and random forest.


