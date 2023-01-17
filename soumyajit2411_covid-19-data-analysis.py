#Importing required packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Loading dataset
df_covid = pd.read_csv('../input/covid19-dataset/owid-covid-data.csv')
#Let's check how the data is distributed
df_covid.head()
# df_covid.tail()
# df_covid.shape
# df_covid.columns
#Information about the data columns
df_covid.info()
# Information on the Dataset
df_covid.describe()
#Checking Null values on the dataset
df_covid.isnull().sum()
# Subsetting those rows where location is India
df=df_covid[df_covid["location"]=="India"]
# Let's check the new dataset
df.head()
# df.tail()
# df.shape
# Information about the new Dataset
df.info()
# More information on the new dataset
df.describe()
#Checking Null values on the new dataset
df.isnull().sum()
# To check for number of columns
df.columns
# Defining a variable cols 
cols=['total_cases', 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand',
       'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
       'stringency_index', 'population', 'population_density', 'median_age',
       'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'cvd_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy']
#Plotting Histograms
for i in cols:
  sns.distplot(df[i],kde=False,hist=True,bins=11,hist_kws=dict(edgecolor="k", linewidth=1))
  plt.title("Histogram")
  plt.ylabel("Frequency")
  plt.show()
#Plotting Boxplots

# for i in cols:
#   sns.boxplot(y=i, data = df)
#   #sns.boxplot(y=i, data = df)
#   plt.title("Boxplot 11")
#   plt.show()
# Checking the mean of each column in the new dataset
df.mean()
# Checking the median of each column in the new dataset
df.median()
# Checking the mode of each column in the new dataset
df.mode()
#sns.pairplot(df)
# Plotting ScatterPlots

# for i in cols:
#   for j in reversed(cols):
#     sns.scatterplot(x=i, y=j, data=df)
#     plt.title("Scatter Plot")
#     plt.show()
# Defining  new variables for cols 
cols=[ 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand',
       'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
       'stringency_index', 'population', 'population_density', 'median_age',
       'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'cvd_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy']
# Plotting ScatterPlots
for i in cols:
    sns.scatterplot(x='total_cases', y=i, data=df)
    plt.title("Scatter Plot")
    plt.show()
#Plotting Lineplots

# for i in cols:
#   for j in reversed(cols):
#     sns.lineplot(x=i,y=j,data=df)
#     plt.title("Line Plots")
#     plt.show()
# Plotting Lineplots
for i in cols:
    sns.lineplot(x='total_cases',y=i,data=df)
    plt.title("Line Plots")
    plt.show()
# Removing outliers in the dataset
df.drop(df.index[168],inplace=True)
# Checking the info once again
# df.info
# Observe the changes after removing outliers
df.describe()
# Replacing the null values with the mean of the columns
#df.fillna(df.mean(), inplace=True)
df=df.fillna(df.mean())
# Replacing the null categorical columns with their mode
df['tests_units'].fillna(df['tests_units'].mode()[0],inplace=True)
# Now there are no Null values in the Dataset
#df.info()
df.isnull().sum()
# Convert date column to ordinal
import datetime as dt 
df["date"]=pd.to_datetime(df["date"]) 
df["date"]=df["date"].map(dt.datetime.toordinal)
# Date column was changed to ordinal
df.head()
# Droping the categorical columns to prepare the dataset for training
df.drop(['iso_code', 'continent','location','tests_units'], axis=1, inplace=True)
df.head()
# Create arrays for the features and the response variable
y = df["total_cases"].values
X = df.drop(["total_cases"],axis=1).values
# Train and Test splitting of data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create the regressor: reg
reg= LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred=reg.predict(X_test)

#Score the model
reg.score(X_test,y_test)
# Compute and print R^2 and RMSE
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor()
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 

# Predict on the test data: y_pred
y_pred=rf.predict(X_test)

#Score the model
rf.score(X_test,y_test)
#Predict total cases for a new data through Linear regression
reg.predict([[733828,10974,11903,2003,256.568,7.952,8.625,1.451,6084256,163187,4.409,0.118,146132,0.106,76.85,1.38e+09,450.419,28.2,5.989,3.414,6426.674,21.2,282.28,10.39,1.9,20.6,59.55,0.53,69.66]])
#Predict total cases for a new data through RandomForestRegressor
rf.predict([[733828,10974,11903,2003,256.568,7.952,8.625,1.451,6084256,163187,4.409,0.118,146132,0.106,76.85,1.38e+09,450.419,28.2,5.989,3.414,6426.674,21.2,282.28,10.39,1.9,20.6,59.55,0.53,69.66]])