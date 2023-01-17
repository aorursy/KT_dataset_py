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
# Importing the data set
df = pd.read_csv("/kaggle/input/automobile-dataset/Automobile_data.csv")
df.head()
# Since pandas does not understands '?' values we must replace it with 'NAN' values to perfom pandas operation.
df.replace("?",np.nan,inplace = True)

# Loading dataset
df.head()
# Before dealing with missing values we must check for the data type
print(df.dtypes)
# Calculate number of nan values and percentage of nan values in 'normalized-losses'
# Number of missing values of 'normalized-losses' columns.
missing_values = df['normalized-losses'].isnull().sum()
print("The number of missing value is normalized-losses: ",missing_values)
# Percentage of nan values.
total_num_cells = np.product(df['normalized-losses'].shape)
percent_of_nan = (missing_values/total_num_cells)*100
print("The percentage of missing value is normalized-losses: ",percent_of_nan,"%")
# Since percentage of mising values in normalized losses is less we can replace it with either mean or '0'.
# Replace nan values of 'normalised losses' with mean values of the column
# But to find mean we need to change datatype of the 'normalised losses'
mean = df['normalized-losses'].astype(float).mean()
df['normalized-losses'].replace(np.nan,mean,inplace = True)
df.head()
# Similarly we shall replace for other numerical columns

# For 'bore' column
m_1 = df['bore'].astype('float').mean(axis=0)
df['bore'].replace(np.nan,m_1,inplace = True)

# For 'horsepower' column
m_2 = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan,m_2,inplace = True)

# For 'Peak-rpm' column
m_3 = df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan,m_3,inplace = True)

# For 'Stroke' column
m_4 = df['stroke'].astype('float').mean(axis = 0)
df['stroke'].replace(np.nan,m_4,inplace = True)

# Checking for null value
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    
# For categorical values we replace missing values with most occurred values
print(df['num-of-doors'].value_counts())
print()
print("Most occurred value",df['num-of-doors'].value_counts().idxmax())

# Replace missing values with 'four'
df['num-of-doors'].replace(np.nan,'four',inplace = True)
# Delete the nan values from price column as  price is what we want to predict. 
# Any data entry without price data cannot be used for prediction
df.dropna(subset=["price"],axis=0,inplace = True)

# Check for nan 
print("The number of missing values in price:",df['price'].isnull().sum())
# Checking for more columns with null values
print(df.isnull().sum())
# Check for data format and make correction
df.dtypes
# Convert data format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df[["horsepower"]] = df[["horsepower"]].astype("int")
# Check 
df.dtypes
# Define important variables for prediction
# Select variables with correlation close to either -1 or 1 with price
df.corr()
# Since more than one variable has impact on price column
# We must use Multipe Linear regression and here we selected variables most close to -1 or 1
# The equation is given by 
# yhat = a + b1𝑋1 + b2𝑋2 + b3𝑋3 + b4𝑋4

# Training and Testing Data
# X has data of all the important variables/ Predictor variable/ Independent variables
x = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# Y has data of dependent variable/ target variable
y = df['price']
# For predictive analysis, We first split data into two groups training data and testing data(For this case train_data = 80% and test_data=20%)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Here, the variables:
# x_train = 80% data of independent variables.
# y_train = 80% data from price column w.r.t x_train data.
# x_test  = 20% data of independent variables for prediction of price.
# y_test  = The price predicted from above 'x_test' data is checked with original y_test data to get the accuracy of the model.

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
reg = LinearRegression()

# Training the model
reg.fit(x_train,y_train)

# Price prediction based on x_test data
price_predict = reg.predict(x_test)
price_pred = np.round(price_predict,2)
price_pred_df = pd.DataFrame({'Predicted_price':price_pred})

# Printing first 10 values of price prediction dataframe
print(price_pred_df.head(10))

# Accuracy percentage using y_test data
# Here R^2 method is used to evaluate the model
accuracy = r2_score(y_test,price_predict)
print()
print("The accuracy of the model based on current test data: ",accuracy*100,"%")
# Model evaluation
import seaborn as sns
import matplotlib.pyplot as plt

print("Here is comparision between predicted values from train_data and test_data")

# Distribution plot for Training data
train_data_plot = reg.predict(x_train)
plt.figure(figsize=(7,7))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual price")
sns.distplot(train_data_plot, hist=False, color="b", label="Predicted price" , ax=ax1)

# Note here:
# t_data --> train data
# a_price --> actual price
# p_price --> pridected price
plt.title('A_price of t_data   vs   p_price of t_data')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# Distribution plot for Test data
# Using 'price_predict' variable from above
plt.figure(figsize=(7,7))

ax2 = sns.distplot(df['price'], hist=False, color="r", label="Actual price")
sns.distplot(price_predict, hist=False, color="b", label="Predicted price" , ax=ax2)

# Note here:
# t_data --> test data
# a_price --> actual price
# p_price --> pridected price
plt.title('A_price of t_data   vs   p_price of t_data')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

print("Here see that the predicted values are close to the actual values, since the two distributions overlap a bit","."
      ,"Thus the model is reasonably correct.")