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
#Importing Seaborn



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#Reading from the CSV File



df = pd.read_csv("../input/insurance.csv")
#Check the header to get a rough idea about the columns and rows



df.head()
#Describe the dataframe to know about it's statistics 



df.describe().T
#Obtaining the total rows and columns (Entries)



df.shape
#We need to get to know about the data types of columns



df.info()
#We are checking for Null values



df.isna().sum()
#We are looking for duplicated rows 



df.duplicated().sum()
#Removing duplicated rows



df = df.drop_duplicates()

df.duplicated().sum()
#Checking the dataframe



df
#Checking the 'Sex' column for any ambiguous/incorrect entries



df.sex.unique()
#Checking the 'Region' column for any ambiguous/incorrect entries



df.region.unique()
#Checking the 'Smoker' column for any ambiguous/incorrect entries



df.smoker.value_counts()
#Replacing 'no' with 0 and 'yes' with 1 in the dataframe



df.smoker.replace({"no":0,"yes":1}, inplace=True)
df
#Checking the Correlation between each columns



df.corr()
#We are using Pairplot to get a visual representation of complete data distribution



sns.pairplot(data=df)
#We use regplot to draw a linear regression line between two columns, namely, age & expenses here.



sns.regplot(x=df["age"],y=df["expenses"])
#We use scatterplot so that we can include the smoker column, which has high correlation with expenses 

#and see whether it gives any knowledge on the dataframe 



sns.scatterplot(x=df["bmi"],y=df["expenses"],hue=df["smoker"])
#We can alter the parameters to get a better visual representation



sns.scatterplot(x=df["age"],y=df["expenses"],hue=df["smoker"])
#We will make use of the lmplot to draw two regression lines for the parameters.



sns.lmplot(x="bmi", y="expenses", hue="smoker", data=df)
#We can make use of swarmplot to better understand the relationship between expenses and smoker columns



sns.swarmplot(x=df["smoker"],y=df["expenses"])
#Selecting the categorical columns



df_categorical_col = df.select_dtypes(exclude=np.number).columns

df_categorical_col
#Selecting the numerical columns



df_numeric_col = df.select_dtypes(include=np.number).columns

df_numeric_col
#Get the truth table of each row for the categorical columns



df_onehot = pd.get_dummies(df[df_categorical_col])
#Viewing the obatined truth table



df_onehot
#Concatenation of encoded data and existing numerical columns we obtained earlier.



df_after_encoding = pd.concat([df[df_numeric_col],df_onehot], axis = 1)

df_after_encoding
#Importing necessary libraries for Linear Regression



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#Selecting the 'y' value (Target Data)



y = df_after_encoding["expenses"]
#Selecting the 'x' value (Coefficients array)



x = df_after_encoding.drop(columns = "expenses")
#Splitting the dataframe into train & test datasets in 70:30 ratio



train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1)
#Selecting the model



model = LinearRegression()
#We need to draw a best-fit line for our model



model.fit(train_x,train_y)
#Print the obtained 'c' value



print(model.intercept_)
#Print the obtained 'x' coefficients value



print(model.coef_)
#We are Predicting the target data for our dataset



print("Predicting train data")

train_predict = model.predict(train_x)

print("Predicting test data")

test_predict = model.predict(test_x)



#Test using MAE, MSE, RMSE, R^2 Error

print(" ")

print("MAE")

print("Train data: ",mean_absolute_error(train_y,train_predict))

print("Test data: ",mean_absolute_error(test_y,test_predict))

print(" ")

print("MSE")

print("Train data: ",mean_squared_error(train_y,train_predict))

print("Test data: ",mean_squared_error(test_y,test_predict))

print(" ")

print("RMSE")

print("Train data: ",np.sqrt(mean_squared_error(train_y,train_predict)))

print("Test data: ",np.sqrt(mean_squared_error(test_y,test_predict)))

print(" ")

print("R^2")

print("Train data: ",r2_score(train_y,train_predict))

print("Test data: ",r2_score(test_y,test_predict))