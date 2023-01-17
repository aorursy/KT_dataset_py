import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline 
df=pd.read_csv('../input/OLX_Car_Data_CSV.csv', delimiter=',', encoding = "ISO-8859-1")

df.head()
print(df.columns)
df.dtypes
df.describe()
df.describe(include = "all")
df.info
missing_data = df.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")  
avg_KMsDriven = df["KMs Driven"].astype("float").mean(axis = 0)

print("Average of KMs Driven:", avg_KMsDriven)

df["KMs Driven"].replace(np.nan, avg_KMsDriven, inplace = True)
avg_Year = df["Year"].astype("float").mean(axis = 0)

print("Average of Year:", avg_Year)

df["Year"].replace(np.nan, avg_Year, inplace = True)
# simply drop whole row with NaN in "price" column

df.dropna(subset=["Brand","Condition", "Fuel","Model", "Registered City","Transaction Type"], axis=0, inplace=True)



# reset index, because we droped two rows

df.reset_index(drop=True, inplace=True)
df.replace("", np.nan, inplace = True)

df.head(5)
df['Fuel'].value_counts()
var = df.groupby('Fuel').Price.sum()

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Fuel')

ax1.set_ylabel('Price')

ax1.set_title("Fuel Vs Price")

var.plot(kind='bar')
df['Condition'].value_counts()
var = df.groupby('Condition').Price.sum() 

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Condition Of Car')

ax1.set_ylabel('Increase In price')

ax1.set_title("Condition Vs Price")

var.plot(kind='bar')
df['Brand'].value_counts()
var = df.groupby('Brand').Price.sum()

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Brand')

ax1.set_ylabel('Increase In price')

ax1.set_title("Brand Vs Price")

var.plot(kind='bar')
df['Model'].value_counts()
var = df.groupby('Model').Price.sum() 

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Model Of Car')

ax1.set_ylabel('Increase In price')

ax1.set_title("Model Vs Price")

var.plot(kind='line')
df['Year'].value_counts()
var = df.groupby('Year').Price.sum() 

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Condition Of Car')

ax1.set_ylabel('Increase In price')

ax1.set_title("Year Vs Price")

var.plot(kind='line')
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(df['KMs Driven'],df['Price']) 

plt.show()
df['Transaction Type'].value_counts()
var = df.groupby('Transaction Type').Price.sum()

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel('Transaction Type')

ax1.set_ylabel('Increase In price')

ax1.set_title("Transaction Type Vs Price")

var.plot(kind='bar')