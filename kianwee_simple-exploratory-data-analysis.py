# Import required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

sns.set(rc={'figure.figsize':(11, 4)})
def overview():

    data = pd.read_csv("../input/meat-prices-19902020/Meat_prices.csv")

    print("The first 5 rows of data are:\n")

    print(data.head)

    print("\n\n\nDataset has {} rows and {} columns".format(data.shape[0], data.shape[1]))

    print("\n\n\nDatatype: \n")

    print(data.dtypes)

    print("\n\n\nThe number of null values for each column are: \n")

    print(data.isnull().sum())

    print("\n\n\nData summary: \n")

    print(data.describe())

    return data



# Lastly, assigning a variable to overview()

data = overview()
# Replacing %

data = data.replace('%', '', regex=True)



# Dropping rows with NaN values for chicken and beef price % change

data = data.dropna(subset=['Chicken price % Change', 'Beef price % Change'])



# Replacing the remaining NaN values with median

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

data.iloc[:,5:11] = imputer.fit_transform(data.iloc[:,5:11])



# Check to see if all NaN values are resolved

data.isnull().sum()
data.Month  = pd.to_datetime(data.Month.str.upper(), format='%b-%y', yearfirst=False)

data["Chicken price % Change"] = data["Chicken price % Change"].astype("float")

data["Beef price % Change"] = data["Beef price % Change"].astype("float")

print(data.dtypes)
# Indexing month

data = data.set_index('Month')
axes = data[["Chicken Price", "Chicken price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)
axes = data[["Beef Price", "Beef price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)
axes = data[["Lamb price", "Lamb price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)
axes = data[["Pork Price", "Pork price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)
axes = data[["Salmon Price", "Salmon price % Change"]].plot(figsize=(11, 9), subplots=True, linewidth=1)