import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
#Importing the required File

df =pd.read_excel("../input/companysalarydataset/companysalary.xlsx")
# what are the columns in dataset

df.columns
# top rows in dataset 

df.head(10)
#Bottom rows of dataset

df.tail()
Scaler = MinMaxScaler()

df_values = df.values

df_valued = Scaler.fit_transform(df_values)

normalized_df = pd.DataFrame(df_valued)

normalized_df
# Describe  function used to find the total count, mean, standard deviation, minimum value, maximum value

df.describe()
std_scaler = StandardScaler()

df_values = df.values

df_std = std_scaler.fit_transform(df_values)

std_df = pd.DataFrame(df_std)

std_df
df.describe()
#nulls

print(df.isnull().sum())
# To remove the null values we can use fillna method 

df.Salary = df.Salary.fillna("unknown")

print(df.isnull().sum())
print(df.shape)

print("\n")

print(df.dtypes)
df.info()

print('_'*40)
df = pd.read_excel("../input/prodcut/ptoduct.xlsx")

df
df.info()
lb = LabelEncoder()

df['Country'] = lb.fit_transform(df['Country'])

df
df.info()
new = {'No':0, 'Yes':1}

df.Purchased = df.Purchased.map(new)

df
df.info()