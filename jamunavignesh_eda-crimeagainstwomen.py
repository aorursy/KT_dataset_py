# Importing required libraries.

import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)

df = pd.read_csv('/kaggle/input/CrimeAgainstWomen.csv')

# To display the top 5 rows

df.head(5)
# To display the bottom 5 rows

df.tail(5)
# Checking the data type

df.dtypes
df.columns
# Total number of rows and columns

print(df.shape)
# Used to count the number of rows 

df.count()
# Finding the null values.

print(df.isnull().sum())
#describing the data

df.describe(include = "all")
# to find the unique values in each column

print(df.nunique())
df['States/UTs'].unique()
df['Crime Head'].unique()
df['2014'].unique()
sexual_harras=df[(df['Crime Head'] == "5.1 - Sexual Harassment")]

sexual_harras.describe()

Andhra = df[df['States/UTs'] == 'Andhra Pradesh']

Andhra
#total crime in the states

total_crime = df[df['Crime Head']== '15 - Total Crimes against Women']

total_crime
#violence above 20,000

high_violence = total_crime[total_crime['2014'] >= 20000].sort_values('2014')

high_violence
Maharashtra = df[df['States/UTs']== 'Maharashtra']

print(Maharashtra )
#violence in maharashtra above 5000

high_violence_maha = Maharashtra[Maharashtra['2014'] >= 2000].sort_values('2014')

high_violence_maha
#reports in Madhya pradesh

MP = df[df['States/UTs']== 'Madhya Pradesh']

high_violence_MP = MP[MP['2014'] >= 3000].sort_values('2014')

high_violence_MP
#reports in Rajasthan

rajasthan = df[df['States/UTs']== 'Rajasthan']

high_violence_rajasthan = rajasthan[rajasthan['2014'] >= 3000].sort_values('2014')

high_violence_rajasthan
#reports in West Bengal

WB = df[df['States/UTs']== 'West Bengal']

high_violence_WB = WB[WB['2014'] >= 3000].sort_values('2014')

high_violence_WB
#reports in Uttar Pradesh

UP = df[df['States/UTs']== 'Uttar Pradesh']

high_violence_UP = UP[UP['2014'] >= 3000].sort_values('2014')

high_violence_UP
