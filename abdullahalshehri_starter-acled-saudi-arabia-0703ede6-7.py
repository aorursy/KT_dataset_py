# importing all required libraries



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



data = pd.read_csv('../input/acled_saudi.csv')
# inspecting columns 



data.columns 
# inspecting dataframe shape 



data.shape
# inspecting dataframe head 



data.head(3)

# checking null values

data.isnull().sum()
# exploring some features 



data.admin1.value_counts()
# exploring some features 



data.admin2.value_counts()[:15]
# checking data types



data.info()
# how many unique values in each column

data.nunique()
# fatalities distribution and outliers based on year

sns.boxplot(y='fatalities', data= data , x='year')

plt.show();
# fatalities distribution and outliers based on year and event type

sns.catplot(x='event_type' ,y='fatalities', data=data  , hue='year', height=6.5 , aspect=2.5 , kind='boxen')

plt.show();