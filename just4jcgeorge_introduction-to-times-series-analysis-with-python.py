import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing required modules

import pandas as pd          

import numpy as np               # For mathematical calculations 

import matplotlib.pyplot as plt  # For plotting graphs 

import datetime as dt

from datetime import datetime    # To access datetime 

from pandas import Series        # To work on series 

%matplotlib inline 



import warnings                   # To ignore the warnings 

warnings.filterwarnings("ignore")





# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()
# Load the 'diet' csv file into a dataframe - diet

diet = pd.read_csv('../input/week6dataset/diet.csv')

diet.head()
# Convert string to datetime64

diet['Week'] = diet['Week'].apply(pd.to_datetime)

diet.info()
# We can use pandas datetime attribute 'dt' to access the datetime components

diet['year'] = diet['Week'].dt.year

print(diet['year'])
# Convert the date index to datetime

diet['year'] = diet['Week'].dt.year





# Plot data  

diet['year'].plot()

plt.show()
# Read the MSFT csv into a dataframe MSFT

MSFT = pd.read_csv('/kaggle/input/financial-time-series-datasets/data/MSFT.csv')



# DateTimeIndex is needed to convert 'Date' from string to DateTime

MSFT['Date'] = pd.to_datetime(MSFT['Date'])



# Convert the daily data to weekly data

MSFT = MSFT.set_index('Date').resample(rule='w', how='last')



# Compute the percentage change of prices

returns = MSFT.pct_change()



# Compute and print the autocorrelation of returns

autocorrelation = returns['Adj Close'].autocorr()

print("The autocorrelation of weekly returns is %4.2f" %(autocorrelation))