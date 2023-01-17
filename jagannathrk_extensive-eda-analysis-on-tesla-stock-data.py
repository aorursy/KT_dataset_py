# Import necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from pandas.plotting import table

import datetime as dt
# Load the dataset

Data = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")
# View the dataset

Data # 2416 rows * 7 columns
# Check datatypes of the variables1

Data.dtypes
# Convert the Date column to DateTime object

Data['Date'] = pd.to_datetime(Data['Date'])
# Disable the scientific notation to understand figures better

pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Get initial descriptive statistics

Data.describe(include="all")
# Before proceeding, check for NULL values. If found, perform imputation

Data.isnull().values.sum() # In this case, it is 0. So, we can proceed
# A glimpse of how the market shares varied over the given time



# Create a list for numerical columns that are to be visualized

Column_List = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']



# Plot to view the same

Data.plot(x = "Date", y = Column_List, subplots = True, layout = (3, 3), figsize = (15, 15), sharex = False, title = "Stock Value Trend from 2010 - 2012", rot = 90)
# Visualize the spread and skweness through the distribution plot



# Use the Column_List : list initialized above in the following steps

fig, ax = plt.subplots(len(Column_List), figsize = (15, 10))



for i, col_list in enumerate(Column_List):

    sns.distplot(Data[col_list], hist = True, ax = ax[i])

    ax[i].set_title ("Frequency Distribution of" + " " + col_list, fontsize = 10)

    ax[i].set_xlabel (col_list, fontsize = 8)

    ax[i].set_ylabel ('Distribution Value', fontsize = 8)

    fig.tight_layout (pad = 1.1) # To provide space between plots

    ax[i].grid('on') # Enabled to view and make markings
# Check for factors responsible in overall volume trade

fig, ax = plt.subplots (figsize = (10, 10))

corr_matrix = Data.corr() # Perform default correlation using Pearson Method 



# Plot the correlation matrix in a heatmap to understand better

sns.heatmap(corr_matrix, xticklabels = corr_matrix.columns.values, yticklabels = corr_matrix.columns.values)
# View the matrix in a table to identify the numerical values of strengths

corr_matrix
# Generate whisker plots to detect the presence of any outliers

fig, ax = plt.subplots (len(Column_List), figsize = (10, 20))



for i, col_list in enumerate(Column_List):

    sns.boxplot(Data[col_list], ax = ax[i], palette = "winter", orient = 'h')

    ax[i].set_title("Whisker Plot for Outlier Detection on" + " " + col_list, fontsize = 10)

    ax[i].set_ylabel(col_list, fontsize = 8)

    fig.tight_layout(pad = 1.1)
# It is evident from the whisker plots that there are some outliers in all the variables



# Remove the variables either using IQR technique or Z-Score

Descriptive_Statistics = Data.describe()

Descriptive_Statistics = Descriptive_Statistics.T # Convert into a dataframe



# Extract the IQR values 

Descriptive_Statistics['IQR'] = Descriptive_Statistics['75%'] - Descriptive_Statistics['25%']



# In this scenario, the outliers are removed using Z-Score due to the variability in historical data

Data = Data[(np.abs(stats.zscore(Data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])) < 3).all(axis = 1)] # abs for (+/-) 3-sigma

Data = Data.reset_index() # Due to elimination of rows, index has to be reset



# Now compare the new dimension with the old one (The one mentioned during the reading of the file)

Data # 2359 * 8
# Since the data is a time series data, we should be able to predict the future through forecasting techniques



# Delete the index column due to reset

#del Data['index']



# Decompose the time series year-wise and month-wise to analyse further

Data['Year'] = Data['Date'].dt.year

Data['Month'] = Data['Date'].dt.month

Data['WeekDay'] = Data['Date'].dt.weekday



# Firstly, plot the data year-wise to see the duration of when it hiked and dipped

fig, ax = plt.subplots(len(Column_List), figsize = (10, 20))



# Group the data by year and plot

for i, col_list in enumerate(Column_List):

    Data.groupby('Year')[col_list].plot(ax = ax[i], legend = True)

    ax[i].set_title("Stock Price Movement Grouped by Year on" + " " + col_list, fontsize = 10)

    ax[i].set_ylabel(col_list + " " + "Price", fontsize = 8)

    fig.tight_layout(pad = 1.1)

    ax[i].yaxis.grid(True) # To enable grid only on the Y-axis
# Visualzing only the total volume of stocks traded grouped year-wise

check = Data.groupby('Year')['Volume'].sum()

plt.figure(figsize = (30, 4))

ax1 = plt.subplot(121)

check.plot(y = "Volume", legend = False, fontsize = 12, sharex = False, title = "Total Volume of Stocks Traded Year-wise from 2010 - 2020", rot = 90, color = "green")

ax1.ticklabel_format(useOffset = False, style = 'plain')

ax1.set_ylabel("Total Stock Volumes")

ax1.yaxis.grid(True)



# Visualzing only the total volume of stocks traded grouped month-wise

check = Data.groupby('Month')['Volume'].sum()

plt.figure(figsize = (30, 4))

ax1 = plt.subplot(121)

check.plot(y = "Volume", legend = False, fontsize = 12, sharex = False, title = "Total Volume of Stocks Traded Month-wise from 2010 - 2020", rot = 90, color = "blue")

ax1.ticklabel_format(useOffset = False, style = 'plain')

ax1.set_ylabel("Total Stock Volumes")

ax1.yaxis.grid(True)



# Visualzing only the total volume of stocks traded grouped weekday-wise

check = Data.groupby('WeekDay')['Volume'].sum()

plt.figure(figsize = (30, 4))

ax1 = plt.subplot(121)

check.plot(y = "Volume", legend = False, fontsize = 12, sharex = False, title = "Total Volume of Stocks Traded WeekDay-wise from 2010 - 2020", rot = 90, color = "red")

ax1.ticklabel_format(useOffset = False, style = 'plain')

ax1.set_ylabel("Total Stock Volumes")

ax1.yaxis.grid(True)
# Analyse based on Year

for i, col_list in enumerate(Column_List):

    var = Data.groupby('Year')[col_list].sum()

    

# Convert the variable into a pandas dataframe

var = pd.DataFrame(var)



# Plot to understand the trend

plt.figure(figsize = (16, 7))

ax1 = plt.subplot(121)

var.plot(kind = "pie", y = "Volume", legend = False, fontsize = 12, sharex = False, title = "Time Series Influence on Total Volume Trade by Year", ax = ax1)



# Plot the table to identify numbers

ax2 = plt.subplot(122)

plt.axis('off') # Since we are plotting the table

tbl = table(ax2, var, loc = 'center')

tbl.auto_set_font_size(False)

tbl.set_fontsize(12)

plt.show()
# Analyse based on Year

for i, col_list in enumerate(Column_List):

    var = Data.groupby('Month')[col_list].sum()

    

# Convert the variable into a pandas dataframe

var = pd.DataFrame(var)



# Plot to understand the trend

plt.figure(figsize = (16, 7))

ax1 = plt.subplot(121)

var.plot(kind = "pie", y = "Volume", legend = False, fontsize = 12, sharex = False, title = "Time Series Influence on Total Volume Trade by Month", ax = ax1)



# Plot the table to identify numbers

ax2 = plt.subplot(122)

plt.axis('off') # Since we are plotting the table

tbl = table(ax2, var, loc = 'center')

tbl.auto_set_font_size(False)

tbl.set_fontsize(12)

plt.show()
# Analyse based on Year

for i, col_list in enumerate(Column_List):

    var = Data.groupby('WeekDay')[col_list].sum()

    

# Convert the variable into a pandas dataframe

var = pd.DataFrame(var)



# Plot to understand the trend

plt.figure(figsize = (16, 7))

ax1 = plt.subplot(121)

var.plot(kind = "pie", y = "Volume", legend = False, fontsize = 12, sharex = False, title = "Time Series Influence on Total Volume Trade by WeekDay", ax = ax1)



# Plot the table to identify numbers

ax2 = plt.subplot(122)

plt.axis('off') # Since we are plotting the table

tbl = table(ax2, var, loc = 'center')

tbl.auto_set_font_size(False)

tbl.set_fontsize(12)

plt.show()