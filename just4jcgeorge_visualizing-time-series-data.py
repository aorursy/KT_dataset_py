# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import pandas

import pandas as pd



# Read in the file content in a DataFrame called discoveries

discoveries = pd.read_csv('/kaggle/input/week6dataset/discoveries.csv')



# Display the first five lines of the DataFrame

discoveries.head()
# Print the data type of each column in discoveries

print(discoveries.dtypes)



# Convert the date column to a datestamp type

discoveries['date'] = pd.to_datetime(discoveries['date'])



# Print the data type of each column in discoveries, again

print(discoveries.dtypes)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Set the date column as the index of your DataFrame discoveries

discoveries.set_index('date', inplace = True)



# Plot the time series in your DataFrame

ax = discoveries.plot(color='blue')



# Specify the x-axis label in your plot

ax.set_xlabel('Date')



# Specify the y-axis label in your plot

ax.set_ylabel('Number of great discoveries')



# Show plot

plt.show()
# Import the matplotlib.pyplot sub-module

import matplotlib.pyplot as plt



# Use the ggplot style

plt.style.use('ggplot')

ax2 = discoveries.plot()



# Set the title

ax2.set_title('ggplot Style')

plt.show()
# Import the matplotlib.pyplot sub-module

import matplotlib.pyplot as plt



# Use the fivethirtyeight style

plt.style.use('fivethirtyeight')



# Plot the time series

ax1 = discoveries.plot()

ax1.set_title('FiveThirtyEight Style')

plt.show()
# Plot a line chart of the discoveries DataFrame using the specified arguments

ax = discoveries.plot(color='blue',figsize =(8, 3), linewidth=2, fontsize=6)



# Specify the title in your plot

ax.set_title('Number of great inventions and scientific discoveries from 1860 to 1959', fontsize=8)



# Show plot

plt.show()
# Select the subset of data between 1939 and 1958

discoveries_subset_2 = discoveries ['1939-01-01': '1958-01-01']

# Plot the time series in your DataFrame as a blue area chart

ax = discoveries_subset_2.plot(color='blue', fontsize=15)



# Show plot

plt.show()
# Plot your the discoveries time series

ax = discoveries.plot(color='blue', fontsize=6)



# Add a red vertical line

ax.axvline('1939-01-01', color='red', linestyle='--')



# Add a green horizontal line

ax.axhline(4, color='green', linestyle='--')



plt.show()
# Plot your the discoveries time series

ax = discoveries.plot(color='blue', fontsize=6)



# Add a vertical red shaded region

ax.axvspan('1900-01-01', '1915-01-01', color='red', alpha=0.3)



# Add a horizontal green shaded region

ax.axhspan(6, 8, color='green', alpha=0.3)



plt.show()
co2_levels = pd.read_csv('/kaggle/input/week6dataset/co2_levels.csv', parse_dates = ['datestamp'])

print(co2_levels)
# Set datestamp column as index

co2_levels = co2_levels.set_index('datestamp')



# Print out the number of missing values

print(co2_levels.isnull().sum())
# Impute missing values with the next valid observation

co2_levels = co2_levels.fillna(method='bfill')



# Print out the number of missing values

print(co2_levels.isnull().sum())
# Compute the 52 weeks rolling mean of the co2_levels DataFrame

ma = co2_levels.rolling(window=52).mean()



# Compute the 52 weeks rolling standard deviation of the co2_levels DataFrame

mstd = co2_levels.rolling(window=52).std()



# Add the upper bound column to the ma DataFrame

ma['upper'] = ma['co2'] + (mstd['co2'] * 2)



# Add the lower bound column to the ma DataFrame

ma['lower'] = ma['co2'] - (mstd['co2'] * 2)



# Plot the content of the ma DataFrame

ax = ma.plot(linewidth=0.8, fontsize=6)



# Specify labels, legend, and show the plot

ax.set_xlabel('Date', fontsize=10)

ax.set_ylabel('CO2 levels in Mauai Hawaii', fontsize=10)

ax.set_title('Rolling mean and variance of CO2 levels\nin Mauai Hawaii from 1958 to 2001', fontsize=10)

plt.show()
# Get month for each dates in the index of co2_levels

index_month = co2_levels.index.month



# Compute the mean CO2 levels for each month of the year

mean_co2_levels_by_month = co2_levels.groupby(index_month).mean()



# Plot the mean CO2 levels for each month of the year

mean_co2_levels_by_month.plot(fontsize = 6)



# Specify the fontsize on the legend

plt.legend(fontsize=10)



# Show plot

plt.show()
# Print out summary statistics of the co2_levels DataFrame

print(co2_levels.describe())



# Print out the minima of the co2 column in the co2_levels DataFrame

print(co2_levels.co2.min())



# Print out the maxima of the co2 column in the co2_levels DataFrame

print(co2_levels.co2.max())
# Generate a histogram

ax = co2_levels.plot(kind = 'hist', bins = 50, fontsize=6)



# Set the labels and display the plot

ax.set_xlabel('CO2', fontsize=10)

ax.set_ylabel('Histogram of CO2 levels in Maui Hawaii', fontsize=10)

plt.legend(fontsize=10)

plt.show()
# Display density plot of CO2 levels values

ax = co2_levels.plot(kind = 'density', linewidth = 4, fontsize=6)



# Annotate x-axis labels

ax.set_xlabel('CO2', fontsize=10)



# Annotate y-axis labels

ax.set_ylabel('Density plot of CO2 levels in Maui Hawaii', fontsize=10)



plt.show()
# Import required libraries

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from statsmodels.graphics import tsaplots



# Display the autocorrelation plot of your time series

fig = tsaplots.plot_acf(co2_levels['co2'], lags=24)



# Show plot

plt.show()
# Import required libraries

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from statsmodels.graphics import tsaplots



# Display the partial autocorrelation plot of your time series

fig = tsaplots.plot_pacf(co2_levels['co2'], lags=24)



# Show plot

plt.show()
# Import statsmodels.api as sm

import statsmodels.api as sm



# Perform time series decompositon

decomposition = sm.tsa.seasonal_decompose(co2_levels)



# Print the seasonality component

print(decomposition.seasonal)
# Extract the trend component

trend = decomposition.trend



# Plot the values of the trend

ax = trend.plot(figsize=(12, 6), fontsize=6)



# Specify axis labels

ax.set_xlabel('Date', fontsize=10)

ax.set_title('Seasonal component the CO2 time-series', fontsize=10)

plt.show()
airline = pd.read_csv('/kaggle/input/week6dataset/airline_passengers.csv', parse_dates = ['Month'], index_col = 'Month')

airline.head()
# Plot the time series in your dataframe

ax = airline.plot(color = 'blue', fontsize=12)



# Add a red vertical line at the date 1955-12-01

ax.axvline('1955-12-01', color='red', linestyle='--')



# Specify the labels in your plot

ax.set_xlabel('Date', fontsize=12)

ax.set_title('Number of Monthly Airline Passengers', fontsize=12)

plt.show()
# Print out the number of missing values

print(airline.isnull().sum())



# Print out summary statistics of the airline DataFrame

print(airline.describe())
# Display boxplot of airline values

ax = airline.boxplot()



# Specify the title of your plot

ax.set_title('Boxplot of Monthly Airline\nPassengers Count', fontsize=20)

plt.show()
# Get month for each dates from the index of airline

index_month = airline.index.month



# Compute the mean number of passengers for each month of the year

mean_airline_by_month = airline.groupby(index_month).mean()



# Plot the mean number of passengers for each month of the year

mean_airline_by_month.plot()

plt.legend(fontsize=20)

plt.show()
# Import statsmodels.api as sm

import statsmodels.api as sm



# Perform time series decompositon

decomposition = sm.tsa.seasonal_decompose(airline)



# Extract the trend and seasonal components

trend = decomposition.trend

seasonal = decomposition.seasonal
airline_decomposed = pd.concat([trend, seasonal], axis = 1)

airline_decomposed.columns = ['trend', 'seasonal']
# Print the first 5 rows of airline_decomposed

print(airline_decomposed.head())



# Plot the values of the df_decomposed DataFrame

ax = airline_decomposed.plot(figsize=(12, 6), fontsize=15)



# Specify axis labels

ax.set_xlabel('Date', fontsize=15)

plt.legend(fontsize=15)

plt.show()
# Read in meat DataFrame

meat = pd.read_csv('/kaggle/input/week6dataset/meat.csv')



# Review the first five lines of the meat DataFrame

display(meat.head(5))
# Convert the date column to a datestamp type

meat['date'] = pd.to_datetime(meat['date'] )



# Set the date column as the index of your DataFrame meat

meat = meat.set_index('date')



# Print the summary statistics of the DataFrame

display(meat.describe())
# Plot time series dataset

ax = meat.plot(linewidth = 2, fontsize = 12)



# Additional customizations

ax.set_xlabel('Date')

ax.legend(fontsize=15)



# Show plot

plt.show()
# Plot an area chart

ax = meat.plot.area(fontsize=12)



# Additional customizations

ax.set_xlabel('Date')

ax.legend(fontsize=15)



# Show plot

plt.show()
# Plot time series dataset using the cubehelix color palette

ax = meat.plot(colormap='cubehelix', fontsize=15)



# Additional customizations

ax.set_xlabel('Date')

ax.legend(fontsize=18)



# Show plot

plt.show()
# Plot time series dataset using the cubehelix color palette

ax = meat.plot(colormap = 'PuOr', fontsize=15)



# Additional customizations

ax.set_xlabel('Date')

ax.legend(fontsize=18)



# Show plot

plt.show()
meat_mean = meat.mean(axis = 0)

meat_mean = pd.DataFrame(meat_mean).transpose()

meat_mean.index = ['mean']

meat_mean
# Plot the meat data

ax = meat.plot(fontsize=6, linewidth=1)



# Add x-axis labels

ax.set_xlabel('Date', fontsize=6)



# Add summary table information to the plot

ax.table(cellText=meat_mean.values,

         colWidths = [0.15]*len(meat_mean.columns),

         rowLabels=meat_mean.index,

         colLabels=meat_mean.columns,

         loc='top')



# Specify the fontsize and location of your legend

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=6)



# Show plot

plt.show()
# Create a facetted graph with 2 rows and 4 columns

meat.plot(subplots=True, 

          layout=(2, 4), 

          sharex=False, 

          sharey=False, 

          colormap='viridis', 

          fontsize=2, 

          legend=False, 

          linewidth=0.2)



plt.show()
# Print the correlation matrix between the beef and pork columns using the spearman method

print(meat[['beef', 'pork']].corr(method='spearman'))
# Print the correlation matrix between the pork, veal and turkey columns using the pearson method

print(meat[['pork', 'veal', 'turkey']].corr(method='pearson'))
# Import seaborn library

import seaborn as sns



# Get correlation matrix of the meat DataFrame

corr_meat = meat.corr(method='spearman')





# Customize the heatmap of the corr_meat correlation matrix

sns.heatmap(corr_meat,

            annot=True,

            linewidths=0.4,

            annot_kws={"size": 10})



plt.xticks(rotation=90)

plt.yticks(rotation=0) 

plt.show()
# Import seaborn library

import seaborn as sns



# Get correlation matrix of the meat DataFrame

corr_meat = meat.corr(method = 'pearson')



# Customize the heatmap of the corr_meat correlation matrix and rotate the x-axis labels

fig = sns.clustermap(corr_meat,

                     row_cluster=True,

                     col_cluster=True,

                     figsize=(10, 10))



plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()
# Read in jobs file

jobs = pd.read_csv('/kaggle/input/week6dataset/employment.csv', parse_dates = ['datestamp'])



# Review the first five lines of your DataFrame

display(jobs.head(5))



# Review the type of each column in your DataFrame

print(jobs.dtypes)



# Convert datestamp column to a datetime object

jobs['datestamp'] = pd.to_datetime(jobs['datestamp'])



# Set the datestamp columns as the index of your DataFrame

jobs = jobs.set_index('datestamp')



# Check the number of missing values in each column

display(jobs.isnull().sum())
# Generate a boxplot

jobs.boxplot(fontsize=6, vert=False)

plt.show()



# Generate numerical summaries

display(jobs.describe())
# A subset of the jobs DataFrame

jobs_subset = jobs[['Finance', 'Information', 'Manufacturing', 'Construction']]



# Print the first 5 rows of jobs_subset

print(jobs_subset.head())



# Create a facetted graph with 2 rows and 2 columns

ax = jobs_subset.plot(subplots=True,

                      layout=(2,2),

                      sharex=False,

                      sharey=False,

                      linewidth=0.7,

                      fontsize=3,

                      legend=False)



plt.show()
# Plot all time series in the jobs DataFrame

ax = jobs.plot(colormap = 'Spectral', fontsize=6, linewidth=0.8)



# Set labels and legend

ax.set_xlabel('Date', fontsize=10)

ax.set_ylabel('Unemployment Rate', fontsize=10)

ax.set_title('Unemployment rate of U.S. workers by industry', fontsize=10)

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))



# Annotate your plots with vertical lines

ax.axvline('2001-07-01', color='blue', linestyle='--', linewidth=0.8)

ax.axvline('2008-09-01', color='blue', linestyle='--', linewidth=0.8)



# Show plot

plt.show()
# Extract the month from the index of jobs

index_month = jobs.index.month



# Compute the mean unemployment rate for each month

jobs_by_month = jobs.groupby(index_month).mean()



# Plot the mean unemployment rate for each month

ax = jobs_by_month.plot(fontsize=6, linewidth=1)



# Set axis labels and legend

ax.set_xlabel('Month', fontsize=10)

ax.set_ylabel('Mean unemployment rate', fontsize=10)

ax.legend(bbox_to_anchor=(0.8, 0.6), fontsize=10)

plt.show()
# Extract of the year in each date indices of the jobs DataFrame

index_year = jobs.index.year



# Compute the mean unemployment rate for each year

jobs_by_year = jobs.groupby(index_year).mean()



# Plot the mean unemployment rate for each year

ax = jobs_by_year.plot(fontsize=6, linewidth=1)



# Set axis labels and legend

ax.set_xlabel('Year', fontsize=10)

ax.set_ylabel('Mean unemployment rate', fontsize=10)

ax.legend(bbox_to_anchor=(0.1, 0.5), fontsize=10)

plt.show()
# Initialize dictionary

jobs_decomp = {}



# Get the names of each time series in the DataFrame

jobs_names = jobs.columns



# Run time series decomposition on each time series of the DataFrame

for ts in jobs_names:

    ts_decomposition = sm.tsa.seasonal_decompose(jobs[ts])

    jobs_decomp[ts] = ts_decomposition
jobs_seasonal = {}



# Extract the seasonal values for the decomposition of each time series

for ts in jobs_names:

    jobs_seasonal[ts] = jobs_decomp[ts].seasonal

    

# Create a DataFrame from the jobs_seasonal dictionnary

seasonality_df = pd.DataFrame(jobs_seasonal)



# Remove the label for the index

seasonality_df.index.name = None



# Create a faceted plot of the seasonality_df DataFrame

seasonality_df.plot(subplots=True,

                   layout=(4, 4),

                   sharey=False,

                   fontsize=2,

                   linewidth=0.3,

                   legend=False)



# Show plot

plt.show()
# Get correlation matrix of the seasonality_df DataFrame

seasonality_corr = seasonality_df.corr(method='spearman')



# Customize the clustermap of the seasonality_corr correlation matrix

fig = sns.clustermap(seasonality_corr, annot=True, annot_kws={"size": 4}, linewidths=.4, figsize=(15, 10))

plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

plt.show()