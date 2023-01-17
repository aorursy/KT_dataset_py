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
import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sn

import sklearn.ensemble

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression



# FUNCTIONS #

#############

# Plot multiple histograms - missigness (maximum 3x3=9)

def multiple_hist_missing(columns):

    if(len(columns) > 9):

            print("Maximum number of graphics reached")

            return

    plt.rcParams['figure.figsize'] = (19, 19)

    i=1

    for column in missing_prices_interesting_columns:

        plt.subplot(3,3,i)

        plt.title('Effect of missingness in: ' + column)

        plt.hist(missing_prices[column], alpha=0.7, label=column)

        plt.legend(loc='upper right')

        plt.xticks(rotation=30, ha='right')

        i=i+1

    plt.show()



# Plot multiple histograms - grouped by brand (maximum 3x3=9)

def multiple_hist_grouped(columns):

    if(len(columns) > 9):

            print("Maximum number of graphics reached")

            return

    plt.rcParams['figure.figsize'] = (19, 19)

    i=1

    for column in missing_prices_interesting_columns:

        plt.subplot(3,3,i)

        plt.title('Priced data distribution - per brand: ' + column)

        plt.hist([prices[prices['cab_type'] == 'Uber'][column], prices[prices['cab_type'] == 'Lyft'][column]], alpha=0.5, label=['Uber', 'Lyft'])

        plt.legend(loc='upper right')

        plt.xticks(rotation=30, ha='right')

        i=i+1

    plt.show()



# DATA IMPORTING #

##################

# Import data as dataframe

rides = pd.read_csv("/kaggle/input/data-mining-project-boston/rideshare_data.csv")



# DATA EXPLORATION AND CLEANING #

#################################

# Show variable types

print(rides.info())



# Show column names

print(rides.columns.to_list())



# Show top data sample

print(rides.head())



# Since it has an extra column of indexing it will be droped

rides = rides.drop('Unnamed: 0', axis=1)



# Check if any columns contain missing values and bring them

print(rides.isna().any())



# It seems it has missing values in 'price' column, let's dig a bit on the matter, make a Bar plot of missing values by variable

rides.isna().sum().plot(kind="bar")



# Calculate the percentage of missing data

rate_price_missing_data = (float(rides.price.isna().sum())/float(len(rides['price']))) * 100

print('Total percentage of missing pricing data: %.2f' % (rate_price_missing_data))



# Let's explore more this missing data, missingness matrix

msno.matrix(rides)

plt.title('Exploratory missingness matrix')

plt.show()



# Convert categories to numbers ['source', 'destination', 'cab_type', 'name', 'weekday'] and replace them, 'id' will be dropped later, it could be better to use One-Hot-Encoder

# But let's try this one first

rides['source_encoded'] = LabelEncoder().fit_transform(rides['source'])

rides['destination_encoded'] = LabelEncoder().fit_transform(rides['destination'])

rides['cab_type_encoded'] = LabelEncoder().fit_transform(rides['cab_type'])

rides['name_encoded'] = LabelEncoder().fit_transform(rides['name'])

rides['short_summary_encoded'] = LabelEncoder().fit_transform(rides['short_summary'])

rides['weekday_encoded'] = LabelEncoder().fit_transform(rides['weekday'])



# Let's try to see what categories could be worst affected for this lack of info, spliting the info a bit

# Missing ones rows

missing_prices = rides[rides['price'].isna()]

missing_prices_size = len(missing_prices)



# With prices on it

prices = rides[~rides['price'].isna()]

prices_size = len(prices)



# Let's try to see the missingness impact on 'id'

number_of_unique_id = len(rides['id'].unique())

number_of_unique_id_missing = len(missing_prices['id'].unique())

rate_price_missing_data_id = (float(number_of_unique_id_missing)/float(number_of_unique_id)) * 100

print('Percentage of missing pricing data per id: %.2f' % (rate_price_missing_data_id))



# Since these latest two rates are equal, the id is unique per record and not associated with rider id or user id

# Just in case, let's check for duplicates on id

print('Number of duplicated ids: ' + str(rides.id.duplicated().sum()))
# They are unique id in the whole file!, nevertheless let's move on and try to see the effect of missingness on all other columns

# Plot them

missing_prices_interesting_columns = ['cab_type', 'name', 'month', 'weekday',  'day', 'hour', 'source', 'destination', 'short_summary']

multiple_hist_missing(missing_prices_interesting_columns)



# As per above information, month = [11, 12], cab_type = ['uber'], name = ['taxi'], hours = [0, 1, 6, 7, 8, 14, 15, 16, 21, 22, 23]

# weekday = [ 'Mon', 'Tue'], day = [] are the most affected ones, this is interesting as all these seem peak events with importance for the bussiness (paydays, early morning,

# lunch time, late afternoon, late night, etc.)

# Let's see this time the distribution of these interensting columns with prices on them, just to be double sure before dropping these

interesting_columns = ['name', 'month', 'weekday',  'day', 'hour', 'source', 'destination', 'short_summary']    

multiple_hist_grouped(interesting_columns)



# After later graphs, the data is balanced per brand, we could fill the missings with .meadian() grouped by ['cab_type', 'name', 'month', 'day', 'hour']

# But for this exercise we just can drop the missing values

uber_prices = prices[prices['cab_type'] == 'Uber']

lyft_prices = prices[prices['cab_type'] == 'Lyft']    

    

# Let's see the variation on princing per brand

plt.title('Range of prices per Brand')

plt.hist([uber_prices.price, lyft_prices.price], alpha=0.5, bins=30, label=['Uber','Lyft'])

plt.xlabel('Price in USD')

plt.ylabel('Count')

plt.legend(loc='upper right')

plt.show()



# Scatterplot

# Ramdom sampling as size of dataframes differ

n=30000

plt.title('Scatterplot on price per Brand')

plt.scatter(uber_prices.price.sample(n), lyft_prices.price.sample(n), alpha=0.2)

plt.xlabel('Uber - price')

plt.ylabel('Lyft - price')

plt.show()



# After these latest plots, it seems we could introduce certain noise if we group the data as a whole in a search of a joint model (specially for low costs)

# Even the scatterplot tends to suggest Lyft have higher prices. We will evaluate the three options for modeling: all, uber, lyft and search for an ensembled model later on. 

# In order to seize the effects on 'price' from the other features, let's do a Pearson's correlation study for all data



# Let's drop categorical data as they were already encoded. Drop 'id' too as is unique and not usable to predict anything

rides = rides.drop(['id', 'source', 'destination', 'cab_type', 'name', 'short_summary', 'weekday'], axis=1)

# Synch with the remaining dataframes

prices = prices.drop(['id', 'source', 'destination', 'cab_type', 'name', 'short_summary', 'weekday'], axis=1)

missing_prices = missing_prices.drop(['id', 'source', 'destination', 'cab_type', 'name', 'short_summary', 'weekday'], axis=1)

uber_prices = uber_prices.drop(['id', 'source', 'destination', 'cab_type', 'name', 'short_summary', 'weekday'], axis=1)

lyft_prices = lyft_prices.drop(['id', 'source', 'destination', 'cab_type', 'name', 'short_summary', 'weekday'], axis=1)



# All data

plt.title('Correlation matrix: All data')

sn.heatmap(rides.corr(), annot=False, cmap='RdYlGn')

plt.show()



# Uber's

plt.title('Correlation matrix: Uber data')

sn.heatmap(uber_prices.corr(), annot=False, cmap='RdYlGn')

plt.show()



# Lyft's

plt.title('Correlation matrix: Lyft data')

sn.heatmap(lyft_prices.corr(), annot=False, cmap='RdYlGn')

plt.show()
# FEATURE TREATMENT #

#####################

# Let's perform feature scaling, let's separate the data from the target attributes

# All data

X_prices = prices.drop('price', axis=1)

y_prices = prices.price

# Uber's

X_uber_prices = uber_prices.drop('price', axis=1)

y_uber_prices = uber_prices.price

# Lyft's

X_lyft_prices = lyft_prices.drop('price', axis=1)

y_lyft_prices = lyft_prices.price



# normalize the data attributes

# All data

X_prices_norm = preprocessing.normalize(X_prices)

# Uber's

X_uber_prices_norm = preprocessing.normalize(X_uber_prices)

# Lyft's

X_lyft_prices_norm = preprocessing.normalize(X_lyft_prices)



# ML MODELING #

###############

# Create the training and testing sets for all cases

X_prices_train, X_prices_test, y_prices_train, y_prices_test = train_test_split(X_prices_norm, y_prices, test_size=0.30, random_state=0)

X_uber_prices_train, X_uber_prices_test, y_uber_prices_train, y_uber_prices_test = train_test_split(X_uber_prices_norm, y_uber_prices, test_size=0.30, random_state=0)

X_lyft_prices_train, X_lyft_prices_test, y_lyft_prices_train, y_lyft_prices_test = train_test_split(X_lyft_prices_norm, y_lyft_prices, test_size=0.30, random_state=0)



# Fit a logistic regression model for all cases for our data

model_all = LinearRegression()

model_uber = LinearRegression()

model_lyft = LinearRegression()



# Fit them accordingly

model_all.fit(X_prices_train, y_prices_train)

model_uber.fit(X_uber_prices_train, y_uber_prices_train)

model_lyft.fit(X_lyft_prices_train, y_lyft_prices_train)



# Obtain model predictions

predicted_all = model_all.predict(X_prices_test)

predicted_uber = model_uber.predict(X_uber_prices_test)

predicted_lyft = model_lyft.predict(X_lyft_prices_test)



# PRICING PREDICTIONS COMPARATIONS (Uber vs Lyft vs Combined) #

###############################################################

# Print the classifcation report and confusion matrix

# All

print('Metric MSE (All data):\n', mean_squared_error(y_prices_test, predicted_all))

# Uber's

print('Metric MSE (Uber data):\n', mean_squared_error(y_uber_prices_test, predicted_uber))

# Lyft's

print('Metric MSE (Lyft data):\n', mean_squared_error(y_lyft_prices_test, predicted_lyft))



# FINAL MODEL FOR BOSTON, MA (Uber, Lyft, Combined) #

#####################################################

# According to the results, the best is to have separated models for each brand. 

# It could be improved more using ensemble, or parameter adjusting, and others but for now, this would be it. IJIT.