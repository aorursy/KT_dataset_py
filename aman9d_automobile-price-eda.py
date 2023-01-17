import os
os.getcwd()
your_local_path = os.getcwd()
import pandas as pd               
import numpy as np
import pickle

from sklearn.model_selection import train_test_split   #splitting data

from sklearn.linear_model import LinearRegression         #linear regression
from sklearn.metrics.regression import mean_squared_error #error metrics
from sklearn.metrics import mean_absolute_error

import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

%matplotlib inline     
sns.set(color_codes=True)
automobile_data = pd.read_csv('../input/Automobile_data.csv')
automobile_data.head()                               # check the head of the data
automobile_data.describe()                        # summary statistucs
automobile_data.info()                         
automobile_data.replace('?', np.nan, inplace=True)
# count the number of NaN values in each column
print(automobile_data.isnull().sum())
## remove columns that are 90% empty


thresh = len(automobile_data) * .1
automobile_data.dropna(thresh = thresh, axis = 1, inplace = True)
# count the number of NaN values in each column
print(automobile_data.isnull().sum())
## Define a function impute_median
def impute_median(series):
    return series.fillna(series.median())

#automobile_data['num-of-doors']=automobile_data['num-of-doors'].transform(impute_median)
automobile_data.bore=automobile_data['bore'].transform(impute_median)
automobile_data.stroke=automobile_data['stroke'].transform(impute_median)
automobile_data.horsepower=automobile_data['horsepower'].transform(impute_median)
automobile_data.price=automobile_data['price'].transform(impute_median)

automobile_data['num-of-doors'].fillna(str(automobile_data['num-of-doors'].mode().values[0]),inplace=True)
automobile_data['peak-rpm'].fillna(str(automobile_data['peak-rpm'].mode().values[0]),inplace=True)
automobile_data['normalized-losses'].fillna(str(automobile_data['normalized-losses'].mode().values[0]),inplace=True)
# count the number of NaN values in each column
print(automobile_data.isnull().sum())
automobile_data.head()
automobile_data.make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))
plt.title("Number of vehicles by make")
plt.ylabel('Number of vehicles')
plt.xlabel('Make');
#histogram
automobile_data['price']=pd.to_numeric(automobile_data['price'],errors='coerce')
sns.distplot(automobile_data['price']);
#skewness and kurtosis
print("Skewness: %f" % automobile_data['price'].skew())
print("Kurtosis: %f" % automobile_data['price'].kurt())
plt.figure(figsize=(20,10))
c=automobile_data.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
sns.lmplot('engine-size', # Horizontal axis
           'price', # Vertical axis
           data=automobile_data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="make", # Set color
           palette="Paired",
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size
sns.lmplot('city-mpg', # Horizontal axis
           'price', # Vertical axis
           data=automobile_data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="body-style", # Set color
           palette="Paired",
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size
sns.boxplot(x="fuel-type", y="price",data = automobile_data)