# Let's bring in our libraries



# ignore warnings

import warnings

warnings.filterwarnings("ignore")



# Wrangling

import numpy as np

import pandas as pd



# Exploring

import scipy.stats as stats



# Visualizing

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('classic')



# Modeling

import statsmodels.api as sm



from scipy.stats import pearsonr



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error



import eli5
# Acquire

df = pd.read_csv("../input/raw_lemonade_data.csv")
# Prepare

df["Date"] = pd.to_datetime(df["Date"]) # setup the right data-type for the date column

df["Price"] = df.Price.str.replace("$", "").replace(" ", "") # remove 

df["Price"] = df.Price.astype(np.float64) # setup appropriate numeric datatype for price column

df = df.set_index(df['Date']) # Set the date as the index 

df = df.drop("Date", 1) # drop the old Date column

df["Revenue"] = df.Price * df.Sales # Calculate revenue from sales(units) times price.



df = df[["Revenue", "Temperature", "Rainfall", "Flyers"]]
X = df[["Temperature", "Rainfall", "Flyers"]]

y = df[["Revenue"]] # Dependend variable (target variable) 
lm = LinearRegression().fit(X, y)
coefficients = dict(zip(X.columns, lm.coef_[0])) # [0] notation is here because of how "y" was setup with a matrix not a series

coefficients = pd.Series(coefficients)

print(coefficients)
offsets = dict(zip(X.columns, lm.intercept_)) # [0] notation is here because of how "y" was setup with a matrix not a series

offsets
eli5.show_weights(lm)