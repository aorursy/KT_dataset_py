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
# Acquire

df = pd.read_csv("../input/raw_lemonade_data.csv")
# Peek at the first rows of data

df.head()
# Prepare

df["Date"] = pd.to_datetime(df["Date"]) # setup the right data-type for the date column

df["Price"] = df.Price.str.replace("$", "").replace(" ", "") # remove 

df["Price"] = df.Price.astype(np.float64) # setup appropriate numeric datatype for price column

df = df.set_index(df['Date']) # Set the date as the index 

df = df.drop("Date", 1) # drop the old Date column

df["Revenue"] = df.Price * df.Sales # Calculate revenue from sales(units) times price.



df = df[["Revenue", "Temperature", "Rainfall", "Flyers"]] # Only focus on the (dependent) target variable and the independent variables that make sense to model

df.head()
X = df[["Flyers"]] # Independent variable we're using to predict the target

y = df[["Revenue"]] # Dependend variable (target variable) 
# Create the linear model

lm = LinearRegression().fit(X, y) # "to fit" a line means to match the line as close as possible to the datapoints on the X and Y axes.
print("Intercept:", lm.intercept_[0])

print("We'll sell about $4.18 in revenue at the minimum level of flyers")
print("Coefficient for flyers:", lm.coef_[0][0])

print("Linear model coefficients are the weights of the model")

print("This is the slope of the regression prediction line that best fits the datapoints!")
# y = m*x + b

intercept = 4.18 #$

m = .21059

number_of_fliers = 100
revenue = m*number_of_fliers + intercept

print(f'If our only model is fliers to revenue, the for {number_of_fliers}, we earn ${revenue}')
y_predicted = lm.predict(X)



flyer_r2 = r2_score(y, y_predicted)

print("Variance of revenue as explained by the number of flyers: %" + str(flyer_r2 * 100))
# Mean absolute error is the average vertical distance between the actual and expected values from this model

mae = mean_absolute_error(y, y_predicted)

print("Average vertical distance between predicted and actual revenue is $" + str(mae))
X = df[["Rainfall"]] # Independent variable we're using to predict the target

y = df[["Revenue"]] # Dependend variable (target variable) 
# Create the linear model

lm = LinearRegression().fit(X, y) # "to fit" a line means to match the line as close as possible to the datapoints on the X and Y axes.
print("Intercept:", lm.intercept_[0])

print("We'll sell about $22 in revenue when we have 0 rainfall")
# Remember, the coefficient is the slope of the linear model

print("Coefficient for rainfall:", lm.coef_[0][0])

print("in dollar terms, this means that for each additional inch of rain, we lose $" + str(lm.coef_[0][0]))
y_predicted = lm.predict(X)



r2 = r2_score(y, y_predicted)

print("Variance of revenue as explained by the number of rainfall: %" + str(r2 * 100))
# Mean absolute error is the average vertical distance between the actual and expected values from this model

mae = mean_absolute_error(y, y_predicted)

print("Average vertical distance between actual and predicted revenue values is:", mae)
print("For any $ Revenue prediction based on rainfall, our models are off plus or minus $" + str(mae))
# Use the above two examples as a template for creating your single variable linear regression between temperature and revenue