import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set()
import os

print(os.listdir("../input/"))
df=pd.read_csv("../input/1.04. Real-life example.csv")

#df=pd.read_csv(r"C:\Users\Rudra\Documents\Python Scripts\Udemy\S35_L226\1.04. Real-life example.csv")

df.head()

df_copy=df
df.describe(include='all')
df=df.drop(columns='Model',axis=1)
df.describe(include='all')
df.isnull().sum()
df=df.dropna(axis=0)
df.isnull().sum()
sns.distplot(np.array(df['Price']))

df['Price'].quantile(0.99)
df=df[df['Price']<(df['Price'].quantile(0.99))]
df.describe(include='all')
sns.distplot(np.array(df['Mileage']))
df=df[df['Mileage']<(df['Mileage'].quantile(0.99))]
sns.distplot(np.array(df['Mileage']))
sns.distplot(np.array(df['EngineV']))
df=df[df['EngineV']<6.5]
sns.distplot(np.array(df['EngineV']))
sns.distplot(np.array(df['Year']))
df=df[df['Year']>(df['Year'].quantile(0.01))]
sns.distplot(np.array(df['Year']))
df.describe(include='all')
f,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))

ax1.scatter(df['Year'],df['Price'])

ax1.set_title('Price vs Year')

ax2.scatter(df['EngineV'],df['Price'])

ax2.set_title('Price vs EngineV')

ax3.scatter(df['Mileage'],df['Price'])

ax3.set_title('Price vs Mileage')

plt.show()
df['log_price']=np.log(df['Price'])
f,(ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))

ax1.scatter(df['Year'],df['log_price'])

ax1.set_title('Price vs Year')

ax2.scatter(df['EngineV'],df['log_price'])

ax2.set_title('Price vs EngineV')

ax3.scatter(df['Mileage'],df['log_price'])

ax3.set_title('Price vs Mileage')

plt.show()
data_cleaned=df
data_cleaned=data_cleaned.drop(['Price'], axis=1)
data_cleaned
data_cleaned.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data_cleaned[['Mileage','Year','EngineV']]
def find_vif(variables):

    vif = pd.DataFrame()

    vif['vif']=[variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]

    vif['feature']=variables.columns

    return vif
find_vif(variables)
data_mul_col_removed=data_cleaned.drop(['Year'],axis=1)
data_with_dummies=pd.get_dummies(data_mul_col_removed,drop_first=True)

data_with_dummies.head()
data_with_dummies.columns
find_vif(data_with_dummies)
y=data_with_dummies['log_price']

x=data_with_dummies.drop(['log_price'],axis=1)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=365)
#linear regression object

reg = LinearRegression()

# Fit the regression with the scaled TRAIN inputs and targets

reg.fit(x_train,y_train)
# checking the outputs of the regression

# storing them in y_hat as this is the 'theoretical' name of the predictions

y_hat = reg.predict(x_train)
# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot

# The closer the points to the 45-degree line, the better the prediction

plt.scatter(y_train, y_hat)

# Let's also name the axes

plt.xlabel('Targets (y_train)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

# Sometimes the plot will have different scales of the x-axis and the y-axis

# This is an issue as we won't be able to interpret the '45-degree line'

# We want the x-axis and the y-axis to be the same

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
# Another useful check of our model is a residual plot

# We can plot the PDF of the residuals and check for anomalies

sns.distplot(y_train - y_hat)



# Include a title

plt.title("Residuals PDF", size=18)



# In the best case scenario this plot should be normally distributed

# In our case we notice that there are many negative residuals (far away from the mean)

# Given the definition of the residuals (y_train - y_hat), negative values imply

# that y_hat (predictions) are much higher than y_train (the targets)

# This is food for thought to improve our model
# Find the R-squared of the model

reg.score(x_train,y_train)



# Note that this is NOT the adjusted R-squared

# in other words... find the Adjusted R-squared to have the appropriate measure :)
# Import the scaling module

from sklearn.preprocessing import StandardScaler



# Create a scaler object

scaler = StandardScaler()

# Fit the inputs (calculate the mean and standard deviation feature-wise)

scaler.fit(x)
x_scaled = scaler.transform(x)
x_scaled
from sklearn.model_selection import train_test_split



# Split the variables with an 80-20 split and some random state

# To have the same split as mine, use random_state = 365

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=365)
# Create a linear regression object

reg = LinearRegression()

# Fit the regression with the scaled TRAIN inputs and targets

reg.fit(x_train,y_train)
# Let's check the outputs of the regression

# I'll store them in y_hat as this is the 'theoretical' name of the predictions

y_hat = reg.predict(x_train)
# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot

# The closer the points to the 45-degree line, the better the prediction

plt.scatter(y_train, y_hat)

# Let's also name the axes

plt.xlabel('Targets (y_train)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

# Sometimes the plot will have different scales of the x-axis and the y-axis

# This is an issue as we won't be able to interpret the '45-degree line'

# We want the x-axis and the y-axis to be the same

plt.xlim(6,13)

plt.ylim(6,13)

plt.show()
# Find the R-squared of the model

reg.score(x_train,y_train)



# Note that this is NOT the adjusted R-squared

# in other words... find the Adjusted R-squared to have the appropriate measure :)
reg.intercept_
reg.coef_