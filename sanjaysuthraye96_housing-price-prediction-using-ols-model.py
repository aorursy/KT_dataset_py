import pandas as pd

import numpy as np
# Importing Housing.csv

housing = pd.read_csv('../input/housing-simple-regression/Housing.csv')
# Looking at the first five rows

housing.head()
# What type of values are stored in the columns?

housing.info()
# Converting Yes to 1 and No to 0

housing['mainroad'] = housing['mainroad'].map({'yes': 1, 'no': 0})

housing['guestroom'] = housing['guestroom'].map({'yes': 1, 'no': 0})

housing['basement'] = housing['basement'].map({'yes': 1, 'no': 0})

housing['hotwaterheating'] = housing['hotwaterheating'].map({'yes': 1, 'no': 0})

housing['airconditioning'] = housing['airconditioning'].map({'yes': 1, 'no': 0})

housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})
# Now let's see the head

housing.head()
# Creating a dummy variable for 'furnishingstatus'

status = pd.get_dummies(housing['furnishingstatus'])
# The result has created three variables that are not needed.

status.head()
# we don't need 3 columns. Because any one category should be 1, so only 2 is enough

# we can use drop_first = True to drop the first column from status df.

status = pd.get_dummies(housing['furnishingstatus'], drop_first=True)
status
#Adding the results to the master dataframe

housing = pd.concat([housing,status],axis=1)
# Now let's see the head of our dataframe.

housing.head()
# Dropping furnishingstatus as we have created the dummies for it

housing.drop(['furnishingstatus'],axis=1,inplace=True)
# Now let's see the head of our dataframe.

housing.head()
# Let us create the new metric and assign it to "areaperbedroom"

housing['areaperbedroom'] = housing['area']/housing['bedrooms']
# Metric:bathrooms per bedroom

housing['bbratio'] = housing['bathrooms']/housing['bedrooms']
housing.head()
#defining a normalisation function 

def normalize (x): 

    return ( (x-np.min(x))/ (max(x) - min(x)))

                                            

                                              

# applying normalize ( ) to all columns 

housing = housing.apply(normalize) 
housing.head(5)
housing.columns
# Putting feature variable to X

X = housing[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',

       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',

       'parking', 'prefarea', 'semi-furnished', 'unfurnished',

       'areaperbedroom', 'bbratio']]



# Putting response variable to y

y = housing['price']
#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)
import statsmodels.api as sm          # Importing statsmodels

X_train = sm.add_constant(X_train)    # Adding a constant column to our dataframe

# create a first fitted model

lm_1 = sm.OLS(y_train,X_train).fit()
#Let's see the summary of our first linear model

print(lm_1.summary())


# UDF for calculating vif value

def vif_cal(input_data, dependent_col):

    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])

    x_vars=input_data.drop([dependent_col], axis=1)

    xvar_names=x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.OLS(y,x).fit().rsquared  

        vif=round(1/(1-rsq),2)

        vif_df.loc[i] = [xvar_names[i], vif]

    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)
# Calculating Vif value|

vif_cal(input_data=housing, dependent_col="price")
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's see the correlation matrix 

plt.figure(figsize = (16,10))     # Size of the figure

sns.heatmap(housing.corr(),annot = True)
# Dropping highly correlated variables and insignificant variables

X_train = X_train.drop('bbratio', 1)
# Create a second fitted model

lm_2 = sm.OLS(y_train,X_train).fit()
#Let's see the summary of our second linear model

print(lm_2.summary())
# Calculating Vif value

vif_cal(input_data=housing.drop(["bbratio"], axis=1), dependent_col="price")
# Dropping highly correlated variables and insignificant variables

X_train = X_train.drop('bedrooms', 1)
# Create a third fitted model

lm_3 = sm.OLS(y_train,X_train).fit()
#Let's see the summary of our third linear model

print(lm_3.summary())
# Calculating Vif value

vif_cal(input_data=housing.drop(["bedrooms","bbratio"], axis=1), dependent_col="price")
# # Dropping highly correlated variables and insignificant variables

X_train = X_train.drop('areaperbedroom', 1)
# Create a fourth fitted model

lm_4 = sm.OLS(y_train,X_train).fit()
#Let's see the summary of our fourth linear model

print(lm_4.summary())
# Calculating Vif value

vif_cal(input_data=housing.drop(["bedrooms","bbratio","areaperbedroom"], axis=1), dependent_col="price")
# # Dropping highly correlated variables and insignificant variables

X_train = X_train.drop('semi-furnished', 1)
# Create a fifth fitted model

lm_5 = sm.OLS(y_train,X_train).fit()
#Let's see the summary of our fifth linear model

print(lm_5.summary())
# Calculating Vif value

vif_cal(input_data=housing.drop(["bedrooms","bbratio","areaperbedroom","semi-furnished"], axis=1), dependent_col="price")
# # Dropping highly correlated variables and insignificant variables

X_train = X_train.drop('basement', 1)
# Create a sixth fitted model

lm_6 = sm.OLS(y_train,X_train).fit()
#Let's see the summary of our sixth linear model

print(lm_6.summary())
# Calculating Vif value

vif_cal(input_data=housing.drop(["bedrooms","bbratio","areaperbedroom","semi-furnished","basement"], axis=1), dependent_col="price")
# Adding  constant variable to test dataframe

X_test_m6 = sm.add_constant(X_test)
# Creating X_test_m6 dataframe by dropping variables from X_test_m6

X_test_m6 = X_test_m6.drop(["bedrooms","bbratio","areaperbedroom","semi-furnished","basement"], axis=1)
# Making predictions

y_pred_m6 = lm_6.predict(X_test_m6)
# Actual vs Predicted

c = [i for i in range(1,165,1)]

fig = plt.figure()

plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")     #Plotting Actual

plt.plot(c,y_pred_m6, color="red",  linewidth=2.5, linestyle="-")  #Plotting predicted

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Housing Price', fontsize=16)                       # Y-label
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred_m6)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
# Error terms

fig = plt.figure()

c = [i for i in range(1,165,1)]

plt.plot(c,y_test-y_pred_m6, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
# Plotting the error terms to understand the distribution.

fig = plt.figure()

sns.distplot((y_test-y_pred_m6),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label
import numpy as np

from sklearn import metrics

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m6)))