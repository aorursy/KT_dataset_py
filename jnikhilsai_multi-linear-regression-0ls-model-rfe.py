import pandas as pd
# Importing advertising.csv

advertising_multi = pd.read_csv('../input/advertising-mul/advertising.csv')
# Looking at the first five rows

advertising_multi.head()
# Looking at the last five rows

advertising_multi.tail()
# What type of values are stored in the columns?

advertising_multi.info()
# Let's look at some statistical information about our dataframe.

advertising_multi.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's plot a pair plot of all variables in our dataframe

sns.pairplot(advertising_multi)
# Visualise the relationship between the features and the response using scatterplots

sns.pairplot(advertising_multi, x_vars=['TV','Radio','Newspaper'], y_vars='Sales',size=7, aspect=0.7, kind='scatter')
# Putting feature variable to X

X = advertising_multi[['TV','Radio','Newspaper']]



# Putting response variable to y

y = advertising_multi['Sales']
#random_state is the seed used by the random number generator. It can be any integer.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)
from sklearn.linear_model import LinearRegression
# Representing LinearRegression as lr(Creating LinearRegression Object)

lm = LinearRegression()
# fit the model to the training data

lm.fit(X_train,y_train)
# print the intercept

print(lm.intercept_)
# Let's see the coefficient

coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])

coeff_df
# Making predictions using the model

y_pred = lm.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
import statsmodels.api as sm

X_train_sm = X_train

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X_train_sm = sm.add_constant(X_train_sm)

# create a fitted model in one line

lm_1 = sm.OLS(y_train,X_train_sm).fit()



# print the coefficients

lm_1.params
print(lm_1.summary())
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize = (5,5))

sns.heatmap(advertising_multi.corr(),annot = True)
# Removing Newspaper from our dataset

X_train_new = X_train[['TV','Radio']]

X_test_new = X_test[['TV','Radio']]
# Model building

lm.fit(X_train_new,y_train)
# Making predictions

y_pred_new = lm.predict(X_test_new)
#Actual vs Predicted

c = [i for i in range(1,61,1)]

fig = plt.figure()

plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")

plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Sales', fontsize=16)                               # Y-label
# Error terms

c = [i for i in range(1,61,1)]

fig = plt.figure()

plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred_new)

r_squared = r2_score(y_test, y_pred_new)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
X_train_final = X_train_new

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X_train_final = sm.add_constant(X_train_final)

# create a fitted model in one line

lm_final = sm.OLS(y_train,X_train_final).fit()



print(lm_final.summary())
from sklearn.feature_selection import RFE
rfe = RFE(lm, 2)
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)

print(rfe.ranking_)
import pandas as pd

import numpy as np

# Importing dataset

advertising_multi = pd.read_csv('../input/advertising-mul/advertising.csv')



x_news = advertising_multi['Newspaper']



y_news = advertising_multi['Sales']



# Data Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_news, y_news, 

                                                    train_size=0.7 , 

                                                    random_state=110)



# Required only in the case of simple linear regression

X_train = X_train[:,np.newaxis]

X_test = X_test[:,np.newaxis]



# Linear regression from sklearn

from sklearn.linear_model import LinearRegression

lm = LinearRegression()



# Fitting the model

lm.fit(X_train,y_train)



# Making predictions

y_pred = lm.predict(X_test)



# Importing mean square error and r square from sklearn library.

from sklearn.metrics import mean_squared_error, r2_score



# Computing mean square error and R square value

mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)



# Printing mean square error and R square value

print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)