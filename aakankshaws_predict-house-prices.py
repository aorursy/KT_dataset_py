import numpy as np 

import pandas as pd 



import warnings

warnings.filterwarnings



import os

print(os.listdir("../input"))

data = pd.read_csv('../input/Housing.csv')

data.head()
data.describe()
#Look for more info on data set 

data.info()
#first get all features with data type as object

data.columns[data.dtypes == object]
#Convert all nominal data to numerical data

data['mainroad'] = data['mainroad'].map({'yes':1,'no':0})

data['guestroom'] = data['guestroom'].map({'yes':1,'no':0})

data['basement'] = data['basement'].map({'yes':1,'no':0})

data['hotwaterheating'] = data['hotwaterheating'].map({'yes':1,'no':0})

data['airconditioning'] = data['airconditioning'].map({'yes':1,'no':0})

data['prefarea'] = data['prefarea'].map({'yes':1,'no':0})
data.head()
#check furnishingstatus values

data.furnishingstatus.value_counts()
## Creating a dummy variable for 'furnishingstatus'

furnishingstatus = pd.get_dummies(data['furnishingstatus'])

furnishingstatus.head()
# we don't need 3 columns.

# we can use drop_first = True to drop the first column from furnishingstatus df.

furnishingstatus = pd.get_dummies(data['furnishingstatus'],drop_first=True)
#Adding the results to the master dataframe

data = pd.concat([data,furnishingstatus],axis=1)
# Now let's see the head of our dataframe.

data.head()
# Dropping furnishingstatus as we have created the dummies for it

data.drop(['furnishingstatus'],axis=1,inplace=True)



# Now let's see the head of our dataframe.

data.head()
# Let us create the new metric and assign it to "areaperbedroom"

data['areaperbedroom'] = data['area']/data['bedrooms']

# Metric:bathrooms per bedroom

data['bbratio'] = data['bathrooms']/data['bedrooms']

data.head()
def normalize (x): 

    return ( (x-np.min(x))/ (max(x) - min(x)))

                                            

                                              

# applying normalize ( ) to all columns 

data = data.apply(normalize) 
# Putting feature variable to X

X = data[data.columns[1:]]

# Putting response variable to y

y = data[data.columns[:1]]



#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 9

lm = LinearRegression()

rfe = RFE(lm, 9)             # running RFE

rfe = rfe.fit(X_train, y_train)

print(rfe.support_)           # Printing the boolean results

print(rfe.ranking_)  
col = X_train.columns[rfe.support_]

col
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.tools.tools as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
import statsmodels.regression.linear_model as sm

lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor



X = data.drop(['price','area','bedrooms','stories','basement','semi-furnished','areaperbedroom'], axis=1)

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns
vif.sort_values(by='VIF Factor', ascending=False)
# Now let's use our model to make predictions.



# Creating X_test_6 dataframe by dropping variables from X_test

X_test_rfe = X_test[col]



# Adding a constant variable 

X_test_rfe = sm.add_constant(X_test_rfe)



# Making predictions

y_pred = lm.predict(X_test_rfe)
# Now let's check how well our model is able to make predictions.



# Importing the required libraries for plots.

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Actual and Predicted

c = [i for i in range(1,165,1)] # generating index 

fig = plt.figure() 

plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Housing Price', fontsize=16)                       # Y-label
# Error terms

c = [i for i in range(1,165,1)]

fig = plt.figure()

plt.plot(c,y_test.price - y_pred, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
# Plotting the error terms to understand the distribution.

fig = plt.figure()

sns.distplot((y_test.price-y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label
# Now let's check the Root Mean Square Error of our model.

import numpy as np

from sklearn import metrics

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))