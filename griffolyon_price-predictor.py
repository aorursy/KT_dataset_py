import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing Housing.csv

housing = pd.read_csv("../input/mydatasets/Housing.csv")
# Looking at the first five rows

housing.head()
# Converting Yes to 1 and No to 0

housing['mainroad'] = housing['mainroad'].map({'yes': 1, 'no': 0})

housing['guestroom'] = housing['guestroom'].map({'yes': 1, 'no': 0})

housing['basement'] = housing['basement'].map({'yes': 1, 'no': 0})

housing['hotwaterheating'] = housing['hotwaterheating'].map({'yes': 1, 'no': 0})

housing['airconditioning'] = housing['airconditioning'].map({'yes': 1, 'no': 0})

housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})
# Creating dummy variable for variable furnishingstatus and dropping the first one

status = pd.get_dummies(housing['furnishingstatus'],drop_first=True)
#Adding the results to the master dataframe

housing = pd.concat([housing,status],axis=1)
# Dropping the variable 'furnishingstatus'

housing.drop(['furnishingstatus'],axis=1,inplace=True)
# Let us create the new metric and assign it to "areaperbedroom"

housing['areaperbedroom'] = housing['area']/housing['bedrooms']
# Metric: bathrooms per bedroom

housing['bbratio'] = housing['bathrooms']/housing['bedrooms']
#defining a normalisation function 

def normalize (x): 

    return ( (x-np.min(x))/ (max(x) - min(x)))

                                            

                                              

# applying normalize ( ) to all columns 

housing = housing.apply(normalize)
# Putting feature variable to X

X = housing[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',

       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',

       'parking', 'prefarea', 'semi-furnished', 'unfurnished',

       'areaperbedroom', 'bbratio']]



# Putting response variable to y

y = housing['price']
housing.plot.line(x='price', y='area')
#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)
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
plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(housing.corr(),annot = True)
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
# Calculating Vif value

vif_cal(input_data=housing.drop(['area','bedrooms','stories','basement','semi-furnished','areaperbedroom'], axis=1), dependent_col="price")
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

import matplotlib.pyplot as plt

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

plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")

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

sns.distplot((y_test-y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label
# Now let's check the Root Mean Square Error of our model.

import numpy as np

from sklearn import metrics

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

