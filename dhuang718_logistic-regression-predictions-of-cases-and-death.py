import pandas as pd

import numpy as np

from datetime import datetime as dt

from matplotlib import pyplot as plt

import re

import seaborn as sns

from array import *

from sklearn import linear_model

from scipy.special import expit
plt.figure(figsize=(7,7))



# General a toy dataset:s it's just a straight line with some Gaussian noise:

xmin, xmax = -5, 5

n_samples = 100

np.random.seed(0)

X = np.random.normal(size=n_samples)

y = (X > 0).astype(np.float)

X[X > 0] *= 4

X += 1 * np.random.normal(size=n_samples)



# Fit the classifier

X = X[:, np.newaxis]

clf = linear_model.LogisticRegression(C=1e5)

clf.fit(X, y)



# and plot the result

plt.figure(1, figsize=(4, 3))

plt.clf()

plt.scatter(X.ravel(), y, color='black', zorder=20)

X_test = np.linspace(-5, 10, 300)



loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()

plt.plot(X_test, loss, color='red', linewidth=3)



plt.axhline(.5, color='.5')



plt.ylabel('y')

plt.xlabel('X')

plt.xticks(range(-5, 10))

plt.yticks([0, 0.5, 1])

plt.ylim(-.25, 1.25)

plt.xlim(-4, 5)

plt.tight_layout()

plt.show()
#import US data

df_us = pd.read_csv('/kaggle/input/covid19-in-usa/us_states_covid19_daily.csv')



#Create a column in days from first outbreak

df_us_dt = pd.to_datetime(df_us.date, format='%Y%m%d')

us_from_first_case = df_us_dt - min(df_us_dt)

df_us['from_first_case']= us_from_first_case



#Choose a state

df_ny = df_us.loc[df_us['state'] == 'NY']



#Sort US data ascending from_first_case

#Reset the index for future use

df_ny = df_ny.sort_values(['from_first_case'], ascending = True)

df_ny = df_ny.reset_index(drop=True)



df_ny['Growth_factor'] = float(0)

for i in range(len(df_ny.Growth_factor.tolist())):

    if i <= 1:

        df_ny.at[i,'Growth_factor'] = 0

    else:

        df_ny.at[i,'Growth_factor'] = df_ny.positiveIncrease[i]/df_ny.positiveIncrease[i-1]

df_ny
#Plot the growth factor and see the half time of the pandemic in each location

plt.plot(df_ny.from_first_case, df_ny.Growth_factor)

plt.xlabel('Days After Outbreak (100 Cases)')

plt.ylabel('Growth Factor')

plt.title('NY Growth Factor')

plt.show()
#Replace the zero and inf growth factors with average values 2 previous and 2 in the future

df_ny.at[8, 'Growth_factor'] = (df_ny.at[6, 'Growth_factor'] + df_ny.at[7, 'Growth_factor'] + df_ny.at[10, 'Growth_factor'] + df_ny.at[11, 'Growth_factor'])/4 

df_ny.at[9, 'Growth_factor'] = df_ny.at[8, 'Growth_factor']
#Creates regression model to predict the next days

#Note: from_first_cases needs to be converted to an integer column for sytnax issues



def regression_model(num_next_days, degrees, y):

    #create x and column, remove first 2 index values

    global x

    x = df_ny.iloc[2:, -2:-1].values 

    x = (x / np.timedelta64(1, 'D')).astype(int)

    X = x[:, np.newaxis]



    #creates the 2D array for number of days prediction

    a = np.zeros([num_next_days,1]).tolist()

    for i in range(len(a)):

        a[i]= [df_ny.last_valid_index()+1+i]

    

    # Fitting Linear Regression to the dataset 

    from sklearn.linear_model import LinearRegression 

    lin = LinearRegression() 

  

    lin.fit(x, y)



    #Fitting polynomial regression to the dataset

    from sklearn.preprocessing import PolynomialFeatures 

  

    global poly

    poly = PolynomialFeatures(degree = degrees) 

    x_poly = poly.fit_transform(x) 

  

    global lin2

    poly.fit(x_poly, y) 

    lin2 = LinearRegression() 

    lin2.fit(x_poly, y) 



    # Visualising the Polynomial Regression results 

    plt.scatter(x, y, color = 'blue') 

      

    plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red') 

    plt.title('Polynomial Regression') 

    plt.xlabel('Days from Outbreak (100 Days)') 

    plt.ylabel('Growth Factor') 

    

    #create dataframe for predictions

    #need to create a previous known positive column to start the first prediction since each prediction is based on the last

    reg_ny = {'Growth_factor': lin2.predict(poly.fit_transform(a)).tolist(), 

              'Prev_positiveIncrease': np.array([0] * num_next_days),

              'Pred_positiveIncrease': np.array([0] * num_next_days)}

    df_pred_ny = pd.DataFrame(reg_ny, columns = ['Growth_factor','Prev_positiveIncrease','Pred_positiveIncrease'])

    df_pred_ny.at[0, 'Prev_positiveIncrease'] = df_ny.positiveIncrease.iat[-1] 



    #Fill in the predicted_newpositive, 

    #index 0 is based on last known positiveIncrease

    #index 1 is based on predicted last positiveIncrease

    for i in range(len(df_pred_ny.Pred_positiveIncrease.tolist())):

        if i == 0:

            df_pred_ny.at[i,'Pred_positiveIncrease'] = df_pred_ny.Growth_factor[i]*df_pred_ny.Prev_positiveIncrease[0]

        else:

            df_pred_ny.at[i,'Pred_positiveIncrease'] = df_pred_ny.Growth_factor[i]*df_pred_ny.Pred_positiveIncrease[i-1]



    #fill in the predicted confirmed column        

    df_pred_ny['Pred_Confirmed'] = 0

    for i in range(len(df_pred_ny.Pred_Confirmed.tolist())):

        if i == 0:

            df_pred_ny.at[i,'Pred_Confirmed'] = df_pred_ny.Pred_positiveIncrease[i] + df_ny.positive.iat[-1]

        else:

            df_pred_ny.at[i,'Pred_Confirmed'] = df_pred_ny.Pred_positiveIncrease[i] + df_pred_ny.Pred_Confirmed[i-1]





    #Outputs predicted growth factor values and the graph 

    print('Regression Degrees:', degrees)

    print(df_pred_ny[['Growth_factor', 'Pred_positiveIncrease','Pred_Confirmed']])

    plt.plot(a, lin2.predict(poly.fit_transform(a)), 'ro')

    plt.show()

    

regression_model(num_next_days = 8, degrees = 2,y = df_ny.iloc[2:, -1].values)
cleansed_reg = {'Growth_factor_regression': lin2.predict(poly.fit_transform(x)).tolist()}

cleansed_reg = pd.DataFrame(cleansed_reg, columns = ['Growth_factor_regression'])

cleansed_reg['Growth_factor'] = df_ny.iloc[2:, -1].reset_index(drop=True)

cleansed_reg['Growth_factor_cleansed'] = 0

for i in range(len(cleansed_reg['Growth_factor_cleansed'])):

    if cleansed_reg.Growth_factor[i] > 1.4*cleansed_reg.Growth_factor_regression[i]:

        cleansed_reg.loc[i, 'Growth_factor_cleansed'] = (cleansed_reg.loc[i-1, 'Growth_factor']+cleansed_reg.loc[i+1, 'Growth_factor'])/2

    else:

        cleansed_reg.loc[i, 'Growth_factor_cleansed'] = cleansed_reg.loc[i, 'Growth_factor']
regression_model(num_next_days = 14, degrees = 2, y = cleansed_reg.iloc[:, -1].values)
#Using the logistic curve model:

expected_confirm = int(2 *df_ny.iloc[-1, 2])

death_rate = df_ny.iloc[-1, 14]/df_ny.iloc[-1, 2]

expected_death = int(death_rate*expected_confirm)

print('Logistic Model')

print('Predicted Positive Cases:', expected_confirm, '\nPredicted Deaths:', expected_death)

#Using the regression curve model

print('Regression Model')

print('Predicted Positive Cases:', '142669', '\nPredicted Deaths:', int(142669*death_rate))
print('Current Patients in ICU', int(df_ny.iloc[-1, 7]))