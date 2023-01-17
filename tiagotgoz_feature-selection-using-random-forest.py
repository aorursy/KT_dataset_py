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

from math import sqrt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error,accuracy_score,r2_score,mean_absolute_error

from sklearn.model_selection import TimeSeriesSplit



# Setting seed for reproducibility

np.random.seed(1234)  

PYTHONHASHSEED = 0
##################################

# Data Preprocessing

# Validation method considering that the degradation occurs at a certain moment onwards. Based on: https://ieeexplore.ieee.org/abstract/document/7998311/



##################################

MAXLIFE = 15



def Piecewise(cycle_list, max_cycle):

    '''

    Piecewise linear function with zero gradient and unit gradient

            ^

            |

    MAXLIFE |-----------

            |            \

            |             \

            |              \

            |               \

            |                \

            |----------------------->

    '''

    #print(max(cycle_list))

    knee_point = max_cycle - MAXLIFE

    Piecewise_RUL = []

    stable_life = MAXLIFE

    for i in range(0, len(cycle_list)):

        if i < knee_point:

            Piecewise_RUL.append(MAXLIFE)

        else:

            tmp = Piecewise_RUL[i - 1] - (stable_life / (max_cycle - knee_point))

            Piecewise_RUL.append(tmp)



    return Piecewise_RUL



def Apply_Piecewise(lf_dataset):    

    lf_piecewise = []        

    for id in lf_dataset['machineID'].unique():

        for i in lf_dataset.loc[(lf_dataset['machineID']==id)

                              & (lf_dataset['failed'] == 1)].index:

            tam = int(lf_dataset[lf_dataset.index==i]['RUL'])

            i_ini = i - tam + 1

            lf_piecewise.extend(Piecewise(lf_dataset.loc[(lf_dataset.index>=i_ini)&(lf_dataset.index<=i)]['RUL'].values,tam))

    return lf_piecewise  
data = pd.read_csv('/kaggle/input/predictive-useful-life-based-into-telemetry/ALLtrainMescla5D.csv')

data_test = pd.read_csv('/kaggle/input/predictive-useful-life-based-into-telemetry/ALLtestMescla5D.csv')

#######

# TRAIN

#######

mapping = {'model1': 1, 'model2': 2, 'model3': 3, 'model4': 4}

data = data.replace({'model': mapping})

mapping = {'none': 0, 'comp1': 1, 'comp2': 2, 'comp3': 3, 'comp4': 4}

data = data.replace({'failure': mapping})

data = data.drop('datetime',axis=1)



data['time_in_cycles'] = data['RUL'] 

data['RULWise'] = Apply_Piecewise(data)

data = data.astype(np.float)



n_features = 31

n_target = 35

target = 'RUL_I'



feature_list = [data.columns[i] for i in range(0,n_features)]



X_train = data.iloc[:, 0:n_features].values.astype(np.float)

# generate labels

Y_train = data.iloc[:, n_target-1:n_target].values.astype(np.float).ravel()



#######

# TEST

#######

mapping = {'model1': 1, 'model2': 2, 'model3': 3, 'model4': 4}

data_test = data_test.replace({'model': mapping})

mapping = {'none': 0, 'comp1': 1, 'comp2': 2, 'comp3': 3, 'comp4': 4}

data_test = data_test.replace({'failure': mapping})

data_test = data_test.drop('datetime',axis=1)

data_test['time_in_cycles'] = data_test['RUL'] #* -1

data_test['RULWise'] = Apply_Piecewise(data_test)

data_test = data_test.astype(np.float)



X_test = data_test.iloc[:, 0:n_features].values.astype(np.float)

# generate labels

Y_test = data_test.iloc[:, n_target-1:n_target].values.astype(np.float).ravel()
##################################

# MODELING

##################################

rf = RandomForestRegressor(n_estimators= 50, criterion = 'mse', random_state=42, verbose = 1)#random_state=42,warm_start=True



# Train the model on training data

rf.fit(X_train, Y_train);
##################################

# TEST DATA

##################################

score = rf.score(X_test, Y_test)

    

# Use the forest's predict method on the test data

Y_pred = rf.predict(X_test).round()



#KPIs

#Number of features

k = X_test.shape[1]

#No. of data samples

n = len(X_test)



errors = abs(Y_pred - Y_test)

ACC = accuracy_score(Y_test, Y_pred)

RMSE = float(format(np.sqrt(mean_squared_error(Y_test, Y_pred)), '.3f'))

MSE = mean_squared_error(Y_test, Y_pred)

MAE = mean_absolute_error(Y_test, Y_pred)

R2 = r2_score(Y_test, Y_pred, multioutput='variance_weighted')

adjR2 = 1- ((1-R2)*(n-1))/(n-k-1)



print('\nAccuracy: ',ACC,'%')

print('Average absolute error:', round(np.mean(errors), 2))

print("Root Mean Squared Error (RMSE): ", RMSE)

print("Mean Squared Error (MSE): ", MSE)

print("Mean Absolute Error (MAE): ", MAE)

print("R2_Score: ", R2)

print("Adjusted R2: ", adjR2)

print('Score: ',score.round(2))



# Plot in blue color the predicted data and in green color the

# actual data to verify visually the accuracy of the model.

plt_title = 'Predict RF Acurracy '+str(ACC)+'%. R^2 '+str(round(R2,2))

fig_verify = plt.figure(figsize=(20, 10))

plt.plot(Y_pred, color="red")

plt.plot(Y_test, color="green")

plt.title('prediction')

plt.ylabel('RUL in cycles')

plt.xlabel('machineID')

plt.legend(['predicted', 'actual data'], loc='upper left')

plt.title(plt_title)

plt.show()

#fig_verify.savefig("Output/model_regression_verifyRF_test.png")



fig_verify = plt.figure(figsize=(20, 10))

plt.plot(Y_pred, color="red")

plt.title('prediction')

plt.ylabel('RUL in cycles')

plt.xlabel('machineID')

plt.legend(['predicted'], loc='upper left')

plt.show()

#fig_verify.savefig("Output/model_regression_verifyRF_pred.png")



fig_verify = plt.figure(figsize=(20, 10))

plt.plot(Y_test, color="green")

plt.title('true data')

plt.ylabel('RUL in cycles')

plt.xlabel('machineID')

plt.legend(['actual data'], loc='upper left')

plt.show()

#fig_verify.savefig("Output/model_regression_verifyRF_true.png")
##################################

# FETURES IMPORTANCES

##################################



# Get numerical feature importances

importances = list(rf.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Feature: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



# Set the style

plt.style.use('fivethirtyeight')

# list of x locations for plotting

x_values = list(range(len(importances)))

# Make a bar chart

plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title('Feature Importances');