#Importing Libraries  



import pandas as pd 

import numpy as np 





from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder





import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline





import scipy.stats as st

import sklearn.metrics as sm
#Loading the data set 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



data = pd.read_csv('/kaggle/input/austin-weather/austin_weather.csv')

data
# checking the shape and variable types in dataset 

print(data.shape)

print(data.dtypes)
# Cleaning the data 



data.isnull().sum()



# Here the number of null values in Event column is displayed as 0 but we can see some missing rows in that column. 

# So, its good to go through the raw data beforehand

# Checking for charecters in object columns with numerical values

print('Unique values in DewPointHF:\n',data.DewPointHighF.unique())

print('\nUnique values in DewPointAvgF:\n',data.DewPointAvgF.unique())

print('\nUnique values in DewPointLowF:\n',data.DewPointLowF.unique())



print('\nUnique values in HumidityHighPercent:\n',data.HumidityHighPercent.unique())

print('\nUnique values in HumidityAvgPercent:\n',data.HumidityAvgPercent.unique())

print('\nUnique values in HumidityLowPercent:\n',data.HumidityLowPercent.unique())



print('\nUnique values in SeaLevelPressureHighInches:\n',data.SeaLevelPressureHighInches.unique())

print('\nUnique values in SeaLevelPressureAvgInches:\n',data.SeaLevelPressureAvgInches.unique())

print('\nUnique values in SeaLevelPressureLowInches:\n',data.SeaLevelPressureLowInches.unique())



print('\nUnique values in VisibilityHighMiles:\n',data.VisibilityHighMiles.unique())

print('\nUnique values in VisibilityAvgMiles:\n',data.VisibilityAvgMiles.unique())

print('\nUnique values in VisibilityLowMiles:\n',data.VisibilityLowMiles.unique())



print('\nUnique values in WindHighMPH:\n',data.WindHighMPH.unique())

print('\nUnique values in WindAvgMPH:\n',data.WindAvgMPH.unique())

print('\nUnique values in WindGustMPH:\n',data.WindGustMPH.unique())



print('\nUnique values in PrecipitationSumInches:\n',data.PrecipitationSumInches.unique())



print('\nUnique values in Events:\n',data.Events.unique())



# Removing charecters from the datset



data = data.replace(to_replace ="-", value ="0")

data = data.replace(to_replace = ' ', value = 'Nan')

data['PrecipitationSumInches'] = data[['PrecipitationSumInches']].replace('T', '0')
# Changing dtypes



data['Date'] = pd.to_datetime(data.Date)



data['DewPointHighF'] = data['DewPointHighF'].astype(float)

data['DewPointAvgF'] = data['DewPointAvgF'].astype(float)

data['DewPointLowF'] = data['DewPointLowF'].astype(float)

data['HumidityHighPercent'] = data['HumidityHighPercent'].astype(float)

data['HumidityAvgPercent'] = data['HumidityAvgPercent'].astype(float)

data['HumidityLowPercent'] = data['HumidityLowPercent'].astype(float)

data['SeaLevelPressureHighInches'] = data['SeaLevelPressureHighInches'].astype(float)

data['SeaLevelPressureAvgInches'] = data['SeaLevelPressureAvgInches'].astype(float)

data['SeaLevelPressureLowInches'] = data['SeaLevelPressureLowInches'].astype(float)

data['VisibilityHighMiles'] = data['VisibilityHighMiles'].astype(float)

data['VisibilityAvgMiles'] = data['VisibilityAvgMiles'].astype(float)

data['VisibilityLowMiles'] = data['VisibilityLowMiles'].astype(float)

data['WindHighMPH'] = data['WindHighMPH'].astype(float)

data['WindAvgMPH'] = data['WindAvgMPH'].astype(float)

data['WindGustMPH'] = data['WindGustMPH'].astype(float)

data['PrecipitationSumInches'] = data['PrecipitationSumInches'].astype(float)

# Encoding categorial Column with LabelEncoder



label = LabelEncoder()

data['Events'] = label.fit_transform(data['Events'].astype('str'))



print(data['Events'].head())

print(data.Events.unique())
data.dtypes
# Extracting Year , Month and Day from Column Date



data['Year'] = data['Date'].dt.year

data['Month'] = data['Date'].dt.month

data['Day'] = data['Date'].dt.day
data = data.drop('Date' ,axis = 1)

data.head()
# Correlation between the data Dimensionss 



corrMatrix = data.corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corrMatrix, annot=True,)
#Spliting the data 

x = data.drop('TempHighF' , axis = 1).values

y = data['TempHighF'].values
X_train , X_test , Y_train , Y_test = train_test_split(x,y, test_size = 0.3 , random_state = 0)
lnr = LinearRegression()



lnr.fit(X_train,Y_train)
y_pred = lnr.predict(X_test)

df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})



df.head()
plt.style.use('dark_background')

plt.figure(figsize=(10, 10))

plt.scatter(Y_test, y_pred, c = 'red')

plt.xlabel('Actual values')

plt.ylabel('Predicted Values')
# Checking the accuracy  of the model 



print("Accuracy of the model =",lnr.score(X_train, Y_train))

print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, y_pred),2)) 

print("Mean squared error =", round(sm.mean_squared_error(Y_test, y_pred), 2)) 

print("Median absolute error =", round(sm.median_absolute_error(Y_test, y_pred), 2)) 

print("Explain variance score =", round(sm.explained_variance_score(Y_test, y_pred), 2)) 

print("R2 score =", round(sm.r2_score(Y_test, y_pred), 2))
