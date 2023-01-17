#Importing packages

import numpy as np

import pandas as pd

import seaborn as sb

#import category_encoders as ce

import matplotlib.pyplot as plt

import pylab as pl

from pandas import ExcelFile

import xlrd

from sklearn import preprocessing



%matplotlib inline
#Load spreadsheet.

file = pd.read_csv('../input/WeatherAUS.csv')



#Exploratory data analysis

print (file.shape)

file.head(5)

print (file.info())

print ("*******************************************")

file.describe()
# count target class

sb.countplot(x='RainTomorrow', data=file,  palette = "RdBu")

plt.show()

sb.countplot(x='RainTomorrow', data=file, hue='RainToday', palette = "RdBu")

plt.show()
#Changing the target feature to binary values

file.RainToday = [1 if each=="Yes" else -1 for each in file.RainToday]

file.RainTomorrow = [1 if each=="Yes" else -1 for each in file.RainTomorrow]

file.head(5)
Missing_Value_Percentage = 100 - ((file.count().sort_values()/len(file))*100)

print ("The percentage of missing values in each column is:")

Missing_Value = pd.DataFrame(Missing_Value_Percentage)

Missing_Value.columns =  ['Summary']

Missing_Value
# we can also plot a heatmap to visualize missing values

plt.figure(figsize=(8,6))

sb.heatmap(file.isnull(),yticklabels=False,cbar=False,cmap='nipy_spectral_r')
Data = file.drop(['Sunshine','Evaporation','Cloud3pm','Cloud9am','RISK_MM','Location','Date','WindGustDir',

       'WindDir9am', 'WindDir3pm'],axis=1)

Data.shape
plt.figure(figsize=(8,6))

sb.heatmap(Data.isnull(),yticklabels=False,cbar=False,cmap='nipy_spectral_r')

plt.title('Missing values')
# replace rest of the nulls with respective means

fill_feat = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm',

             'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm','Temp9am', 'Temp3pm',

             'RainToday', 'RainTomorrow']



for i in fill_feat:

    Data[i].fillna(np.mean(Data[i]),inplace=True)



sb.heatmap(Data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()

#plt.savefig('Figure 1.png')

Data.count().sort_values()