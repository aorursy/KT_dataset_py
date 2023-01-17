# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

# generate related variables

from numpy import mean

from numpy import std

import seaborn as sns

from scipy.stats import norm 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        





# Any results you write to the current directory are saved as output.
# Question 1

dirtData = pd.read_csv("../input/dwdm-dirt-lab-1/DWDM Dirt Lab 1.csv",encoding='ISO-8859-1',  na_values=' ')
dirtData.head(5)
# Question 2 - selecting the regions column then counting unique values with the largest value at top and lowest at bottom



RegionCount = dirtData['Region'].value_counts().rename_axis('Region').reset_index(name='counter')

RegionCount
#the 3rd item will be the 3rd most occuring in the result and therefore we isolate that option

RegionCount.iloc[2] 
#

RegionCount[RegionCount.counter > 8].plot(kind='bar', x= "Region", y="counter", title= "Occurance of Regions over 8 times", figsize=(14,7))

#Question 4 - get gender column

Gender = dirtData['Gender']

Gender
#seperate male from the data into its own variable

GM = Gender[dirtData['Gender']=='Male']

GM
#seperate female from the data into its own variable

GF = Gender[dirtData['Gender']=='Female']

GF
#count males

GenderMC = GM.value_counts()

GenderMC
#count females

GenderFC = GF.value_counts()

GenderFC
#add results back into one variable

newGenderC = pd.concat([GenderMC, GenderFC])

newGenderC
#produce pie chart of results

labels = 'Male','Female'

plt.pie(newGenderC,labels=labels,autopct='%1.1f%%', shadow=True, startangle=90 )

plt.title = ("Male Vs Female Data")



#Question 5 - determine data types N.B csv was imported and set empty values to NaN by default

dirtData.dtypes 
#ensure NaN values are set to 0

Hcm = dirtData['Height_cm'].fillna(0)

Flcm = dirtData['Footlength_cm'].fillna(0)

Ascm = dirtData['Armspan_cm'].fillna(0)

Rt = dirtData['Reaction_time'].fillna(0)

#run column test to ensure value is set to 0

Hcm

#Convert arguments to a numeric type

pd.to_numeric(Hcm)

pd.to_numeric(Flcm)

pd.to_numeric(Ascm)

pd.to_numeric(Rt)
#Question 6 - Count all Nan records

dirtData.isnull().sum().sum()
#Question 7 - delete first record

dirtData.drop(dirtData.index[0])
#Question 8 - delete all records with NaN

cleanData = dirtData.dropna()
cleanData
#count NaN records to verify deleted

cleanData.isnull().sum().sum()
#Question 9 - get required columns to correlate 

correlData =cleanData[['Age-years','Height_cm','Footlength_cm','Armspan_cm','Languages_spoken','Travel_time_to_School','Reaction_time','Score_in_memory_game' ]]

correlData
correlData.shape 
#check if all columns are similar types i.e int

correlData.dtypes
#show correlation Matrix

correlMap = correlData.corr()

correlMap
#Visual display of results - green means positive, red means negative

correlMapv = sns.heatmap(

    correlMap, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

#Armspan_cm and Height_cm have the most significant correlation

#Question 10

newCleanData = cleanData[['Footlength_cm','Armspan_cm', 'Height_cm','Reaction_time']]

newCleanData
#Question 11

newCleanData.plot(kind='scatter', x='Armspan_cm', y="Height_cm", title="Relationship between Armspan and Height", figsize=(12,9))
#Question 12 - put data into dependent Y and independent X variables

X_Cdata = newCleanData[['Height_cm','Footlength_cm','Armspan_cm']]

Y_Cdata = newCleanData['Reaction_time']
#70/30 testing/training split of data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_Cdata, Y_Cdata, test_size=0.30) 

 
# Create an instance of linear regression

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.coef_
X_train.columns

print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
reg.intercept_
#Question 13



test_predicted = reg.predict(X_test)

#determine residuals

y_test_ar = y_test.values

residuals = y_test_ar - test_predicted



# create dataframe for plot

plotDF = pd.DataFrame(residuals)



plotDF = plotDF.rename(index=str, columns={0:'Residuals'})



plotDF2 = pd.DataFrame(test_predicted)

plotDF2 = plotDF2.rename(index=str, columns={0:'Predicted'})





plotF = pd.concat([plotDF,plotDF2], axis=1)



plotF.plot(kind="scatter",

    x='Predicted',

    y='Residuals',

    title="Residuals vs Predicted Values",

    figsize=(12,8)

)
#Question 14

reg.score(X_test,y_test)

#The score given is negative and tells us that the model is not acceptable and will not predict outcomes accurately

# as the fact that the X data minimally affects the outcome of Y data 
#Question 16

#regression formula Y = M1 X1 + M2 X2 + M3 * X3 + B





#Question 17

reg.predict([[23.3,172.0,173.2]])
#Question 18

import sklearn.metrics as metrics

# mean squared error

print("Mean squared error: %.2f" % metrics.mean_squared_error(y_test, test_predicted))
# root mean squared error RMSE

import math  

print("Root Mean squared error: %.2f" % math.sqrt(metrics.mean_squared_error(y_test, test_predicted)))
# mean absolute error MAE

print("Mean absolute error: %.2f" % metrics.mean_absolute_error(y_test, test_predicted))