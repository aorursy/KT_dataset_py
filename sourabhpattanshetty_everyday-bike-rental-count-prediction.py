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
#Importing Necessary Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



import warnings

warnings.filterwarnings("ignore")
inp0=pd.read_csv('../input/boom-bikes-sharing-dataset/day.csv')
#Data Dimensions

inp0.shape
#First Few rows of the data

inp0.head()
#More information about the columns

inp0.info()
#Missing Values in the data

inp0.isnull().sum()
#Keeping original data intact and Dropping Columns 'Instant' and 'dteday' in new dataframe

inp1=inp0.drop({'instant','dteday'},axis='columns')
#Checking Correlation between temp and atemp as they look mostly similar

inp1['temp'].corr(inp1['atemp'])

#Since both the columns are highly related, We can drop one of them. 
#Dropping column 'temp'

inp1=inp1.drop({'temp'},axis='columns')
#dropping Casual and Registered user counts

inp1=inp1.drop({'casual','registered'},axis='columns')
inp1.columns
inp1.rename(columns={

    'atemp':'Temperature',

    'season':'Season',

    'yr':'Year',

    'mnth':'Month',

    'holiday':'Holiday',

    'weekday':'Weekday',

    'workingday':'WorkingDay',

    'weathersit':'WeatherSituation',

    'hum':'Humidity',

    'windspeed':'Windspeed',

    'cnt':'Count'}, inplace=True)
#Checking the Distribution of Data in all columns 

inp1.describe()
plt.figure(figsize=(10,10))

plt.subplot(221)

sns.boxplot(inp1['Temperature'])



plt.subplot(222)

sns.boxplot(inp1['Humidity'])



plt.subplot(223)

sns.boxplot(inp1['Windspeed'])



plt.subplot(224)

sns.boxplot(inp1['Count'])



plt.show()
#In above graph, We can see Humidity is 0 in one of the case, Lets see what are the rows there.

inp1[inp1['Humidity']==0]
#Given the conditions of earth, Humidity can never be zero. Lets drop this row to ensure we normalise the humidity.

inp1=inp1[~(inp1['Humidity']==0)]

sns.boxplot(inp1['Humidity'])

plt.show()
inp1.head()
#Creating dummies for Season

dummies=pd.get_dummies(inp1['Season'], drop_first = True)

for i in dummies.columns:

    inp1['Season'+'_'+str(i)]=dummies[i]

inp1.drop(columns={'Season'},inplace=True)



#Creating Dummies for Weekday

dummies=pd.get_dummies(inp1['Weekday'],drop_first=True)

for i in dummies.columns:

    inp1['Weekday'+'_'+str(i)]=dummies[i]

inp1.drop(columns={'Weekday'},inplace=True)



#Creating Dummies for Weather Situation

dummies=pd.get_dummies(inp1['WeatherSituation'],drop_first=True)

for i in dummies.columns:

    inp1['WeatherSituation'+'_'+str(i)]=dummies[i]

inp1.drop(columns={'WeatherSituation'},inplace=True)



#Creating Dummies for Month

dummies=pd.get_dummies(inp1['Month'],drop_first=True)

for i in dummies.columns:

    inp1['Month'+'_'+str(i)]=dummies[i]

inp1.drop(columns={'Month'},inplace=True)
#Pair plots between numeric variables.

sns.pairplot(inp1[['Temperature','Humidity','Windspeed','Count']])

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (20, 15))

sns.heatmap(round(inp1.corr()*100,1), annot = True, cmap="YlGnBu")

plt.show()
#Distribution of Categorical Variables

plt.figure(figsize=(20, 12))



features = ['yr','mnth','season','weathersit','holiday','workingday','weekday']



for i in enumerate(features):

    plt.subplot(2,4,i[0]+1)

    sns.boxplot(x = i[1], y = 'cnt', data = inp0)

    plt.title(i[1])
#Distribution of Numeric columns

plt.figure(figsize=(20, 6))



features = ['Temperature','Humidity','Windspeed','Count']



for i in enumerate(features):

    plt.subplot(1,4,i[0]+1)

    sns.distplot(inp1[i[1]])

    plt.title(i[1])

plt.show()
y=inp1.Count

X=inp1.drop('Count',axis=1)
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
Scaler=MinMaxScaler()

obj=Scaler.fit(X_train[['Temperature', 'Humidity','Windspeed']])

X_train[['Temperature', 'Humidity','Windspeed']]=obj.transform(X_train[['Temperature', 'Humidity','Windspeed']])
lr = LinearRegression()

rfe = RFE(lr, n_features_to_select=15)

rfe = rfe.fit(X_train, y_train)
RFE=pd.DataFrame()

RFE['Feature']=X_train.columns

RFE['support']=rfe.support_

RFE['ranking']=rfe.ranking_

RFE.sort_values('ranking')
Features=list(RFE[RFE['ranking']==1].Feature.reset_index(drop=True))

Features
#Below are some functions that make further evaluation easy

def Model(Features):

    X_train_temp = X_train[Features] # get feature list 

    X_train_temp = sm.add_constant(X_train_temp) # required by statsmodels 

    lr = sm.OLS(y_train, X_train_temp).fit() # build model and learn coefficients

    print(lr.summary())

    vif(Features)# OLS summary with R-squared, adjusted R-squared, p-value etc.

def vif(Features):

    vif = pd.DataFrame()

    vif['Features'] = X_train[Features].columns # Read the feature names

    vif['ViF'] = [variance_inflation_factor(X_train[Features].values,i) for i in range(X_train[Features].shape[1])] # calculate VIF

    vif['ViF'] = round(vif['ViF'],2)

    vif.sort_values(by='ViF', ascending = False, inplace=True)  

    print(vif.reset_index(drop=True)) # prints the calculated VIFs for all the features
Features1=Features.copy()

Model(Features1)
Features2=Features.copy()

Features2.remove('Temperature')

Model(Features2)
Features3=Features.copy()

Features3.remove('Humidity')

Model(Features3)
Features4=Features3.copy()

Features4.remove('Season_3')

Model(Features4)
Features5=Features4.copy()

Features5.remove('Month_5')

Model(Features5)
Features6=Features5.copy()

Features6.remove('Month_3')

Model(Features6)
Features7=Features6.copy()

Features7.remove('Month_10')

Model(Features7)
plt.figure(figsize=(10,10))

sns.heatmap(inp1[Features7].corr())

plt.show()
Features8=Features7.copy()

Features8.remove('Month_8')

Model(Features8)
X_train_final=X_train[Features8]

X_train_final = sm.add_constant(X_train_final) 

lr=sm.OLS(y_train,X_train_final).fit()

print(lr.summary())
#Scaling y For better Interpretation after we get final features

y_train_scaled=pd.DataFrame(y_train)

scaler_y = MinMaxScaler()

scaled_array = scaler_y.fit_transform(y_train_scaled)

y_train_scaled=pd.DataFrame(scaled_array,columns=['Count'])
X_train_final=X_train[Features8]

X_train_final = sm.add_constant(X_train_final) 

lr=sm.OLS(y_train_scaled.Count.values.reshape(-1,1),X_train_final).fit()

print(lr.summary())
X_train_final=X_train[Features8]

X_train_final = sm.add_constant(X_train_final) 

lr=sm.OLS(y_train,X_train_final).fit()

print(lr.summary())
y_train_predicted = lr.predict(X_train_final) # get predicted value on training dataset using statsmodels predict()

residual_values = y_train - y_train_predicted # difference in actual Y and predicted value

plt.figure(figsize=[10,5])

plt.subplot(121)

sns.distplot(residual_values, bins = 15) # Plot the histogram of the error terms

plt.title('Residuals follow normal distribution', fontsize = 18)

plt.subplot(122) 

plt.scatter(y_train_predicted, residual_values) # Residual vs Fitted Values

plt.plot([0,0],'r') # draw line at 0,0 to show that residuals have constant variance

plt.title('Residual vs Fitted Values: No Pattern Seen')

plt.show()
from math import sqrt

train_rmse = sqrt(mean_squared_error(y_train,y_train_predicted))

print('Root mean square error :',train_rmse)
train_mae=mean_absolute_error(y_train,y_train_predicted)

print('Mean absolute error :',train_mae)
train_r2 = round(r2_score(y_train, y_train_predicted),3)

print('R2 Score :',train_r2)
n = X_train_final.shape[0]

p = len(Features8)

train_Adj_R2=round(1-(1-train_r2)*(n-1)/(n-p-1),3) 

print('Adjusted R2 Score:',train_Adj_R2)
#Scaling the Test Dataset using Object used while Scaling Train Dataset

X_test[['Temperature', 'Humidity','Windspeed']]=obj.transform(X_test[['Temperature', 'Humidity','Windspeed']])
#Selecting The final 9 Features

X_test_final=X_test[Features8]
X_test_final = sm.add_constant(X_test_final)

y_test_predicted = lr.predict(X_test_final)
train_r2 = round(r2_score(y_train, y_train_predicted),3)

test_r2 = round(r2_score(y_test, y_test_predicted),3)

print('R2 Score of Train and Test Datasets are:',train_r2,' and ',test_r2)
n = X_test_final.shape[0]

p = len(Features8)

test_Adj_R2=round(1-(1-test_r2)*(n-1)/(n-p-1),3) 

print('Adjusted R2 Score of Train and Test Datasets are:',train_Adj_R2,' and ',test_Adj_R2)