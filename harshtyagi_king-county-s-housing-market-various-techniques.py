import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = 8, 10
df = pd.read_csv('../input/kc_house_data.csv', parse_dates = ['date']) # The parse_date will change date column to readable format
df.head()
df.info()
df['month'] = df['date'].dt.month

df['year'] = df['date'].dt.year
price = df['price']

df.drop('price', inplace=True, axis = 1)

df['price'] = price
df.head(2)
# Month vs price

sns.barplot(x = df['month'], y = df['price'], data = df)
heatMap = df.corr()

f, ax = plt.subplots(figsize=(25,16))

sns.plt.yticks(fontsize=18)

sns.plt.xticks(fontsize=18)



sns.heatmap(heatMap, cmap='inferno', linewidths=0.1,vmax=1.0, square=True, annot=True)
sns.countplot(x = df['month'],data = df)
# Seperating the data by year.

filter2015 = df['year'] == 2015

filter2014 = df['year'] == 2014
freq2014 = df[filter2014]['price']/df[filter2014]['sqft_living']

freq2015 = df[filter2015]['price']/df[filter2015]['sqft_living']
plt.hist(x = freq2014, bins = 10, histtype = 'stepfilled')

plt.xlabel('price/sqft_living')
plt.hist(x = freq2015, bins = 10, histtype = 'stepfilled')

plt.xlabel('price/sqft_living')
price2015 = sum(df[filter2015]['price'])/len(df[filter2015]['price'])

price2014 = sum(df[filter2014]['price'])/len(df[filter2014]['price'])

print('The average cost in the year 2015 is: ',price2015)

print('The average cost in the year 2014 is: ',price2014)

print('The percentage increase is: ',((price2015-price2014)*100)/price2014,'%')
df.drop(['date', 'id'], inplace = True, axis = 1)
#seperating the dataset into independent variables and dependent variable

x = df.iloc[:,:-1].values    # All the independent variables. 

y = df.iloc[:,20].values     # dependent variable 'price'
#Splitting the dataset into test set and train set

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0,test_size=0.35)
from sklearn.linear_model import LinearRegression

MLregressor = LinearRegression()

MLregressor.fit(x_train, y_train)

scoreML = MLregressor.score(x_test,y_test)
from sklearn.tree import DecisionTreeRegressor

regDT = DecisionTreeRegressor(random_state = 0, criterion = 'mae',min_samples_split=18, min_samples_leaf=10)

regDT.fit(x_train, y_train)
scoreDT = regDT.score(x_test,y_test)
from sklearn.ensemble import RandomForestRegressor

regRF = RandomForestRegressor(n_estimators=400, random_state = 0)

regRF.fit(x_train,y_train)
scoreRF = regRF.score(x_test,y_test)
from sklearn.svm import SVR

regSVR = SVR(kernel = 'sigmoid',degree=5)

regSVR.fit(x_test,y_test)
scoreSVR = regSVR.score(x_test,y_test)
Scores = pd.DataFrame({'Classifiers': ['Multiple Linear Regression', 'Decision Tree', 'Random Forest', 'SVM'],

                      'Scores': [scoreML, scoreDT, scoreRF, scoreSVR]})
Scores
pd.options.display.float_format = '${:,.2f}'.format  #To format the output
regRF.predict(x_test)  #predicts the prices
output = pd.DataFrame({'Actual Price':y_test,

                      'Predicted price': regRF.predict(x_test)})
output.to_csv('output.csv', index=False, encoding='utf-8')