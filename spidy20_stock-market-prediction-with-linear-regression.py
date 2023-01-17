#For Quandl installation in your kernel 

!pip install Quandl
import pandas as pd #For data related tasks

import matplotlib.pyplot as plt #for data visualization 

import quandl #Stock market API for fetching Data

from sklearn.linear_model import LinearRegression
quandl.ApiConfig.api_key = ''## enter your key 

stock_data = quandl.get('NSE/TCS', start_date='2018-12-01', end_date='2018-12-31')

#Let's see the data

print(stock_data)
dataset = pd.DataFrame(stock_data)
dataset.head()

##Now we convert into csv

dataset.to_csv('TCS.csv')
## We have to read our CSV

data = pd.read_csv('TCS.csv')
data.head()
data.isnull().sum()
import seaborn as sns

plt.figure(1 , figsize = (17 , 8))

cor = sns.heatmap(data.corr(), annot = True)
#Let's select our features

x = data.loc[:,'High':'Turnover (Lacs)']

y = data.loc[:,'Open']
x.head()
y.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 0)
LR = LinearRegression()
LR.fit(x_train,y_train)
LR.score(x_test,y_test)
##I given a test data of random day

Test_data = [[2017.0 ,1979.6 ,1990.00 ,1992.70 ,2321216.0 ,46373.71]]

prediction = LR.predict(Test_data)
print(prediction)