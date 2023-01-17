#For Quandl installation in your kernel 

!pip install Quandl




import pandas as pd #For data related tasks

import matplotlib.pyplot as plt #for data visualization 

import quandl #Stock market API for fetching Data

from sklearn.tree import DecisionTreeRegressor # Our DEcision Tree classifier
quandl.ApiConfig.api_key = ''## enter your key 

stock_data = quandl.get('NSE/INFIBEAM', start_date='2018-12-01', end_date='2018-12-31')

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
Classifier = DecisionTreeRegressor()
Classifier.fit(x_train,y_train)
test = [[46.50 ,43.10 ,44.40, 44.45, 13889470.0 ,6219.22]]
prediction = Classifier.predict(test)
prediction