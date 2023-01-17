# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import subprocess
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sales = pd.read_csv("../input/sales data-set.csv")
features = pd.read_csv("../input/Features data set.csv")
stores = pd.read_csv("../input/stores data-set.csv")
print("Sales Total Col.",len(sales.columns),"\nShape:",sales.shape,"\nColumns:",sales.columns.tolist(),"\n=============")
print("Features Total Col.",len(features.columns),"\nShape:", features.shape, "\nColumns:",features.columns.tolist(),"\n=============")
print("Stores Total Col.",len(stores.columns),"\nShape:",stores.shape, "\nColumns:",stores.columns.tolist())

def insight(df):
    print("--------------------")
    print(df.head())
    

insight(sales)
insight(features)
insight(stores)
final = sales.merge(features,how="left", on=['Store', 'Date', 'IsHoliday'])
final = final.merge(stores, how= "left", on=['Store'])
final.head()
print("Final Dataset Col:",len(final.columns),"\nShape: ",final.shape,"\nColumns",final.columns.tolist())
info = pd.DataFrame(final.dtypes).T.rename(index = {0:'Column Type'})
info = info.append(pd.DataFrame(final.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info = info.append(pd.DataFrame(final.isnull().sum()/final.shape[0]*100).T.rename(index = {0:'null values{%}'}))
info
final.fillna(-9999, inplace=True)
info = pd.DataFrame(final.dtypes).T.rename(index = {0:'Column Type'})
info = info.append(pd.DataFrame(final.isnull().sum()).T.rename(index = {0:'null values (nb)'}))
info = info.append(pd.DataFrame(final.isnull().sum()/final.shape[0]*100).T.rename(index = {0:'null values{%}'}))
info
print("Duplicate Values : ",final.duplicated().sum())
final = final.applymap(lambda x: 1 if x ==  True  else x)
final = final.applymap(lambda x: 0 if x ==  False  else x)
final.head()
#Average Sales for all store/department for Week

df_average_sales_week = final.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
df_average_sales = df_average_sales_week.sort_values('Weekly_Sales', ascending=False)

print(df_average_sales[:10])

#Seasonality vs Trend Analysis
plt.figure(figsize=(15,6))
plt.plot(df_average_sales_week.Date, df_average_sales_week.Weekly_Sales)
plt.show()
#Sales variation during Holidays(Store/Dept)
holiday =  final[['Date', 'IsHoliday', 'Weekly_Sales']].copy()
holiday =  holiday.groupby(by=['Date','IsHoliday'], as_index=False)['Weekly_Sales'].sum()
holiday_group =  holiday.groupby(by=['IsHoliday'], as_index=False)['Weekly_Sales'].sum()
print( holiday_group)
#print( holiday[:5])

def holiday_sales(df):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(15,6))
    labels = ['Date', 'IsHoliday_x', 'Weekly_Sales']
    plt.title('Sales Variation During Holidays')
    plt.plot(df.Date, df.Weekly_Sales)
    plt.show()
    
holiday_sales(holiday)
final['Return'] = (final['Weekly_Sales'] < 0).astype('int')
final_group = final.groupby(['Return'], as_index = False)['Weekly_Sales'].sum() 
final_group

#Making Avg MarkDown
final['AvgMarkDown'] = final['MarkDown1'] + final['MarkDown2'] + final['MarkDown3'] + final['MarkDown4'] + final['MarkDown5']
final['AvgMarkDown'] = final['AvgMarkDown'] / 5
final.AvgMarkDown[378:385]
#Creating Weekly sales in a 4 range

final['cum_sum'] = final.Weekly_Sales.cumsum()
final['cum_perc'] = 100*final.cum_sum/final.Weekly_Sales.sum()

final['rangeA'] = 0
final['rangeA'][final['cum_perc'] <= 25] = 1

final['rangeB'] = 0
final['rangeB'][(final['cum_perc'] > 25) & (final['cum_perc'] <= 50)] = 1

final['rangeC'] = 0
final['rangeC'][(final['cum_perc'] > 50) & (final['cum_perc'] <= 75)] = 1

final['rangeD'] = 0
final['rangeD'][final['cum_perc'] > 75] = 1

final = final.drop(['cum_perc', 'cum_sum'], 1)

final.head(100)
#Aggregate the Top performing stores interms of sales
top_stores = final.groupby(by=['Type'], as_index=False)['Weekly_Sales'].sum()
top_stores
clm = final[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size',
                  'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'AvgMarkDown', 'rangeA', 
                  'rangeB', 'rangeC', 'rangeD', 'Return']].copy()
clm.corr()
clm = final[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size',
                   'Return']].copy()
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    
    fig = plt.figure(figsize = (25,15))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 50)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Store Features Correlation')
    labels=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size']
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    #Add colorbar to make sure to specify a tick location to match desired tick labels
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()
    
correlation_matrix(clm)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')


#Dropping the 'Label' from  and assigning to X
X = np.array(final.drop(['Weekly_Sales', 'Date', 'Type', 'MarkDown1', 'MarkDown4'], 1))
X = preprocessing.scale(X)


final.dropna(inplace=True)
Y = np.array(final['Weekly_Sales'])


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

list1 = [X_train, X_test, Y_train, Y_test]
for i in list1:
    print(i.shape)
# Training Model

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

print(accuracy)
