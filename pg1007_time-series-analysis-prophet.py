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
# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
print(os.getcwd())
print(os.path.abspath('../input'))
train_data=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv',parse_dates=['date'])
print(train_data.info())
print(train_data.describe())

train_data.isnull().sum()
train_data=train_data.sort_values(['date'])
train_data
# plotting using plt
plt.figure(figsize=(10,8))
plt.plot(train_data['date'],train_data['sales'])
plt.grid()
plt.show()
#plotting using go
py.iplot([go.Scatter(x=train_data.date,y=train_data.sales)])
sales_vs_item=train_data[['item','sales']]
sales_vs_item.set_index('item',inplace=True)


plt.figure(figsize=(15,10))
sns.distplot(sales_vs_item['sales'])
# maximum no of products  lies between 30 to 50
data=sales_vs_item.groupby(['item']).mean()
plt.figure(figsize=(20,10))
sns.barplot(data.index,data['sales'])
indexed_data=train_data.set_index(['date','store','item'])
indexed_data
# Sales trend over the months and year
train_data['Month']=train_data['date'].dt.month
train_data['Year']=train_data['date'].dt.year
plt.figure(figsize=(10,30))
sns.catplot(data = train_data, x ='Month', y = "sales",row = "Year")
plt.show()
store_df = train_data.copy()
sales_pivoted_df = pd.pivot_table(store_df, index='store', values=['sales'], columns='item', aggfunc=np.mean)
# Pivoted dataframe
display(sales_pivoted_df)
store_data=train_data[['store','sales']]
store_data.set_index('store',inplace=True)
store_data=store_data.groupby('store').mean()
plt.figure(figsize=(10,10))
sns.barplot(store_data.index,store_data['sales'])
# performing time series analysis for particular item of store
# print(train_data)
print('Before Filtering '+str(train_data.shape))
# let consider particular item and store
# store=10
# item=40

sample=train_data[train_data.store==10]
sample=sample[sample.item==40]
print('After Filtering '+str(sample.shape))
# print(sample)
py.iplot([go.Scatter(
    x=sample.date,
    y=sample.sales)])
print('Before Filtering '+str(train_data.shape))

item=[10,20,25,45]
store=[1,5,8,9]
sample=train_data.copy()
sample=sample[sample.item.isin(item)]
sample=sample[sample.store.isin(store)]

print('After Filtering '+str(train_data.shape))

multi_data = []
for i in range(0,4):
    flt = sample[sample.store == store[i]]
    flt = flt[flt.item == item[i]]
    multi_data.append(go.Scatter(x=flt.date, y=flt.sales, name = "Store:" + str(store[i]) + ",Item:" + str(item[i])))
py.iplot(multi_data)
train_data=train_data.rename(columns={'date':'ds','sales':'y'})
train_data
model=Prophet(yearly_seasonality=True)
model.fit(train_data)
forecast=model.make_future_dataframe(periods=90)
forecast=model.predict(forecast)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
figure=model.plot(forecast,xlabel='Date',ylabel='Sales')
figure2=model.plot_components(forecast)
def predictions(item,store):
        pg=Prophet(yearly_seasonality=True)
        test=train_data[train_data.store==store]
        test=test[test.item==item]
        pg.fit(test)
        future=pg.make_future_dataframe(periods=90)
        forecast=pg.predict(future)
        forecast_final=forecast[forecast['ds'].dt.year==2018]
        return forecast_final[['ds','yhat']]
results=list()
for i in range(1,11):
    for j in range(1,51):
        result=predictions(j,i)
        print(result)
        results.append(result['yhat'].values)

final_result=[]
for each in results:
    for ele in each:
        final_result.append(ele)
len(final_result)
final_result=pd.DataFrame(final_result)
final_result.iloc[:,0]
test_data=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')
test_data.id
output = pd.DataFrame({'Id': test_data.id,
                      'sales':final_result.iloc[:,0]})
output.to_csv('submission.csv', index=False)
# output
