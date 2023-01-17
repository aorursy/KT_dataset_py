path = "../input/m5-forecasting-accuracy"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
import os
df1 = pd.read_csv(os.path.join(path, "sell_prices.csv")) #import the data sets
df1.head(5)

df1.shape
df2 = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))
df2.head(5)
df2.shape
df3 = pd.read_csv(os.path.join(path, "calendar.csv"))
df3.head(5)
df3.shape
df2_subset = df2[df2['item_id'] == 'HOBBIES_1_002'] 
df2_subset
plt.figure(figsize=(15, 5))
plt.plot(df2_subset.iloc[0, 6:].values)
plt.xlabel('Days')
plt.ylabel('NUmber of Units sales')
plt.title('HOBBIES_1_002 item sales within time period')
plt.figure(figsize=(15, 5))
plt.plot(df2_subset.iloc[0, 6:].rolling(7).mean().values)
plt.xlabel('Days')
plt.ylabel('NUmber of Units sales')
plt.title('HOBBIES_1_002 item sales within time period with  days moving averages')
fig = plt.figure()
fig.set_figheight(40)
fig.set_figwidth(30)
plt.suptitle('HOBBIES_1_002 item sales within 10 stores with  days moving averages', fontsize=20)
for i in range(10):
    plt.subplot(10, 2, i+1)
    plt.plot(df2_subset.iloc[i, 6:].rolling(7).mean().values,label=df2_subset.iloc[i, 4])
    plt.legend()
    
plt.show()
abc = df2.groupby(['cat_id'])['dept_id'].count()
abc
plt.figure(figsize=(10,6))
abc.plot(kind='bar',color=['r', 'g', 'b'])
plt.title("Number of sales within Categories")
abc1 = df2.groupby(['dept_id'])['item_id'].count()
abc1
plt.figure(figsize=(10,6))
abc1.plot(kind='bar',color=['r', 'g', 'b','y','c', 'm','k'])
plt.title("Number of sales within Department")
df1
group2 = df1.groupby(['store_id'],as_index=False)['sell_price'].sum()
group2
sns.catplot(x='store_id',y='sell_price', kind='bar', data=group2,height=6, aspect=2)
plt.xlabel('Store ID')
plt.ylabel('Total Sales')
plt.title("Distribution of total sales within stores")
group3 = group2.groupby(group2['store_id'].str.contains('CA'))['sell_price'].sum()
group3
df1_subset2 = df1[df1['store_id'].str.contains('CA')]
df1_subset2
group4 = df1_subset2.groupby(['store_id'],as_index=False)['sell_price'].sum()
sns.catplot(x='store_id',y='sell_price', kind='bar', data=group4,height=5, aspect=2.5)
plt.xlabel('Store ID')
plt.ylabel('Total Sales')
plt.title("Distribution of total sales within CA state")
df1_subset = df1[df1['store_id'] == 'CA_1']
df1_subset
qqq=df1_subset.loc[df1_subset['item_id'].str.contains('HOBBIES_1')]
qqq
grouped = qqq.groupby(['wm_yr_wk'])['sell_price'].mean()
fig = plt.figure()
fig.set_figheight(12)
fig.set_figwidth(20)
plt.ylabel('Mean Sales')
plt.suptitle('Mean price variasion within Department on CA_1 Store', fontsize=20)
for i in range(1,3,1):
    #plt.subplot(1, 2, 1)
    qqq=df1_subset.loc[df1_subset['item_id'].str.contains('HOBBIES_' + str(i))]
    grouped = qqq.groupby(['wm_yr_wk'])['sell_price'].mean()
    grouped.plot(label='HOBBIES_' + str(i))
    plt.legend()  
for i in range(1,4,1):   
    qqq=df1_subset.loc[df1_subset['item_id'].str.contains('FOODS_' + str(i))]
    grouped = qqq.groupby(['wm_yr_wk'])['sell_price'].mean()
    grouped.plot(label='FOODS_' + str(i))
    plt.legend() 
for i in range(1,3,1):
    qqq=df1_subset.loc[df1_subset['item_id'].str.contains('HOUSEHOLD_' + str(i))]
    grouped = qqq.groupby(['wm_yr_wk'])['sell_price'].mean()
    grouped.plot(label='HOUSEHOLD_' + str(i))
    plt.legend()
plt.show()
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(20)
plt.suptitle('HOUSEHOLD item sels variation with moving averages', fontsize=20)
for i in range(1,3,1):
    #plt.subplot(1, 2, i)
    qqq=df1_subset.loc[df1_subset['item_id'].str.contains('HOUSEHOLD_' + str(i))]
    grouped = qqq.groupby(['wm_yr_wk'])['sell_price'].mean()
    grouped.plot()
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('mean seles')
    
    
plt.show()
df3.shape
ff38=df3.iloc[0:1913, :]
ff38.shape
ff8=df2.iloc[:, 6:]
ff8
new_df2 = ff8.transpose()
new_df2
ff38.shape
new_df2.shape
df996 = new_df2.reset_index()
df996
del df996['index']
df996
df_col = pd.concat([df996,ff38.date,ff38.weekday,ff38.year], axis=1)
df_col
rows = []
for i in range(0,1913,1):
     rows.append(df_col.iloc[i,:30490].sum())
#print(rows)
dftt = pd.DataFrame(rows, columns=["sum"])
dftt
gh=df_col.iloc[0:1913,30491:30493]
gh
df_col1 = pd.concat([dftt,gh], axis=1)
df_col1
calender5 = df_col1.groupby(['weekday'],as_index=False)['sum'].sum()
calender5
plt.figure(figsize=(15,6))
ax = sns.lineplot(x="weekday", y="sum", data=calender5,color="coral", label="no of units").set_title('Sales by day')
sns.catplot(x='year', kind='count',palette="ch:.25", data=df_col,height=5, aspect=2)
plt.title("Sales by year")
del df_col['weekday']
del df_col['year']
df_col
chan = df_col.columns.tolist() #Get "Date" variable infront as 1st column
chan = chan[-1:] + chan[:-1]
df_col_n = df_col[chan]
df_col_n
df9=df_col_n.iloc[:, :1501] #I used first 1000 items and avalable all 1913 days to the prediction
df9
df9.dtypes
df9['date'] = pd.to_datetime(df9.date) #Convert the "date" variable as datetime because its intial data type is object
data = df9.drop(['date'], axis=1)        
data.index = df9.date                   #Convert date variable as index of the dataframe
data
df9.dtypes
cols = data.columns 
from statsmodels.tsa.vector_ar.var_model import VAR
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1941)
print(yhat)
pred = pd.DataFrame(index=range(0,len(yhat)),columns=[cols]) 
for j in range(0,1500):
    for i in range(1913,1941):
       pred.iloc[i][j] = yhat[i][j]
pred
ffinal=pred.iloc[1913:1941, 0:1500]
ffinal
final_sum = ffinal.transpose() #Convert the output data frame inta standard submission format
final_sum
final_sum.to_csv("submission4.csv", index=False)