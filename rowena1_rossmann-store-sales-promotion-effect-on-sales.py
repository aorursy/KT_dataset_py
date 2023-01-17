#Get tools

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import f_oneway

import sklearn

import datetime
#Get data

store_df=pd.read_csv('../input/store.csv')

train_df=pd.read_csv('../input/train.csv',parse_dates=True,low_memory=False,index_col='Date')
#Allow for data extraction

train_df['Year']=train_df.index.year

train_df['Month']=train_df.index.month

train_df['Week']=train_df.index.week

train_df['Day']=train_df.index.weekday_name



#Sales per customer helps retailers monitor

#the average purchase basket

train_df['SalesperCustomer']=train_df['Sales']/train_df['Customers']



#Concatenate store data with training data

#for extra information on stores

train_store=pd.merge(train_df,store_df,how='inner',on='Store')



train_store.head()



#There are 4 store types and 3 merchandise

#assortment types: a - basic; b - extra;

#c - extended.  They are distributed as

#follows.

store_df.groupby(['StoreType','Assortment']).Assortment.count()
store_open=train_store[(train_store.Open==1)]



saledaystype=store_open.groupby(['StoreType','Day']).sum()['Sales'].groupby(level=0).apply(lambda x:100*x/x.sum()).unstack().loc[:,[

    'Monday',

    'Tuesday',

    'Wednesday',

    'Thursday',

    'Friday',

    'Saturday',

    'Sunday']]



plt.figure(figsize=(10,10))



sns.heatmap(saledaystype,cmap='Blues').set_title('Day % of Weekly Sales by Store Type')

plt.show()
sns.catplot(data=store_open,x='Month',y='Customers',palette='colorblind',hue='Promo',kind='bar',height=5,

           aspect=8/5)

plt.title('Promotional Impact on Customer Volume')



sns.catplot(data=store_open, x='Month',y='Sales',palette='colorblind',hue='Promo',kind='bar',height=5,

           aspect=8/5)



plt.title('Promotional Impact on Sales Volume')

sns.catplot(data=store_open,x='Year',y='SalesperCustomer',col='StoreType',col_order=['a','b','c','d'],

            hue='Promo',palette='colorblind',col_wrap=2, kind='box')
sns.catplot(data=store_open,x='Year',y='Customers',col='StoreType',col_order=['a','b','c','d'],

            hue='Promo',palette='colorblind',col_wrap=2,kind='box')
store_open.groupby(['StoreType','Promo','Assortment']).SalesperCustomer.describe()

#Rossmann store promotions seem like

#weekly events that run from Monday

#through Friday.



store_open.groupby(['Day','Promo']).Promo.count()
#For stores that participate in promotional events,

#promotional impact by day of week.



sns.catplot(data=store_open,x='Day',y='SalesperCustomer',hue='Promo',kind='bar',

           height=5,aspect=8/5,palette='colorblind',order=['Monday','Tuesday','Wednesday',

                                                          'Thursday','Friday','Saturday',

                                                          'Sunday'])

plt.title('Promotional Impact on Average Spending Behaviour by Day')
sns.catplot(data=store_open,x='Day',y='Customers',hue='Promo',kind='bar',

           height=5,aspect=8/5,palette='colorblind',order=['Monday','Tuesday',

                                                           'Wednesday','Thursday',

                                                          'Friday','Saturday',

                                                          'Sunday'])

plt.title('Promotional Impact on Customer Volume by Day')
df=store_open[(store_open.Week>44)]

sns.catplot(data=df,x='Week',y='SalesperCustomer',col='StoreType',col_order=['a','b','c','d'],

            hue='Promo',color='StateHoliday',palette='colorblind', col_wrap=2,kind='bar')

          
sns.catplot(data=df,x='Week',y='Sales',col='StoreType',col_order=['a','b','c','d'],

            hue='Promo',color='StateHoliday',palette='colorblind',col_wrap=2, kind='bar')
#It appears that promotions do provide a lift to 

#sales.  Let's test to see if this is really the

#case and not due to chance market fluctuation.



df_promo=store_open[(store_open.Promo==1)]

df_nopromo=store_open[(store_open.Promo==0)]



#Look at the distribution of sales

f,axes=plt.subplots(ncols=2,figsize=(20,5))



sns.distplot(df_nopromo['Sales'],kde=False, color='orange',ax=axes[0])

axes[0].set_ylabel('Frequency')

axes[0].text(25000, 60000, r'$\mu$='+str(round(df_nopromo['Sales'].mean(),2)), 

         fontsize=12)

axes[0].set_title('Sales Distribution: Stores without Promotions')



sns.distplot(df_promo['Sales'],kde=False, color='orange',ax=axes[1])

axes[1].set_ylabel('Frequency')

axes[1].set_ylim([0,80000])

axes[1].text(25000,60000,r'$\mu$='+str(round(df_promo['Sales'].mean(),2)),

             fontsize=12)

axes[1].set_title('Sales Distribution: Stores with Promotions')

plt.show()
df_nopromo.Sales.describe()
df_promo.Sales.describe()
#Sales are not normally distributed but

#sample size is large.  Consider one-way ANOVA as it

#is a fairly robust test against normality

#assumption.

#H0: Means are equal (Promotions have no effect)

#H1: Means are not equal (Promotions have effect)



stat,p = f_oneway(df_nopromo.Sales,df_promo.Sales)

print('Statistic= %.3f, p= %.3f'%(stat,p))



#Interpret

alpha=0.05

if p>alpha:

    print('Cannot reject H0, i.e. means are not significantly different')

else:

    print('Reject H0, i.e. means are different.')