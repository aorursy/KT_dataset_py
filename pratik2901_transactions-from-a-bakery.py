%load_ext autoreload
%autoreload 2

%matplotlib inline

import pandas as pd
from pandas_summary import DataFrameSummary
from IPython.display import display
from datetime import datetime

from fastai.imports import *
from fastai.structured import *
import seaborn as sns

sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=1.9,rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})              
plt.figure(figsize=(3.1, 3)) # Two column paper. Each column is about 3.15 inch wide.

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
trans = pd.read_csv("../input/BreadBasket_DMS.csv",parse_dates=['Date'])
display(trans.T)
display_all(trans.describe(include='all').T)
trans.info()
trans['Time'] =  pd.to_datetime(trans['Time'], format='%H:%M:%S')
trans.Item = trans.Item.astype('category')
trans.Item.value_counts()[:10]
trans = trans[trans.Item != 'NONE']
trans.Item.cat.categories
trans['Item_codes'] = trans.Item.cat.codes
display_all(trans.isnull().sum().sort_index()/len(trans))
trans.head(10)
def add_datetime_features(df):
    # sleep: 12-5, 6-9: breakfast, 10-14: lunch, 14-17: dinner prep, 17-21: dinner, 21-23: deserts!
    df['Time'] = pd.DatetimeIndex(df['Time']).time
    hour = df['Time'].apply(lambda ts: ts.hour)
    df['Hour'],df['Time_Of_Day'] = hour,hour
    df['Time_Of_Day'].replace([i for i in range(0,6)], 'Sleep',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(6,10)], 'Breakfast',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(10,14)], 'Lunch',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(14,17)], 'Dinner Prep',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(17,21)], 'Dinner',inplace=True)
    df['Time_Of_Day'].replace([i for i in range(21,24)], 'Deserts',inplace=True)
    df.drop('Time',axis=1,inplace=True)  

    
    df['Season'] = pd.DatetimeIndex(df['Date']).month
    df['Season'].replace([1,2,12], 'Winter',inplace=True)
    df['Season'].replace([i for i in range(3,6)], 'Spring',inplace=True)
    df['Season'].replace([i for i in range(6,9)], 'Summer',inplace=True)
    df['Season'].replace([i for i in range(9,12)], 'Fall',inplace=True) 
    
    add_datepart(df, 'Date')
    
    return df
trans = add_datetime_features(trans)
trans.head(10)
trans.pivot_table(index='Season',columns='Item', aggfunc={'Item':'count'}).fillna(0)
trans.pivot_table(index='Time_Of_Day',columns='Item', aggfunc={'Item':'count'}).fillna(0)
trans.pivot_table(index='Year',columns='Item', aggfunc={'Item':'count'}).fillna(0)
plt.figure(figsize=(20,10))
trans['Item'].value_counts()[:20].sort_values().plot.barh(title='Top 20 Sales',grid=True)
plt.figure(figsize=(20,10))
trans['Item'].value_counts()[-20:-1].sort_values().plot.barh(title='Top 20 Least Sales',grid=True)
df1=trans[['Transaction', 'Month', 'Year', 'Time_Of_Day','Dayofweek','Hour','Is_year_end','Is_year_start','Is_month_end','Is_month_start','Season']]
df1=df1.drop_duplicates()
plt.figure(figsize=(20,10))
sns.countplot(x='Hour',data=df1,hue='Time_Of_Day').set_title('General Transation Trend Throughout The Day',fontsize=25)
plt.figure(figsize=(20,10))
sns.countplot(x='Dayofweek',data=df1).set_title('Pattern of Transation Trend Throughout The Week',fontsize=25)
plt.figure(figsize=(20,10))
sns.countplot(x='Season',data=df1).set_title('Pattern of Transation Trend During Different Season\'s',fontsize=25)
plt.figure(figsize=(20,10))
sns.countplot(x='Year',data=df1,hue='Is_year_start').set_title('Transation Trend During Year Start',fontsize=25)
plt.figure(figsize=(20,10))
sns.countplot(x='Year',data=df1,hue='Is_year_end').set_title('Transation Trend During Year End',fontsize=25)
plt.figure(figsize=(20,10))
sns.countplot(x='Month',data=df1,hue='Is_month_start').set_title('Transation Trend During Month Start',fontsize=25)
plt.figure(figsize=(20,10))
sns.countplot(x='Month',data=df1,hue='Is_month_end').set_title('Transation Trend During Month End',fontsize=25)