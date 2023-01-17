import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')

%reload_ext autoreload
%autoreload 2

%matplotlib inline
sns.set(style='darkgrid')
pd.set_option('display.float_format', lambda x: '{:f}'.format(x))

PATH = './data/india_startup_funding/'

## Data Description
def describe(df):
    print('======================================')
    print('No. of Rows.:{0}\nNo. of Columns:{1}\n'.format(df.shape[0], df.shape[1]))
    print('======================================')
    data_type = DataFrame(df.dtypes, columns=['Data Type'])
    null_count =  DataFrame(df.isnull().sum(), columns=['Null Count'])
    not_null_count = DataFrame(df.notnull().sum(), columns=['Not Null Count'])
    unique_count = DataFrame(df.nunique(), columns=['Unique Count'])
    joined = data_type.merge(null_count, left_index=True, right_index=True)
    joined = joined.merge(not_null_count, left_index=True, right_index=True)
    joined = joined.merge(unique_count, left_index=True, right_index=True)
    display(joined)
    display(df.describe())
    return None

## Adding more time columns
def add_datepart(df, date_column):
    date_series = df[date_column]
    df[date_column] = pd.to_datetime(date_series, infer_datetime_format=True)
    for n in ('Year', 'Month', 'Week', 'Day', 'Weekday_Name', 'Dayofweek', 'Dayofyear'):
        df['Date'+'_'+n] = getattr(date_series.dt, n.lower())
## Code to draw histogram    
def histogram(x, bins=None, hist=True, kde=True, color='r', xlabel=None, ylabel='', title='', lower=-1*np.inf, upper=np.inf, figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.distplot(x.dropna()[(x > lower) & (x < upper)], bins=bins,
                 hist=hist, kde=kde, 
                 kde_kws={"shade": True}, 
                 color=color, axlabel=xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    return ax
date_parser = lambda date: pd.datetime.strptime(date, '%d/%m/%Y')

df = pd.read_csv("../input/startup_funding.csv", 
                 thousands=',')

df.drop(columns=['SNo'], axis=1, inplace=True)
df['Date'][df['Date']=='12/05.2015'] = '12/05/2015'
df['Date'][df['Date']=='13/04.2015'] = '13/04/2015'
df['Date'][df['Date']=='15/01.2015'] = '15/01/2015'
df['Date'][df['Date']=='22/01//2015'] = '22/01/2015'

df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y')

add_datepart(df, 'Date')
df['Month'+'_'+'Year'] = df['Date_Month'].astype('str') + '-' + df['Date_Year'].astype('str')
df['Date_quarter'] = (df.Date_Month-1)//3 + 1    

df.InvestmentType = df.InvestmentType.str.split(pat=" ").str.join(sep='').str.lower()
df.IndustryVertical = df.IndustryVertical.str.lower()
display(df.head())
describe(df)
plt.subplots(figsize=(15,5))
p1 = sns.countplot(x = 'CityLocation',
              data = df,
              order = df['CityLocation'].value_counts().index)

plt.xticks(rotation=90);
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Investment in USD over time', fontsize=24, fontweight='bold')
ax = fig.add_subplot(211)
ts_quarter = df.groupby([df['Date'].dt.year, df['Date'].dt.quarter]).agg({'AmountInUSD':'sum'})['AmountInUSD']
ts_quarter.plot(linewidth=4, marker="o", markersize=10, markerfacecolor='#E84855')
plt.ylabel('USD in Billions')
plt.xlabel('Year-Quarter No.')
# plt.title('Total Investment over time')

ax = fig.add_subplot(212)
ts_month = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).agg({'AmountInUSD':'sum'})['AmountInUSD']
ts_month.plot(linewidth=4, color='#F9DC57',marker="o", markersize=10, markerfacecolor='#E84855')
plt.ylabel('USD in Billions')
plt.xlabel('Year-Month');
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Investment Count over time', fontsize=24, fontweight='bold')
ax = fig.add_subplot(211)
ts_quarter_count = df.groupby([df['Date'].dt.year, df['Date'].dt.quarter]).size()#['AmountInUSD']
ts_quarter_count.plot(linewidth=4, color='#5F4B8B',marker="o", markersize=10, markerfacecolor='#E84855')
plt.ylabel('Number of Investments')
plt.xlabel('Year-Quarter');


ax = fig.add_subplot(212)
ts_month_count = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).size()#['AmountInUSD']
ts_month_count.plot(linewidth=4, color='#248A3B',marker="o", markersize=10, markerfacecolor='#E84855')
plt.ylabel('Number of Investments')
plt.xlabel('Year-Month');
# plt.
# bar plot for times series
# plt.subplots(figsize=(8,5))
# sns.countplot(x = 'Month_Year', data = df)
# plt.xticks(rotation=90);
temp = df.groupby(['Date_Weekday_Name']).agg({'Date':'count'}).reset_index()
sns.barplot(x=temp.Date_Weekday_Name, y=temp.Date, color='violet')
plt.xticks(rotation=30);
ax = histogram(df.AmountInUSD, upper=10**6, bins=30, color='#E84855')
ax.annotate('~ $500K', xy=(500000, 4.7e-6), xytext=(700000, 4e-6),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=15);
plt.title('Histogram');
df['rank'] = df.sort_values(['Date_Year', 'Date_Month'])\
               .groupby(['Date_Year', 'Date_Month'])['AmountInUSD'].rank(ascending=False)
temp = df[df['rank'] == 1].sort_values('Date')
sns.countplot(x=temp.Date_Weekday_Name)
plt.xticks(rotation=30);
temp = df.groupby(['InvestmentType']).agg({'AmountInUSD':'sum'}).reset_index()
temp.AmountInUSD = 100*temp.AmountInUSD/np.sum(temp.AmountInUSD)
g = sns.barplot(x=temp.InvestmentType, y=temp.AmountInUSD)
for index, row in temp.iterrows():
    g.text(row.name,row.AmountInUSD, round(row.AmountInUSD,2), color='black', ha="center")
plt.xticks(rotation=45);
temp = df.groupby(['IndustryVertical']).agg({'AmountInUSD':'sum'}).reset_index()\
  .sort_values('AmountInUSD', ascending=False).reset_index(drop=True).iloc[:40, ]
    
plt.subplots(figsize=(20,8))
sns.barplot(x=temp.IndustryVertical, y=temp.AmountInUSD)
plt.xticks(rotation=90);