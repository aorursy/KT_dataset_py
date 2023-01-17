# Import required librarriws

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime

import seaborn as sns
#read the data 

db=pd.read_csv("../input/avocado-prices/avocado.csv")

#set Date as date time -----set Date as the index (not used -but good alternative)

db2=pd.read_csv("../input/avocado-prices/avocado.csv",parse_dates=True,index_col="Date")

db.head()
#check the size, dimension , shape and overall stat 

db2.head()
db.shape
db.size

db.describe()
db.info()

# unnamed should be deleted 

db.dtypes

# all integer and float

# excpet : Date (should chnage to date /time ------region as string / categorical----type as string (categorical))

db.isnull().any().any()

# there is no null value
#years in the report 

db.year.unique()
# regions in the data

db.region.unique()
# region is mix of state , city and divison .for comparison it needs to be seperated.also it has total U.S a sone region .

# I looked up this on Hass website, and it seems that they divided us in  some division  :california, west, south central

#south east,mid south, great lakes and north east .

# each divison  cover some city (maybe distrubution center or ...)

# also there is one region as total U.S.

np.count_nonzero(db.region.unique())

db.region.value_counts()

# report is for eaxh state 
# What is unnamed columns for ? I thought it is region code  . but it is not . 

db["Unnamed: 0"].value_counts()
# check only one  value for unnamed .no insigh

db[db["Unnamed: 0"]==7]
db.year.value_counts()

# it is 4 years 
#convert date to datetime format 

# it seems it is  weekly report

db.Date=pd.to_datetime(db.Date)
db.dtypes
db.columns

# check the columns
# set index ad date 

db=db.set_index("Date")

db.head()
db.index
(db['4046']+db['4225']+db['4770']==db['Total Volume']).sum()

# only in 10  record sum of Avocado sales equals to 3 labels.probably there should be more avocado brand

#
(db['Small Bags']+db['Large Bags']+db['XLarge Bags']==db['Total Bags']).sum()

# for 14000 record total bags wulas to small+large +x large 

# for about 4000 record it dose not match.maybe there is another way to seel the avocado 
(db['4046']+db['4225']+db['4770']+db['Total Bags']==db['Total Volume']).sum()
# probabaly there is  another kind of vocoda or another producer 

(db['4046']+db['4225']+db['4770']+db['Total Bags']-db['Total Volume'])
# we have 2 type of Avovado oragnic and conventional/ and we have 3 code avocado information.

db.type.unique()
db.type.value_counts()
# make different Data frmae based on type 

db_conventional=db.loc[db.type=='conventional'].copy()

db_organic=db.loc[db.type=='organic'].copy()
# make different Data frmae based on type  and region 

db_regional_con=db_conventional.loc[db_conventional.region.isin(['Southeast','GreatLakes','West','SouthCentral','Midsouth','California','Northeast']) ]

db_regional_org=db_organic.loc[db_organic.region.isin(['Southeast','GreatLakes','West','SouthCentral','Midsouth','California','Northeast']) ]
# make different Data frmae based on type  and region (only total us)

db_total_us_con=db_conventional.loc[db_conventional.region.isin(['TotalUS'])]

db_total_us_org=db_organic.loc[db_organic.region.isin(['TotalUS'])]
# if we have a column with avocado type (4046,4225,4770) and then amount sold  , it was easier to aggregate

# same as having the column bag type (small, Large, X large)
# I am wondering if there is better way to do this .

#db_city_con(only cities /type=not organic(conv))

db_city_con=db[~db.region.isin(db_regional_con.region)].copy()

db_city_con=db_city_con.loc[db_city_con.region!='TotalUS']

db_city_con
#db_city_con(only cities /type=organic)



db_city_org=db[~db.region.isin(db_regional_org.region)].copy()

db_city_org=db_city_org.loc[db_city_org.region!='TotalUS']

db_city_org
#question :    correlatiob between price and volume ?(price and demand )

db_city_con.AveragePrice.corr(db_city_con['Total Volume'] )  
#correlation matrix

db_city_con.corr()
db_regional_con.index.year
db_total_us_con['Total Volume'].plot(linewidth=2);
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# line plot , using seaborn 

g=sns.relplot(data=db_total_us_con['Total Volume'],kind="line")

g.set_xticklabels(rotation=30)
# there is spike in price at 2017

db_total_us_con['AveragePrice'].plot(linewidth=2);
ax = db_total_us_con.loc['2017', 'AveragePrice'].plot()

ax.set_ylabel('weekly avocado average price ');
# same cycle for 2015 and 2016 .2015 is sort of flat 

# there is like a cycle for price for each year .

ax = db_total_us_con.loc['2018', 'AveragePrice'].plot()

ax = db_total_us_con.loc['2017', 'AveragePrice'].plot()

ax = db_total_us_con.loc['2016', 'AveragePrice'].plot()

ax = db_total_us_con.loc['2015', 'AveragePrice'].plot()



ax.set_ylabel('weekly avocado average price ');

# only jan 2017 as example 

ax = db_total_us_con.loc['2017-01':'2017-02', 'AveragePrice'].plot(marker='o', linestyle='-')

ax.set_ylabel('weekly avocado average price');
# only jan/feb  2017 as example  for price , better graph

import matplotlib.dates as mdates

fig, ax = plt.subplots()

ax.plot(db_total_us_con.loc['2017-01':'2017-02', 'AveragePrice'], marker='o', linestyle='-')

ax.set_ylabel('Daily AveragePrice')

ax.set_title('Jan-Feb 2017 avocado average price ')

# Set x-axis major ticks to weekly interval, on Mondays

ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))

# Format x-tick labels as 3-letter month name and day number

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));
db_total_us_con.index.year
# for total us :



#box plot for price for months .It shows price is higher in month 9 and 10. and lowest on 1,2

sns.boxplot(x=db_total_us_con.index.month, y=db_total_us_con['AveragePrice'], )

ax.set_ylabel('average price ')

ax.set_title('average price per month')





    

    

   
#box plot for total volume  for months .It shows price volume is higher  in month 1 and 2. and lowest on 10,11

# it is exactly opposite of the price . 

# when supply is high , price will go down and when thesupply is low, price will go up.



sns.boxplot(x=db_total_us_con.index.month, y=db_total_us_con['Total Volume'])

ax.set_ylabel('total volume')

ax.set_title('total volume per month')
db_total_us_con.hist(['Total Volume'])

db_total_us_con.hist(['AveragePrice'])
# resampling , monthly 

# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)

data_columns = ['AveragePrice', 'Total Volume']

# Resample to weekly frequency, aggregating with mean

usa_con_monthly_mean = db_total_us_con[data_columns].resample('M').mean()

usa_con_monthly_mean.head(3)
# monthly resampling graph/average price 

plt.rcParams["figure.figsize"] = (30,10)

# Start and end of the date range to extract

start, end = '2015-01', '2018-03'

# Plot daily and weekly resampled time series together

fig, ax = plt.subplots()





ax.plot(db_total_us_con.loc[start:end, ['AveragePrice']],

marker='.', linestyle='-', linewidth=0.5, label='Weekly')



ax.plot(usa_con_monthly_mean.loc[start:end, ['AveragePrice']],

marker='o', markersize=8, linestyle='-', label='monthly Mean Resample')

ax.set_ylabel('average price',fontsize=18)





ax.legend();
# monthly resampling graph/total volume 

plt.rcParams["figure.figsize"] = (30,10)

# Start and end of the date range to extract

start, end = '2015-01', '2018-03'

# Plot daily and weekly resampled time series together

fig, ax = plt.subplots()





ax.plot(db_total_us_con.loc[start:end, ['Total Volume']],

marker='.', linestyle='-', linewidth=0.5, label='Weekly')



ax.plot(usa_con_monthly_mean.loc[start:end, ['Total Volume']],

marker='o', markersize=8, linestyle='-', label='monthly Mean Resample')

ax.set_ylabel('average price',fontsize=18)





ax.legend();
min(db.index),max(db.index)
# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)

data_columns = ['4046', '4225','4770']

# Resample to weekly frequency, aggregating with mean

usa_con_monthly_mean = db_total_us_con[data_columns].resample('M').mean()

usa_con_monthly_mean.head(3)
# 2 plu(medium and large with seasonal up and down , but small one fixed lower volume 



a=plt.plot(usa_con_monthly_mean.index,usa_con_monthly_mean["4046"],label='4046')

b=plt.plot(usa_con_monthly_mean.index,usa_con_monthly_mean["4225"],label='4225')

c=plt.plot(usa_con_monthly_mean.index,usa_con_monthly_mean["4770"],label='4770')

plt.legend()



db_total_us_con.head()
#7 day rolling price to smoth the curve 

data_columns = ['AveragePrice', 'Total Volume']

us_con_7d = db_total_us_con[data_columns].rolling(7, center=True).mean()

us_con_7d.head()
#365 day rolling price to smoth the curve 

us_con_365d = db_total_us_con[data_columns].rolling(window=365, center=True, min_periods=360).mean()

us_con_365d.head()
# Plot daily, 7-day rolling mean, and 365-day rolling mean time series

fig, ax = plt.subplots()

ax.plot(db_total_us_con['Total Volume'], marker='.', markersize=2, color='0.6',

linestyle='None', label='Daily')

ax.plot(us_con_7d['Total Volume'], linewidth=2, label='7-d Rolling Mean')

ax.plot(us_con_365d['Total Volume'], color='0.2', linewidth=3,

label='Trend (365-d Rolling Mean)')

#Set x-ticks to yearly interval and add legend and labels

ax.xaxis.set_major_locator(mdates.YearLocator())

ax.legend()

ax.set_xlabel('Year')

ax.set_ylabel('Consumption (GWh)')

ax.set_title('total volume ');
y_r = db_regional_con.groupby([pd.Grouper(freq="Y"),"region"])["Total Volume"].sum() 

y_r_df=y_r.to_frame().reset_index()

y_r
m_r = db_regional_con.groupby([pd.Grouper(freq="M"),"region"])["Total Volume"].sum() 
m_r.head()
m_r_df=m_r.to_frame().reset_index()

m_r_df.head(14)
m_r_df.columns

m_r_df.index
# I do not know how to add region as index when resampling 

m=db_regional_con.resample("M")["Total Volume"].sum()

y=db_regional_con.resample("Y")["Total Volume"].sum()


y
GB=db_regional_con.groupby([(db_regional_con.index.year),(db_regional_con.index.month)])["Total Volume"].sum()
GB
db_regional_con['month']=db_regional_con.index.month
db_regional_con.head()
# category plot  for the 7 region   weekly sales volume 

g=sns.catplot(x='region',y='Total Volume',data=db_regional_con)

g.set_xticklabels(rotation=30)
# box plot plot  for the 7 region   weekly sales volume 

g=sns.catplot(x='region',y='Total Volume',data=db_regional_con,kind='box')

g.set_xticklabels(rotation=30)
# category plot  for the 7 region   weekly sales volume --year seperated



g=sns.catplot(x='region',y='Total Volume',data=db_regional_con,hue='year')

g.set_xticklabels(rotation=30)
# category plot  for the 7 region   weekly sales volume --year seperated-----swarm type



g=sns.catplot(x='region',y='Total Volume',data=db_regional_con,hue='year',kind="swarm")

g.set_xticklabels(rotation=30)
#box plot  for the 7 region   weekly sales volume --year seperated



g=sns.catplot(x='region',y='Total Volume',data=db_regional_con,hue='year',kind="box")

g.set_xticklabels(rotation=30)
# bar plot  for the 7 region   weekly sales volume --year seperated



g=sns.catplot(x='region',y='Total Volume',data=db_regional_con,hue='year',kind="bar")

g.set_xticklabels(rotation=30)
# bar plot  for the 7 region   weekly sales volume -



g=sns.catplot(x='region',y='Total Volume',data=db_regional_con,kind="bar")

g.set_xticklabels(rotation=30)
# bar plot  for the 7 region   weekly sales volume --year seperated--each year ine one column chart



g=sns.catplot(x='region',y='Total Volume',data=db_regional_con,col='year',kind="bar")

g.set_xticklabels(rotation=30)
# same as above . this time in rows 

g=sns.catplot(x='year',y='Total Volume',data=db_regional_con,row='region',kind="bar")

db_regional_con['Total Volume']
# category plot  for the 7 region   monthly sales volume 

g=sns.catplot(x='region',y='Total Volume',data=m_r_df)

g.set_xticklabels(rotation=30)
# box plot  for the 7 region   monthly sales volume 

g=sns.catplot(x='region',y='Total Volume',data=m_r_df,kind='box')

g.set_xticklabels(rotation=30)
# bar plot  for the 7 region   monthly sales volume 

g=sns.catplot(x='region',y='Total Volume',data=m_r_df,kind="bar")

g.set_xticklabels(rotation=30)
# category plot  for the 7 region   yearly sales volume 

g=sns.catplot(x='region',y='Total Volume',data=y_r_df)

g.set_xticklabels(rotation=30)
# box plot  for the 7 region   yearly sales volume 

g=sns.catplot(x='region',y='Total Volume',data=y_r_df,kind='box')

g.set_xticklabels(rotation=30)
# bar plot  for the 7 region   yearly sale sales volume 

g=sns.catplot(x='region',y='Total Volume',data=y_r_df,kind="bar")

g.set_xticklabels(rotation=30)