# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 100) #To specify the max number of columns to be printed. 
            #Pandas defaults at <30 I guess. We've 85. Hence using this.
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from collections import Counter    

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_city = pd.read_csv('../input/City_time_series.csv',parse_dates=['Date'])
df_city.shape
Counter(df_city.dtypes.values)
df_city.head(5)
#missing data
total_null = df_city.isnull().sum()
percent = (total_null/df_city.isnull().count())*100
non_null = df_city.isnull().count() - total_null
missing = pd.concat([total_null,non_null, percent], axis=1, keys=['Total_null', 'Non_null','Percent'])
missing.sort_values(by='Percent',inplace=True,ascending=False)
print("No of variables with more than 99% data missing: ",(missing[missing.Percent > 99])['Percent'].count())
missing.transpose()

df_city.describe()
df_city.describe(exclude=[np.number])
df_city['Year'] = df_city['Date'].dt.year
df_city['Month'] = df_city['Date'].dt.month
Grouped_cities = df_city.groupby('RegionName')
sold_prices = Grouped_cities['MedianSoldPricePerSqft_AllHomes'].median()
print("Total Cities : ",sold_prices.size,"Null Values : ",sold_prices.isnull().sum())
sold_prices = sold_prices.dropna()
sns.distplot(sold_prices,kde=False)
sold_prices.describe()
sold_prices.sort_values(ascending=False)[:20]
#Checking how many rows are there contributing to median price of sandy_springmontgomerymd
df_city[df_city['RegionName'] == 'sandy_springmontgomerymd'].dropna(subset = ['MedianSoldPricePerSqft_AllHomes'])
sns.distplot(sold_prices[sold_prices.values<300],kde=False)
mapping = pd.cut(sold_prices[sold_prices.values<300],6,labels=range(6))
mapping.append(pd.Series(data = 6,index = sold_prices[sold_prices.values>300].keys()))
df_city['City_Tier_MedianSoldPricePerSqft'] = Grouped_cities['RegionName'].transform(lambda x:mapping[x])
fig, ax = plt.subplots(nrows = 3, ncols = 2 ,figsize=(20,16))
axarr = ax.flat;
fig.subplots_adjust(wspace=0.4, hspace=0.4)
#ax = list(chain.from_iterable(ax)) #Change ax from matrix to a list for iteration 
#for i in range(len(categ_vars)):
 #   sns.countplot(train[categ_vars[i]], hue=train['Survived'], ax=ax[i])
for title,group in df_city.groupby('City_Tier_MedianSoldPricePerSqft'):
    ZHVI_mean = group['ZHVI_AllHomes']
    group.plot(x='Date', y='ZHVI_AllHomes',title=title,ax=axarr[title]);  

fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(16,6))
group = df_city.groupby(['Year','City_Tier_MedianSoldPricePerSqft'])
median_ZHVI = group.median()['ZHVI_AllHomes'].unstack()
median_ZHVI.plot(ax=ax[0])
#plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.title('Median ZHVI_AllHomes')

median_ZRI = group.median()['ZRI_AllHomes'].unstack()
median_ZRI.plot(ax=ax[1])
#plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.title('Median ZRI_AllHomes')
sales_vol = df_city.RegionName.value_counts()
sns.distplot(sales_vol,kde=False)
mapping = pd.cut(sales_vol,4,labels=range(4))
df_city['City_Tier_SalesVol'] = Grouped_cities['RegionName'].transform(lambda x:mapping[x])
fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(16,6))
group = df_city.groupby(['Year','City_Tier_SalesVol'])
median_ZHVI = group.median()['ZHVI_AllHomes'].unstack()
median_ZHVI.plot(ax=ax[0])
#plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.title('Median ZHVI_AllHomes')

median_ZRI = group.median()['ZRI_AllHomes'].unstack()
median_ZRI.plot(ax=ax[1])
#plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.title('Median ZRI_AllHomes')
df_city['Quarter'] = df_city['Month']//4 + 1
group = df_city.groupby(['Year','Quarter']).count()['RegionName'].unstack()
group.plot()
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.title('Quarterwise Sales')
buying_margin = df_city['MedianListingPrice_AllHomes'] - df_city['MedianSoldPrice_AllHomes'];
buying_margin.describe()
buying_margin = buying_margin.dropna()
buying_margin = buying_margin[(buying_margin>0)&(buying_margin<buying_margin.quantile(0.95))]
sns.distplot(buying_margin,kde=False)
df_city_crosswalk = pd.read_csv('../input/cities_crosswalk.csv')
df_city_crosswalk.shape
df_city_crosswalk.head(5)
df_city_crosswalk.describe()
all_counties = df_city_crosswalk.County.value_counts()
sns.distplot(all_counties,kde=False)
print(all_counties.describe())
print("No. of counties with only one city : ",(all_counties==1).sum())
print("Counties with max cities:",all_counties[:10])
all_cities = df_city_crosswalk.City.value_counts()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(16,5))
sns.distplot(all_cities,kde=False,ax=ax1)
print(all_cities.describe())
print("No. of cities in more than one county : ",(all_cities>1).sum())
all_cities_mult = all_cities[all_cities>1]
sns.distplot(all_cities_mult,kde=False,ax=ax2)
print("Cities shared between more counties:",all_cities_mult[:10])
print(all_cities_mult.describe())

df_county_crosswalk = pd.read_csv('../input/CountyCrossWalk_Zillow.csv')
df_county_crosswalk.head(5)
print("No. of counties in crosswalk :",len(df_county_crosswalk.CountyRegionID_Zillow.unique()))
df_county = pd.read_csv('../input/County_time_series.csv',parse_dates=['Date'])
df_county.shape
df_county.describe(exclude=[np.number])
print("Extra columns in County TS: ", set(list(df_county.columns))-set(list(df_city.columns)))
df_county.head(5)
import re
#Will return true if alphabet is present
f = lambda x: re.search('[^0-9]', x) is not None
false_count = df_county.drop_duplicates('RegionName')['RegionName'].astype(str).apply(f)
print(false_count)
x[:4]
#Cleaning the RegionName
a = pd.Series(df_county['RegionName'].unique())
a = a.convert_objects(convert_numeric=True)
print(a[a!=a])
pd.Series(df_county['RegionName'].unique())[5437]
#Cleaning the RegionName
(df_county['RegionName']=='United_States').sum()
mapping = pd.Series(index=df_county_crosswalk.CountyRegionID_Zillow,data=df_county_crosswalk.CountyName)

def apply_mapping(x):
    try : 
        return mapping[x];
    except:
        return -1
df_county['Region_Mapped'] = df_county['RegionName'].apply(lambda x:apply_mapping(x))
df_county['Region_Mapped'].value_counts()
p = df_county[~pd.isnull(df_county['SalesRaw_AllHomes'])]

p.Day.unique()
p.loc[p.Day==30, 'Month'].unique()

p = df_county[~pd.isnull(df_county['DaysOnZillow_AllHomes'])]
p.Day.unique()
p = df_county[~pd.isnull(df_county['SalesRaw_AllHomes']) & pd.isnull(df_county['DaysOnZillow_AllHomes']) ]
p.head()
#Extracting year from the date time column
df_county['Year'] = df_county['Date'].dt.year
df_county['Month'] = df_county['Date'].dt.month
df_county['Day'] = df_county['Date'].dt.day

#Creating a temporary dataframe to get Sales sum
tmp = df_county.loc[~pd.isnull(df_county.SalesRaw_AllHomes)]
reqd = tmp.groupby(['RegionName', 'Year'])['SalesRaw_AllHomes','DaysOnZillow_AllHomes'].agg({np.mean, \
                                                              np.sum}).reset_index()
reqd.columns = ['RegionName', 'Year','tot_sales','mean_sales','tot_days','mean_days']
reqd.sort_values(by = 'tot_sales', ascending= False, inplace=True)
reqd.head()

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(16,5))

print(reqd[reqd.RegionName == '04013'])
(reqd[reqd.RegionName == '04013']).sort_values(by='Year').plot(x='Year',y="tot_sales",ax=ax1)
(reqd[reqd.RegionName == '04013']).sort_values(by='Year').plot(x='Year',y="mean_days",ax=ax2)

#df_county.loc[df_county.RegionName == '04013', 'Region_Mapped']
df_state = pd.read_csv('../input/State_time_series.csv',parse_dates = ['Date'])
df_state.shape
print("Less columns in State TS: ", set(list(df_county.columns))-set(list(df_state.columns)))