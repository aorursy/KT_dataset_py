import warnings

warnings.filterwarnings('ignore')



#importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

pd.set_option('display.float_format', '{:0.3f}'.format)
# Set ipython's max row display

pd.set_option('display.max_row', 500)



# Set iPython's max column width to 100

pd.set_option('display.max_columns', 100)
# Read the file

consumer = pd.read_csv('../input/eleckartdata/ConsumerElectronics.csv')
# Take a look at the first 5 rows

consumer.head()
# Get info about the dataset

consumer.info()
# Let's take a look at the statistical info of the dataset

consumer.describe(percentiles = [0.25, 0.5, 0.75, 0.90, 0.99, 0.999])
consumer.columns
consumer.replace(r'^\s+$', np.nan, regex=True, inplace = True)

consumer.replace('\\N', np.nan, inplace = True)
# let's check the null percentage for each column

round(100*(consumer.isnull().sum()/len(consumer.index)), 2)
#removing null valued GMV

consumer = consumer.loc[~(consumer.gmv.isnull())]
# let's check the null percentage for each column again

round(100*(consumer.isnull().sum()/len(consumer.index)), 2)
# Let's drop the rows that have product analytic vertical as null.

consumer = consumer[~pd.isnull(consumer.product_analytic_vertical)]
# Let's now check the product_analytic_super_category unique values

consumer.product_analytic_super_category.unique()
consumer.drop('product_analytic_super_category',1, inplace = True)
consumer.product_analytic_category.unique()
consumer.product_analytic_sub_category.unique()
#The three product sub categories for the MMM are - camera accessory, home audio and gaming accessory.

#Removing the rows with other sub categories



consumer = consumer.loc[(consumer.product_analytic_sub_category=='CameraAccessory') |

                       (consumer.product_analytic_sub_category=='GamingAccessory')|

                       (consumer.product_analytic_sub_category=='HomeAudio')]
consumer.product_analytic_vertical.unique()
#Let's convert the data type of GMV



consumer['gmv'] = pd.to_numeric(consumer['gmv'])
#Checking the minimum and maximum values of GMV

print(consumer.gmv.min())

print(consumer.gmv.max())
consumer[consumer.duplicated(['fsn_id','order_date','order_id','order_item_id',

                              'gmv','cust_id','pincode',

                              'product_analytic_category','product_analytic_sub_category',

                             'product_analytic_vertical'])]

#consumer.loc[consumer.duplicated()]
len(consumer[consumer.duplicated(['fsn_id','order_date','order_id','order_item_id',

                              'gmv','cust_id','pincode',

                              'product_analytic_category','product_analytic_sub_category',

                             'product_analytic_vertical'])])
#Removing duplicated values

consumer = consumer[~consumer.duplicated(['fsn_id','order_date','order_id','order_item_id',

                              'gmv','cust_id','pincode',

                              'product_analytic_category','product_analytic_sub_category',

                             'product_analytic_vertical'])]
consumer.loc[consumer.duplicated()]
#Checking nulls in gmv value

consumer.gmv.isnull().sum()
consumer.shape
# The columns deliverybdays and deliverycdays are populated with \N, which is incorrect.

# Let's replace them with null.

print(consumer.deliverybdays.value_counts().head())

print(consumer.deliverycdays.value_counts().head())
print(consumer.deliverybdays.isnull().sum()/len(consumer))

print(consumer.deliverycdays.isnull().sum()/len(consumer))
# We can drop delivercdays and deliverybdays column as it has 79% null values.

consumer.drop(['deliverybdays', 'deliverycdays'],1, inplace = True)
# Befor dealing with null values, let's first correct the data type of order_date

consumer['order_date'] = pd.to_datetime(consumer['order_date'])
# We now need to check if the dates are not outside July 2015 and June 2016.

consumer.loc[(consumer.order_date < '2015-07-01') | (consumer.order_date >= '2016-07-01')]
consumer = consumer.loc[(consumer.order_date >= '2015-07-01')]

consumer = consumer.loc[(consumer.order_date < '2016-07-01')]
#Changing the name of the column s1_fact.order_payment_type

consumer.rename(columns={'s1_fact.order_payment_type':'order_payment_type'}, inplace=True)
consumer.order_payment_type.value_counts()
#Converting the datatype

consumer['pincode'] = pd.to_numeric(consumer['pincode'])
#Let's see the values of pincode field

consumer.pincode.min()
consumer.pincode.isnull().sum()
# Before handling null values, there are negative values for pincode which we need to handle.

# Let's make all the negative values as positive.

consumer.pincode = consumer.pincode.abs()
# Let's now check the frequency of pincodes to decide whether we can impute the missing pincodes with the highest frequency one.

consumer.pincode.value_counts()
#pincode and cust_id doesn't seem to be of any use
consumer.drop(['cust_id','pincode'], axis = 1, inplace = True)
consumer[(consumer.product_mrp == 0)].head()
len(consumer[(consumer.product_mrp == 0)])
#Removing values with 0 MRP, since that is not possible at all

consumer = consumer.loc[~(consumer.product_mrp==0)]
consumer['gmv_per_unit'] = consumer.gmv/consumer.units
#Replacing the values of MRP with GMV per unit where the values of GMV/unit is greater than MRP

consumer['product_mrp'].loc[consumer.gmv_per_unit>consumer.product_mrp] = consumer['gmv_per_unit']
consumer.loc[consumer.gmv_per_unit>consumer.product_mrp]
consumer.drop(['gmv_per_unit'],1,inplace=True)
consumer.shape
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

sns.boxplot(y=consumer.sla, palette=("cubehelix"))



plt.subplot(1,2,2)

sns.boxplot(y=consumer.product_procurement_sla, palette=("cubehelix"))
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

sns.distplot(consumer.sla)



plt.subplot(1,2,2)

sns.distplot(consumer.product_procurement_sla)
consumer.sla.describe(percentiles=[0.0,0.25,0.5,0.75,0.9,0.95,0.99,1.0])
consumer.product_procurement_sla.describe(percentiles=[0.0,0.25,0.5,0.75,0.9,0.95,0.99,1.0])
#Converting negative values to the positive

len(consumer.loc[consumer.product_procurement_sla<0])
consumer.product_procurement_sla = abs(consumer.product_procurement_sla)
consumer.sla.std()
#Taking three sigma values for outliers treatment

print(consumer.sla.mean()+(3*(consumer.sla.std())))

print(consumer.sla.mean()-(3*(consumer.sla.std())))
consumer.product_procurement_sla.std()
#Taking three sigma values for outliers treatment

print(consumer.product_procurement_sla.mean()+(3*(consumer.product_procurement_sla.std())))

print(consumer.product_procurement_sla.mean()-(3*(consumer.product_procurement_sla.std())))
# Capping the values at three sigma value

len(consumer[consumer.sla > 14])
# Let's cap the SLAs.

consumer.loc[consumer.sla > 14,'sla'] = 14
# Similarly, the min value of product procurement sla is 0 and the max value is 15. However, three sigma value is 7. 

print(len(consumer[consumer.product_procurement_sla > 7]))
# Let's cap the product procuremtn SLAs.

consumer.loc[consumer.product_procurement_sla > 7,'product_procurement_sla'] = 7
consumer.shape
consumer.loc[consumer.duplicated()]
len(consumer[consumer.duplicated(['order_id','order_item_id'])])
consumer = consumer[~consumer.duplicated(['order_id','order_item_id'])]
consumer.describe()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

sns.distplot(consumer.gmv)



plt.subplot(1,2,2)

sns.distplot(consumer.product_mrp)



plt.show()
#2. gmv (Gross Merchendising Value - The cost price at which the item is sold multiplied by number of units)



# Let's derive listing price, which is nothing but gmv/units



consumer['listing_price'] = round((consumer.gmv/consumer.units),2)
#Let's check if there are any rows with listing price > MRP



len(consumer.loc[consumer.listing_price>consumer.product_mrp])
# Let's now calculate the discount %, which is nothing but (mrp-list price)/mrp

consumer['discount'] = round(((consumer.product_mrp - consumer.listing_price)/(consumer.product_mrp)),2)
consumer['discount'].describe()
consumer['Order_Item_Value'] = consumer['product_mrp'] * consumer['units']
# We can create the week number

consumer['week'] = np.where(consumer.Year == 2015, (consumer.order_date.dt.week - pd.to_datetime('2015-07-01').week + 1), consumer.order_date.dt.week+27)



# Dates like 2016-01-01 will be 53rd week as per ISO standard, hence the week value would be 53+27=80.

# We can make those values as week 27

consumer.week.values[(consumer.Year == 2016) & (consumer.week == 80)] = 27
### Prepaid = '1' or COD = '0'

consumer['order_payment_type'] = np.where(consumer['order_payment_type'] == "Prepaid",1,0)
### Creating Calendar for the period

calendar = pd.DataFrame(pd.date_range('2015-07-01','2016-06-30').tolist(), columns = ['Date'])

### Mapping week in the calendar

calendar['week'] = calendar.Date.dt.week

### Jan 2016 should be week 54 ,not week 1.

calendar['week'] = np.where((calendar['week'] <= 26) & (calendar.Date.dt.year == 2016), calendar['week']+53, calendar['week'])

### Special Sales List



special_sales_list = ["2015-07-18","2015-07-19","2015-08-15","2015-08-16","2015-08-17","2015-08-28","2015-08-29",

                      "2015-08-30","2015-10-15","2015-10-16","2015-10-17","2015-11-07","2015-11-08","2015-11-09",

                      "2015-11-10","2015-11-11","2015-11-12","2015-11-13","2015-11-14","2015-12-25","2015-12-26",

                      "2015-12-27","2015-12-28","2015-12-29","2015-12-30","2015-12-31","2016-01-01","2016-01-02",

                      "2016-01-03","2016-01-20","2016-01-21","2016-01-22","2016-02-01","2016-02-02","2016-02-14",

                      "2016-02-15","2016-02-20","2016-02-21","2016-03-07","2016-03-08","2016-03-09","2016-05-25",

                      "2016-05-26","2016-05-27"]



ss_list = pd.DataFrame(special_sales_list,columns = ['Date'])

ss_list['Date'] = pd.to_datetime(ss_list['Date'])

ss_list['Special_sales'] = True
calendar = calendar.merge(ss_list, 'left')

calendar.fillna(False, inplace = True)
calendar['Special_sales'] = calendar['Special_sales'].astype(int)
calendar.head()
calendar['Payday'] = ((calendar['Date'].dt.day == 1) | (calendar['Date'].dt.day == 15)).astype(int)
### Ontario Climate data of year 2015-2016 

ontario_climate_2015 = pd.DataFrame(pd.read_csv('../input/eleckartdata/ONTARIO-2015.csv',encoding="ISO-8859-1",skiprows=24))

ontario_climate_2016 = pd.DataFrame(pd.read_csv('../input/eleckartdata/ONTARIO-2016.csv',encoding="ISO-8859-1",skiprows=24))
### Merge Calendar with dataset on week



ontario_climate = ontario_climate_2015.append(ontario_climate_2016)

ontario_climate = ontario_climate.reset_index()

ontario_climate.head()
### Checking for any nan values



round((ontario_climate.isnull().sum()/len(ontario_climate.index))*100,2)
### Dropping columns we do not require in the analysis.

ontario_climate.drop(['index','Data Quality','Max Temp Flag','Min Temp Flag','Mean Temp Flag',

                      'Heat Deg Days Flag','Cool Deg Days Flag','Total Rain Flag','Total Snow Flag',

                      'Total Precip Flag','Snow on Grnd Flag','Dir of Max Gust (10s deg)','Dir of Max Gust Flag',

                      'Spd of Max Gust (km/h)','Spd of Max Gust Flag'], axis = 1, inplace = True)
ontario_climate.columns = ['Date','Year','Month','Day','max_temp_C','min_temp_C','mean_temp_C','heat_deg_days','cool_deg_days','total_rain_mm','total_snow_cm','total_precip_mm','snow_on_grnd_cm']
ontario_climate['Date'] = ontario_climate['Date'].apply(pd.to_datetime)
### Keeping Climate data from July 15 to June 16



ontario_climate=ontario_climate[(ontario_climate['Month'] >= 7) & (ontario_climate['Year'] == 2015) 

                               |(ontario_climate['Month'] <= 6) & (ontario_climate['Year'] == 2016)]
### Mapping week in the Climate data

ontario_climate['week'] = ontario_climate.Date.dt.week



### Jan 2016 should be week 54 ,not week 1.

ontario_climate['week'] = np.where((ontario_climate['week'] <= 26) & (ontario_climate['Year'] == 2016), ontario_climate['week']+53, ontario_climate['week'])



ontario_climate = ontario_climate.reset_index()

ontario_climate.drop('index',axis=1,inplace=True)

ontario_climate.head()
### Checking for any nan values



round((ontario_climate.isnull().sum()/len(ontario_climate.index))*100,2)
### Replacing Nan with mean value

ontario_climate['max_temp_C'] = ontario_climate['max_temp_C'].fillna(ontario_climate['max_temp_C'].mean())

ontario_climate['min_temp_C'] = ontario_climate['min_temp_C'].fillna(ontario_climate['min_temp_C'].mean())

ontario_climate['mean_temp_C'] = ontario_climate['mean_temp_C'].fillna(ontario_climate['mean_temp_C'].mean())

ontario_climate['heat_deg_days'] = ontario_climate['heat_deg_days'].fillna(ontario_climate['heat_deg_days'].mean())

ontario_climate['cool_deg_days'] = ontario_climate['cool_deg_days'].fillna(ontario_climate['cool_deg_days'].mean())

ontario_climate['total_rain_mm'] = ontario_climate['total_rain_mm'].fillna(ontario_climate['total_rain_mm'].mean())

ontario_climate['total_snow_cm'] = ontario_climate['total_snow_cm'].fillna(ontario_climate['total_snow_cm'].mean())

ontario_climate['total_precip_mm'] = ontario_climate['total_precip_mm'].fillna(ontario_climate['total_precip_mm'].mean())

ontario_climate['snow_on_grnd_cm'] = ontario_climate['snow_on_grnd_cm'].fillna(ontario_climate['snow_on_grnd_cm'].mean())

ontario_climate.head()
nps_score = pd.read_excel("../input/eleckartdata/Media data and other information.xlsx", sheet_name='Monthly NPS Score', skiprows=1)
### Transforming NPS and Stock_index

nps_score = nps_score.T.reset_index(drop=True)

nps_score.columns = ['NPS','Stock_Index']

nps_score = nps_score.drop(nps_score.index[[0]]).reset_index(drop=True)
### Adding Month and Year

nps_score['Month'] = pd.Series([7,8,9,10,11,12,1,2,3,4,5,6])

nps_score['Year'] = pd.Series([2015,2015,2015,2015,2015,2015,2016,2016,2016,2016,2016,2016])
nps_score['NPS'] = nps_score['NPS'].astype(float)

nps_score['Stock_Index'] = nps_score['Stock_Index'].astype(float)
nps_score.head()
calendar = calendar.merge(ontario_climate, 'left')
calendar = calendar.merge(nps_score, 'left')
# We can create the week number

calendar['week'] = np.where(calendar.Date.dt.year == 2015, (calendar.Date.dt.week - pd.to_datetime('2015-07-01').week + 1), calendar.Date.dt.week+27)



# Dates like 2016-01-01 will be 53rd week as per ISO standard, hence the week value would be 53+27=80.

# We can make those values as week 27

calendar.week.values[(calendar.Date.dt.year == 2016) & (calendar.week == 80)] = 27
calendar.head()
calendar = pd.DataFrame(calendar.groupby('week').agg({'NPS':'mean','Stock_Index':'mean',

                                                             'Special_sales':'mean','Payday':'mean',

                                                             'max_temp_C':'mean','min_temp_C':'mean',

                                                             'mean_temp_C':'mean','heat_deg_days':'mean',

                                                             'cool_deg_days':'mean','total_rain_mm':'mean',

                                                             'total_snow_cm':'mean','total_precip_mm':'mean',

                                                             'snow_on_grnd_cm':'mean'}))
calendar.reset_index(inplace = True)
calendar.head()
### Marketing Investment Data

marketing = pd.read_excel("../input/eleckartdata/Media data and other information.xlsx", sheet_name='Media Investment', skipfooter = 4, skiprows=2)
marketing.drop('Unnamed: 0', axis = 1, inplace = True)

marketing.replace(np.nan,0,inplace = True)

marketing['Date'] = pd.to_datetime(marketing[['Year', 'Month']].assign(DAY=1))

marketing.set_index('Date', inplace = True)

marketing
### Renaming the columns



marketing.columns = ['Year','Month','Total_Investment','TV','Digital','Sponsorship','Content_marketing',

                     'Online_marketing','Affiliates','SEM','Radio','Other']
### convert to datetimeindex

marketing.index = pd.to_datetime(marketing.index)
marketing
### add new next month for correct resample

idx = marketing.index[-1] + pd.offsets.MonthBegin(1)

idx
marketing = marketing.append(marketing.iloc[[-1]].rename({marketing.index[-1]: idx}))

marketing
#Resampling the data on weekly frequency

marketing = marketing.resample('W').ffill().iloc[:-1]

marketing
### divide by size of months

marketing['Total_Investment'] /= marketing.resample('MS')['Total_Investment'].transform('size')

marketing['TV'] /= marketing.resample('MS')['TV'].transform('size')

marketing['Digital'] /= marketing.resample('MS')['Digital'].transform('size')

marketing['Sponsorship'] /= marketing.resample('MS')['Sponsorship'].transform('size')

marketing['Content_marketing'] /= marketing.resample('MS')['Content_marketing'].transform('size')

marketing['Online_marketing'] /= marketing.resample('MS')['Online_marketing'].transform('size')

marketing['Affiliates'] /= marketing.resample('MS')['Affiliates'].transform('size')

marketing['SEM'] /= marketing.resample('MS')['SEM'].transform('size')

marketing['Radio'] /= marketing.resample('MS')['Radio'].transform('size')

marketing['Other'] /= marketing.resample('MS')['Other'].transform('size')
marketing.head()
marketing.reset_index(inplace = True)



###  Mapping week in the marketing



marketing['Date'] = pd.to_datetime(marketing['Date'])

# We can create the week number

marketing['week'] = np.where(marketing.Date.dt.year == 2015, (marketing.Date.dt.week - pd.to_datetime('2015-07-01').week + 1), marketing.Date.dt.week+27)



marketing.week.values[(marketing.Date.dt.year == 2016) & (marketing.week == 80)] = 27

marketing.sort_values('week', inplace = True)
marketing.head()
def adstocked_advertising(adstock_rate=0.5, advertising = marketing):

    

    adstocked_advertising = []

    for i in range(len(advertising)):

        if i == 0: 

            adstocked_advertising.append(advertising.iloc[i])

        else:

            adstocked_advertising.append(advertising.iloc[i] + adstock_rate * advertising.iloc[i-1])            

    return adstocked_advertising

   
adstock = pd.DataFrame()
adstock['TV_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['TV'])



adstock['Digital_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Digital'])



adstock['Sponsorship_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Sponsorship'])



adstock['Content_marketing_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Content_marketing'])



adstock['Online_marketing_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Online_marketing'])



adstock['Affiliates_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Affiliates'])



adstock['SEM_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['SEM'])



adstock['Radio_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Radio'])



adstock['Other_ads'] = adstocked_advertising(adstock_rate=0.5, advertising = marketing['Other'])
adstock.head()
marketing = pd.concat([marketing,adstock] ,axis=1)
marketing.head()
# The premium-ness of the product depends on the MRP. Higher the MRP, more premium is the product.

# Let's check the percentiles of MRP in the dataset.



consumer.product_mrp.describe(percentiles=[0.25,0.5,0.75,0.8,0.9,0.95,0.99])
# Let's assume that products with MRP greater than 90 percentile to be premium products.

# Create a dataframe with mrp, number of units sold and gmv against each product vertical to analyse better.

prod_cat = pd.DataFrame(pd.pivot_table(consumer, values = ['units','product_mrp', 'gmv'], index = ['product_analytic_vertical'], 

               aggfunc={'units':np.sum, 'product_mrp':np.mean, 'gmv':np.sum}).to_records())
# Marking products with MRP greater than 90th percentile with 1 and rest with 0

prod_cat['premium_product'] = np.where((prod_cat.product_mrp>consumer.product_mrp.quantile(0.9)),1,0)
prod_cat.loc[prod_cat.premium_product==1]
plt.figure(figsize=(15,5))

sns.barplot(x = prod_cat.product_analytic_vertical, y=prod_cat.gmv, hue=prod_cat.premium_product)

plt.xticks(rotation=90)

plt.show()
consumer = consumer.merge(prod_cat[['product_analytic_vertical', 'premium_product']] , left_on='product_analytic_vertical', 

            right_on='product_analytic_vertical',

                   how = 'inner')
sales = consumer.copy()
consumer.drop(['product_analytic_vertical'],1,inplace=True)
consumer.head()
camera_df = consumer[consumer['product_analytic_sub_category'] == 'CameraAccessory']
###  Removing outliers is important as

###  1. There may be some garbage value.

###  2. Bulk orders can skew the analysis
### Outlier Analysis

fig, axs = plt.subplots(1,3, figsize = (20,4))

plt1 = sns.boxplot(camera_df['gmv'], ax = axs[0])

plt2 = sns.boxplot(camera_df['units'], ax = axs[2])

plt4 = sns.boxplot(camera_df['product_mrp'], ax = axs[1])

plt.tight_layout()
### Treating outliers

### Outlier treatment for gmv & product_mrp

Q1 = camera_df.gmv.quantile(0.25)

Q3 = camera_df.gmv.quantile(0.75)

IQR = Q3 - Q1

camera_df = camera_df[(camera_df.gmv >= Q1 - 1.5*IQR) & (camera_df.gmv <= Q3 + 1.5*IQR)]

Q1 = camera_df.product_mrp.quantile(0.25)

Q3 = camera_df.product_mrp.quantile(0.75)

IQR = Q3 - Q1

camera_df = camera_df[(camera_df.product_mrp >= Q1 - 1.5*IQR) & (camera_df.product_mrp <= Q3 + 1.5*IQR)]
### Outlier Analysis

fig, axs = plt.subplots(1,3, figsize = (20,4))

plt1 = sns.boxplot(camera_df['gmv'], ax = axs[0])

plt2 = sns.boxplot(camera_df['units'], ax = axs[2])

plt4 = sns.boxplot(camera_df['product_mrp'], ax = axs[1])

plt.tight_layout()
camera_df.columns
camera_df.head()
### Aggregating dataset on weekly level



ca_week = pd.DataFrame(camera_df.groupby('week').agg({'gmv':'sum','listing_price':'mean',

                                                             'product_mrp':'mean','discount':'mean',

                                                             'sla':'mean','product_procurement_sla':'mean',

                                                             'fsn_id':pd.Series.nunique,'order_item_id':pd.Series.nunique,

                                                             'order_id': pd.Series.nunique,

                                                             'order_payment_type':'sum',

                                                            'premium_product':'sum'}))



ca_week.reset_index( inplace = True)
ca_week.head()
### Sum of GMV / No of unique Orders



ca_week['AOV'] = ca_week['gmv']/ca_week['order_id']
ca_week['online_order_perc'] = ca_week['order_payment_type']*100/ca_week['order_item_id']
ca_week.week.unique()
calendar.week.unique()
ca_week['week'] = ca_week['week'].astype(int)

calendar['week'] = calendar['week'].astype(int)
ca_week = ca_week.merge(marketing, how = 'left', on = 'week')
ca_week = ca_week.merge(calendar, how = 'left', on = 'week')
ca_week.head()
ca_week_viz = ca_week.round(2)
sns.distplot(ca_week_viz['gmv'],kde=True)
plt.figure(figsize=(15, 5))

sns.barplot(ca_week_viz['week'],ca_week_viz['gmv'])
ca_week_viz.columns
fig, axs = plt.subplots(2,4,figsize=(16,8))



plt1 = sns.scatterplot(x = 'Total_Investment', y = 'gmv', data = ca_week_viz, ax = axs[0,0])



plt2 = sns.scatterplot(x = 'TV', y = 'gmv', data = ca_week_viz, ax = axs[0,1])



plt3 = sns.scatterplot(x = 'Digital', y = 'gmv', data = ca_week_viz, ax = axs[0,2])



plt4 = sns.scatterplot(x = 'Sponsorship', y = 'gmv', data = ca_week_viz, ax = axs[0,3])



plt5 = sns.scatterplot(x = 'Content_marketing', y = 'gmv', data = ca_week_viz, ax = axs[1,0])



plt6 = sns.scatterplot(x = 'Online_marketing', y = 'gmv', data = ca_week_viz, ax = axs[1,1])



plt7 = sns.scatterplot(x = 'Affiliates', y = 'gmv', data = ca_week_viz, ax = axs[1,2])



plt8 = sns.scatterplot(x = 'SEM', y = 'gmv', data = ca_week_viz, ax = axs[1,3])



plt.tight_layout()
plt.figure(figsize=(20, 5))

sns.barplot(x= ca_week_viz['week'], y =ca_week_viz['gmv'], hue = ca_week_viz['Special_sales'], dodge = False)

plt.show()
plt.figure(figsize=(20, 5))

sns.barplot(x= ca_week_viz['week'], y =ca_week_viz['gmv'], hue = pd.cut(ca_week_viz['discount'],3), dodge = False)

plt.show()
### ca_week



### Moving Average for listing_price and discount



### ca_week = ca_week.sort_values('order_date')



ca_week[['MA2_LP','MA2_Discount']] = ca_week[['listing_price','discount']].rolling(window=2,min_periods=1).mean()

ca_week[['MA3_LP','MA3_Discount']] = ca_week[['listing_price','discount']].rolling(window=3,min_periods=1).mean()

ca_week[['MA4_LP','MA4_Discount']] = ca_week[['listing_price','discount']].rolling(window=4,min_periods=1).mean()



### Reference listed price Inflation 



ca_week['MA2_listed_price'] = (ca_week['listing_price']-ca_week['MA2_LP'])/ca_week['MA2_LP']

ca_week['MA3_listed_price'] = (ca_week['listing_price']-ca_week['MA3_LP'])/ca_week['MA3_LP']

ca_week['MA4_listed_price'] = (ca_week['listing_price']-ca_week['MA4_LP'])/ca_week['MA4_LP']



### Reference discount Inflation



ca_week['MA2_discount_offer'] = (ca_week['discount']-ca_week['MA2_Discount'])/ca_week['MA2_Discount']

ca_week['MA3_discount_offer'] = (ca_week['discount']-ca_week['MA3_Discount'])/ca_week['MA3_Discount']

ca_week['MA4_discount_offer'] = (ca_week['discount']-ca_week['MA4_Discount'])/ca_week['MA4_Discount']





ca_week.drop(['MA2_LP','MA3_LP','MA4_LP','MA2_Discount','MA3_Discount','MA4_Discount'], axis = 1, inplace = True)  

ca_week.head()



# ### To identify multicollinearity between variable

plt.figure(figsize=(20,20))

sns.heatmap(ca_week.corr(),annot = True, cmap="YlGnBu")

plt.show()
### Highly Correlated Columns should be dropped



ca_week.drop(['TV', 'Digital', 'Sponsorship', 'Content_marketing','Online_marketing', 'Affiliates', 'SEM','Radio',

              'Other'], axis = 1, inplace = True)
plt.figure(figsize=(25,20))

sns.heatmap(ca_week.corr(), cmap="coolwarm", annot=True)

plt.show()
ca_week.drop(['Affiliates_ads','SEM_ads','Digital_ads','Radio_ads','Other_ads','mean_temp_C','min_temp_C',

              'order_id','order_item_id','total_precip_mm','Total_Investment','MA3_discount_offer',

               'MA3_listed_price','AOV','max_temp_C','MA2_listed_price','MA4_discount_offer'],1,inplace=True)
#Successfully removed more than 90% correlation
### Lag of listed_price, discount_offer, NPS, Special_sales



ca_week['lag_1_listed_price'] = ca_week['listing_price'].shift(-1).fillna(0)

ca_week['lag_2_listed_price'] = ca_week['listing_price'].shift(-2).fillna(0)

ca_week['lag_3_listed_price'] = ca_week['listing_price'].shift(-3).fillna(0)



ca_week['lag_1_discount'] = ca_week['discount'].shift(-1).fillna(0)

ca_week['lag_2_discount'] = ca_week['discount'].shift(-2).fillna(0)

ca_week['lag_3_discount'] = ca_week['discount'].shift(-3).fillna(0)



ca_week['lag_1_Stock_Index'] = ca_week['Stock_Index'].shift(-1).fillna(0)

ca_week['lag_2_Stock_Index'] = ca_week['Stock_Index'].shift(-2).fillna(0)

ca_week['lag_3_Stock_Index'] = ca_week['Stock_Index'].shift(-3).fillna(0)



ca_week['lag_1_Special_sales'] = ca_week['Special_sales'].shift(-1).fillna(0)

ca_week['lag_2_Special_sales'] = ca_week['Special_sales'].shift(-2).fillna(0)

ca_week['lag_3_Special_sales'] = ca_week['Special_sales'].shift(-3).fillna(0)



ca_week['lag_1_Payday'] = ca_week['Payday'].shift(-1).fillna(0)

ca_week['lag_2_Payday'] = ca_week['Payday'].shift(-2).fillna(0)

ca_week['lag_3_Payday'] = ca_week['Payday'].shift(-3).fillna(0)



ca_week['lag_1_NPS'] = ca_week['NPS'].shift(-1).fillna(0)

ca_week['lag_2_NPS'] = ca_week['NPS'].shift(-2).fillna(0)

ca_week['lag_3_NPS'] = ca_week['NPS'].shift(-3).fillna(0)
ca_week.head()
gaming_accessory = consumer[consumer['product_analytic_sub_category'] == 'GamingAccessory']
###  Removing outliers is important as

###  1. There may be some garbage value.

###  2. Bulk orders can skew the analysis
### Outlier Analysis

fig, axs = plt.subplots(1,3, figsize = (20,4))

plt1 = sns.boxplot(gaming_accessory['gmv'], ax = axs[0])

plt2 = sns.boxplot(gaming_accessory['units'], ax = axs[2])

plt4 = sns.boxplot(gaming_accessory['product_mrp'], ax = axs[1])

plt.tight_layout()
### Treating outliers

### Outlier treatment for gmv & product_mrp

Q1 = gaming_accessory.gmv.quantile(0.25)

Q3 = gaming_accessory.gmv.quantile(0.75)

IQR = Q3 - Q1

gaming_accessory = gaming_accessory[(gaming_accessory.gmv >= Q1 - 1.5*IQR) & (gaming_accessory.gmv <= Q3 + 1.5*IQR)]

Q1 = gaming_accessory.product_mrp.quantile(0.25)

Q3 = gaming_accessory.product_mrp.quantile(0.75)

IQR = Q3 - Q1

gaming_accessory = gaming_accessory[(gaming_accessory.product_mrp >= Q1 - 1.5*IQR) & (gaming_accessory.product_mrp <= Q3 + 1.5*IQR)]
### Outlier Analysis

fig, axs = plt.subplots(1,3, figsize = (20,4))

plt1 = sns.boxplot(gaming_accessory['gmv'], ax = axs[0])

plt2 = sns.boxplot(gaming_accessory['units'], ax = axs[2])

plt4 = sns.boxplot(gaming_accessory['product_mrp'], ax = axs[1])

plt.tight_layout()
gaming_accessory.columns
### Aggregating dataset on weekly level



ga_week = pd.DataFrame(gaming_accessory.groupby('week').agg({'gmv':'sum','listing_price':'mean',

                                                             'product_mrp':'mean','discount':'mean',

                                                             'sla':'mean','product_procurement_sla':'mean',

                                                             'fsn_id':pd.Series.nunique,'order_item_id':pd.Series.nunique,

                                                             'order_id': pd.Series.nunique,

                                                             'order_payment_type':'sum'}))



ga_week.reset_index( inplace = True)
ga_week.head()
### Sum of GMV / No of unique Orders



ga_week['AOV'] = ga_week['gmv']/ga_week['order_id']
ga_week['online_order_perc'] = ga_week['order_payment_type']*100/ga_week['order_item_id']
ga_week.head()
ga_week = ga_week.merge(marketing, how = 'left', on = 'week')
ga_week = ga_week.merge(calendar, how = 'left', on = 'week')
ga_week.head()
ga_week_viz = ga_week.round(2)
sns.distplot(ga_week_viz['gmv'],kde=True)
plt.figure(figsize=(15, 5))

sns.barplot(ga_week_viz['week'],ga_week_viz['gmv'])
ga_week_viz.columns
fig, axs = plt.subplots(2,4,figsize=(16,8))



plt1 = sns.scatterplot(x = 'Total_Investment', y = 'gmv', data = ga_week_viz, ax = axs[0,0])



plt2 = sns.scatterplot(x = 'TV', y = 'gmv', data = ga_week_viz, ax = axs[0,1])



plt3 = sns.scatterplot(x = 'Digital', y = 'gmv', data = ga_week_viz, ax = axs[0,2])



plt4 = sns.scatterplot(x = 'Sponsorship', y = 'gmv', data = ga_week_viz, ax = axs[0,3])



plt5 = sns.scatterplot(x = 'Content_marketing', y = 'gmv', data = ga_week_viz, ax = axs[1,0])



plt6 = sns.scatterplot(x = 'Online_marketing', y = 'gmv', data = ga_week_viz, ax = axs[1,1])



plt7 = sns.scatterplot(x = 'Affiliates', y = 'gmv', data = ga_week_viz, ax = axs[1,2])



plt8 = sns.scatterplot(x = 'SEM', y = 'gmv', data = ga_week_viz, ax = axs[1,3])



plt.tight_layout()
plt.figure(figsize=(20, 5))

sns.barplot(x= ga_week_viz['week'], y =ga_week_viz['gmv'], hue = ga_week_viz['Special_sales'], dodge = False)

plt.show()
plt.figure(figsize=(20, 5))

sns.barplot(x= ga_week_viz['week'], y =ga_week_viz['gmv'], hue = pd.cut(ga_week_viz['discount'],3), dodge = False)

plt.show()
### ga_week



### Moving Average for listed_price and discount_offer



### ga_week = ga_week.sort_values('order_date')



ga_week[['MA2_LP','MA2_Discount']] = ga_week[['listing_price','discount']].rolling(window=2,min_periods=1).mean()

ga_week[['MA3_LP','MA3_Discount']] = ga_week[['listing_price','discount']].rolling(window=3,min_periods=1).mean()

ga_week[['MA4_LP','MA4_Discount']] = ga_week[['listing_price','discount']].rolling(window=4,min_periods=1).mean()



### Reference listed price Inflation 



ga_week['MA2_listed_price'] = (ga_week['listing_price']-ga_week['MA2_LP'])/ga_week['MA2_LP']

ga_week['MA3_listed_price'] = (ga_week['listing_price']-ga_week['MA3_LP'])/ga_week['MA3_LP']

ga_week['MA4_listed_price'] = (ga_week['listing_price']-ga_week['MA4_LP'])/ga_week['MA4_LP']



### Reference discount Inflation



ga_week['MA2_discount'] = (ga_week['discount']-ga_week['MA2_Discount'])/ga_week['MA2_Discount']

ga_week['MA3_discount'] = (ga_week['discount']-ga_week['MA3_Discount'])/ga_week['MA3_Discount']

ga_week['MA4_discount'] = (ga_week['discount']-ga_week['MA4_Discount'])/ga_week['MA4_Discount']





ga_week.drop(['MA2_LP','MA3_LP','MA4_LP','MA2_Discount','MA3_Discount','MA4_Discount'], axis = 1, inplace = True)  

ga_week



plt.figure(figsize=(25,20))



### Heatmap

sns.heatmap(ga_week.corr(), cmap="coolwarm", annot=True)

plt.show()


ga_week.drop(['TV', 'Digital', 'Sponsorship', 'Content_marketing','Online_marketing', 'Affiliates', 'SEM','Radio',

              'Other','Affiliates_ads','SEM_ads','Digital_ads','Radio_ads','Other_ads','mean_temp_C','min_temp_C',

              'order_id','order_item_id','total_precip_mm','Total_Investment','MA3_discount',

              'MA3_listed_price','AOV','MA4_listed_price'], axis = 1, inplace = True)
ga_week.drop(['max_temp_C'], axis = 1, inplace = True)
###  Successfully removed more than 90% highly correlated variables from dataset.

### Lag of listed_price, discount_offer, NPS, Special_sales



ga_week['lag_1_listed_price'] = ga_week['listing_price'].shift(-1).fillna(0)

ga_week['lag_2_listed_price'] = ga_week['listing_price'].shift(-2).fillna(0)

ga_week['lag_3_listed_price'] = ga_week['listing_price'].shift(-3).fillna(0)



ga_week['lag_1_discount_offer'] = ga_week['discount'].shift(-1).fillna(0)

ga_week['lag_2_discount_offer'] = ga_week['discount'].shift(-2).fillna(0)

ga_week['lag_3_discount_offer'] = ga_week['discount'].shift(-3).fillna(0)



ga_week['lag_1_NPS'] = ga_week['NPS'].shift(-1).fillna(0)

ga_week['lag_2_NPS'] = ga_week['NPS'].shift(-2).fillna(0)

ga_week['lag_3_NPS'] = ga_week['NPS'].shift(-3).fillna(0)



ga_week['lag_1_Stock_Index'] = ga_week['Stock_Index'].shift(-1).fillna(0)

ga_week['lag_2_Stock_Index'] = ga_week['Stock_Index'].shift(-2).fillna(0)

ga_week['lag_3_Stock_Index'] = ga_week['Stock_Index'].shift(-3).fillna(0)



ga_week['lag_1_Special_sales'] = ga_week['Special_sales'].shift(-1).fillna(0)

ga_week['lag_2_Special_sales'] = ga_week['Special_sales'].shift(-2).fillna(0)

ga_week['lag_3_Special_sales'] = ga_week['Special_sales'].shift(-3).fillna(0)



ga_week['lag_1_Payday'] = ga_week['Payday'].shift(-1).fillna(0)

ga_week['lag_2_Payday'] = ga_week['Payday'].shift(-2).fillna(0)

ga_week['lag_3_Payday'] = ga_week['Payday'].shift(-3).fillna(0)

ga_week.head()
home_audio = consumer[consumer['product_analytic_sub_category'] == 'HomeAudio']
###  Removing outliers is important as

###  1. There may be some garbage value.

###  2. Bulk orders can skew the analysis
### Outlier Analysis

fig, axs = plt.subplots(1,3, figsize = (20,4))

plt1 = sns.boxplot(home_audio['gmv'], ax = axs[0])

plt2 = sns.boxplot(home_audio['units'], ax = axs[2])

plt4 = sns.boxplot(home_audio['product_mrp'], ax = axs[1])

plt.tight_layout()
### Treating outliers

### Outlier treatment for gmv & product_mrp

Q1 = home_audio.gmv.quantile(0.25)

Q3 = home_audio.gmv.quantile(0.75)

IQR = Q3 - Q1

home_audio = home_audio[(home_audio.gmv >= Q1 - 1.5*IQR) & (home_audio.gmv <= Q3 + 1.5*IQR)]

Q1 = home_audio.product_mrp.quantile(0.25)

Q3 = home_audio.product_mrp.quantile(0.75)

IQR = Q3 - Q1

home_audio = home_audio[(home_audio.product_mrp >= Q1 - 1.5*IQR) & (home_audio.product_mrp <= Q3 + 1.5*IQR)]
### Outlier Analysis

fig, axs = plt.subplots(1,3, figsize = (20,4))

plt1 = sns.boxplot(home_audio['gmv'], ax = axs[0])

plt2 = sns.boxplot(home_audio['units'], ax = axs[2])

plt4 = sns.boxplot(home_audio['product_mrp'], ax = axs[1])

plt.tight_layout()
home_audio.columns
### Aggregating dataset on weekly level



ha_week = pd.DataFrame(home_audio.groupby('week').agg({'gmv':'sum','listing_price':'mean',

                                                             'product_mrp':'mean','discount':'mean',

                                                             'sla':'mean','product_procurement_sla':'mean',

                                                             'fsn_id':pd.Series.nunique,'order_item_id':pd.Series.nunique,

                                                             'order_id': pd.Series.nunique,

                                                             'order_payment_type':'sum'}))



ha_week.reset_index( inplace = True)
ha_week.head()
### Sum of GMV / No of unique Orders



ha_week['AOV'] = ha_week['gmv']/ha_week['order_id']
ha_week['online_order_perc'] = ha_week['order_payment_type']*100/ha_week['order_item_id']
ha_week.head()
ha_week = ha_week.merge(marketing, how = 'left', on = 'week')
ha_week = ha_week.merge(calendar, how = 'left', on = 'week')
ha_week.head()
ha_week_viz = ha_week.round(2)
sns.distplot(ha_week_viz['gmv'],kde=True)
plt.figure(figsize=(15, 5))

sns.barplot(ha_week_viz['week'],ha_week_viz['gmv'])
ha_week_viz.columns
fig, axs = plt.subplots(2,4,figsize=(16,8))



plt1 = sns.scatterplot(x = 'Total_Investment', y = 'gmv', data = ha_week_viz, ax = axs[0,0])



plt2 = sns.scatterplot(x = 'TV', y = 'gmv', data = ha_week_viz, ax = axs[0,1])



plt3 = sns.scatterplot(x = 'Digital', y = 'gmv', data = ha_week_viz, ax = axs[0,2])



plt4 = sns.scatterplot(x = 'Sponsorship', y = 'gmv', data = ha_week_viz, ax = axs[0,3])



plt5 = sns.scatterplot(x = 'Content_marketing', y = 'gmv', data = ha_week_viz, ax = axs[1,0])



plt6 = sns.scatterplot(x = 'Online_marketing', y = 'gmv', data = ha_week_viz, ax = axs[1,1])



plt7 = sns.scatterplot(x = 'Affiliates', y = 'gmv', data = ha_week_viz, ax = axs[1,2])



plt8 = sns.scatterplot(x = 'SEM', y = 'gmv', data = ha_week_viz, ax = axs[1,3])



plt.tight_layout()
plt.figure(figsize=(20, 5))

sns.barplot(x= ha_week_viz['week'], y =ha_week_viz['gmv'], hue = ha_week_viz['Special_sales'], dodge = False)

plt.show()
plt.figure(figsize=(20, 5))

sns.barplot(x= ha_week_viz['week'], y =ha_week_viz['gmv'], hue = pd.cut(ha_week_viz['discount'],3), dodge = False)

plt.show()
### ha_week



### Moving Average for listed_price and discount_offer



### ha_week = ha_week.sort_values('order_date')



ha_week[['MA2_LP','MA2_Discount']] = ha_week[['listing_price','discount']].rolling(window=2,min_periods=1).mean()

ha_week[['MA3_LP','MA3_Discount']] = ha_week[['listing_price','discount']].rolling(window=3,min_periods=1).mean()

ha_week[['MA4_LP','MA4_Discount']] = ha_week[['listing_price','discount']].rolling(window=4,min_periods=1).mean()



### Reference listed price Inflation 



ha_week['MA2_listed_price'] = (ha_week['listing_price']-ha_week['MA2_LP'])/ha_week['MA2_LP']

ha_week['MA3_listed_price'] = (ha_week['listing_price']-ha_week['MA3_LP'])/ha_week['MA3_LP']

ha_week['MA4_listed_price'] = (ha_week['listing_price']-ha_week['MA4_LP'])/ha_week['MA4_LP']



### Reference discount Inflation



ha_week['MA2_discount'] = (ha_week['discount']-ha_week['MA2_Discount'])/ha_week['MA2_Discount']

ha_week['MA3_discount'] = (ha_week['discount']-ha_week['MA3_Discount'])/ha_week['MA3_Discount']

ha_week['MA4_discount'] = (ha_week['discount']-ha_week['MA4_Discount'])/ha_week['MA4_Discount']





ha_week.drop(['MA2_LP','MA3_LP','MA4_LP','MA2_Discount','MA3_Discount','MA4_Discount'], axis = 1, inplace = True)  

ha_week



plt.figure(figsize=(25,20))



### Heatmap

sns.heatmap(ha_week.corr(), cmap="coolwarm", annot=True)

plt.show()
ha_week.drop(['TV', 'Digital', 'Sponsorship', 'Content_marketing','Online_marketing', 'Affiliates', 'SEM','Radio',

              'Other','Affiliates_ads','SEM_ads','Digital_ads','Radio_ads','Other_ads','mean_temp_C','min_temp_C',

              'order_id','order_item_id','total_precip_mm','Total_Investment','MA3_discount',

              'MA3_listed_price','AOV'], axis = 1, inplace = True)
ha_week.drop(['max_temp_C'], axis = 1, inplace = True)
###  Successfully removed more than 90% highly correlated variables from dataset.
### Lag of listed_price, discount_offer, NPS, Special_sales



ha_week['lag_1_listed_price'] = ha_week['listing_price'].shift(-1).fillna(0)

ha_week['lag_2_listed_price'] = ha_week['listing_price'].shift(-2).fillna(0)

ha_week['lag_3_listed_price'] = ha_week['listing_price'].shift(-3).fillna(0)



ha_week['lag_1_discount_offer'] = ha_week['discount'].shift(-1).fillna(0)

ha_week['lag_2_discount_offer'] = ha_week['discount'].shift(-2).fillna(0)

ha_week['lag_3_discount_offer'] = ha_week['discount'].shift(-3).fillna(0)



ha_week['lag_1_NPS'] = ha_week['NPS'].shift(-1).fillna(0)

ha_week['lag_2_NPS'] = ha_week['NPS'].shift(-2).fillna(0)

ha_week['lag_3_NPS'] = ha_week['NPS'].shift(-3).fillna(0)



ha_week['lag_1_Stock_Index'] = ha_week['Stock_Index'].shift(-1).fillna(0)

ha_week['lag_2_Stock_Index'] = ha_week['Stock_Index'].shift(-2).fillna(0)

ha_week['lag_3_Stock_Index'] = ha_week['Stock_Index'].shift(-3).fillna(0)



ha_week['lag_1_Special_sales'] = ha_week['Special_sales'].shift(-1).fillna(0)

ha_week['lag_2_Special_sales'] = ha_week['Special_sales'].shift(-2).fillna(0)

ha_week['lag_3_Special_sales'] = ha_week['Special_sales'].shift(-3).fillna(0)



ha_week['lag_1_Payday'] = ha_week['Payday'].shift(-1).fillna(0)

ha_week['lag_2_Payday'] = ha_week['Payday'].shift(-2).fillna(0)

ha_week['lag_3_Payday'] = ha_week['Payday'].shift(-3).fillna(0)

ha_week.head(10)
###  Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
ca_week.columns
camera_lm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer','premium_product']]

                            

    

camera_lm.head()
### Checking NaN



camera_lm.isnull().sum()
camera_lm.fillna(0, inplace = True)
from sklearn.model_selection import train_test_split





np.random.seed(0)

df_train, df_test = train_test_split(camera_lm, train_size = 0.7, test_size = 0.3, random_state = 100)
### Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



### Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





### Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer','premium_product']

                                      



### Scale these variables using 'fit_transform'

df_train[varlist] = scaler.fit_transform(df_train[varlist])
df_train.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split

X_train = df_train.drop('gmv',axis=1)

y_train = df_train['gmv']

#RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

    

def checkVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
X_train_new = build_model(X_train_rfe,y_train)
checkVIF(X_train_new)
X_train_new = X_train_rfe.drop(["discount"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["heat_deg_days"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
X_train_new = X_train_new.drop(["snow_on_grnd_cm"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["sla"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["MA2_discount_offer"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["product_procurement_sla"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["MA4_listed_price"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)   
#Scaling the test set

num_vars = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer','premium_product']

df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

#Dividing into X and y

y_test = df_test.pop('gmv')

X_test = df_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
#EVALUATION OF THE MODEL

# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)   
print(lm.summary())
###  Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
ca_week.columns
camera_lm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer']]

                            

    

camera_lm.head()
### Checking NaN

camera_lm.isnull().sum()
camera_lm.fillna(0, inplace = True)
### Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



### Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





### Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer']

                                      



### Scale these variables using 'fit_transform'

camera_lm[varlist] = scaler.fit_transform(camera_lm[varlist])
camera_lm.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = camera_lm.drop('gmv',axis=1)

y = camera_lm['gmv']



camera_train_lm = camera_lm
print("x dataset: ",x.shape)

print("y dataset: ",y.shape)
###  Instantiate

lm = LinearRegression()



###  Fit a line

lm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(lm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=[ 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

lm1 = sm.OLS(y, x_rfe1).fit() 



print(lm1.params)
print(lm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]





###  Predicition with selected features on the test data

y_pred = lm1.predict(sm.add_constant(x_2))

###  Mean square error (MSE)



mse = np.mean((y_pred - y)**2)

mse
###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(lm.coef_)

coef

### Mean Square Error 

###  Using K-Fold Cross validation evaluating on selected dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(lm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#    features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(lm1,camera_train_lm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(camera_train_lm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
ca_week.columns
camera_mm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer']]         



camera_mm.head()

### Applying Log 

camera_mm=np.log(camera_mm)



camera_mm = camera_mm.fillna(0)

camera_mm = camera_mm.replace([np.inf, -np.inf], 0)
camera_mm.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()



###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer']      







### Scale these variables using 'fit_transform'

camera_mm[varlist] = scaler.fit_transform(camera_mm[varlist])
camera_mm.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split



x = camera_mm.drop('gmv',axis=1)

y = camera_mm['gmv']



camera_train_mm = camera_mm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)

### Instantiate

mm = LinearRegression()



### Fit a line

mm.fit(x,y)

### Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(mm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
### Import statsmodels

import statsmodels.api as sm  



### Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)





### Fitting the model with selected variables

mm1 = sm.OLS(y, x_rfe1).fit() 



print(mm1.params)
print(mm1.summary())
x_rfe1.drop('TV_ads',1,inplace=True)



x_rfe1 = sm.add_constant(x_rfe1)





### Fitting the model with selected variables

mm1 = sm.OLS(y, x_rfe1).fit() 



print(mm1.params)
print(mm1.summary())
### Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
### Model Evaluation on testing data

x_2 = x[features]





### Predicition with selected features on the test data

y_pred = mm1.predict(sm.add_constant(x_2))

### Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse
### Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(mm.coef_)

coef

### Mean Square Error 

###  Using K-Fold Cross validation evaluating on selected dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(mm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#     features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(mm1,camera_train_mm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
ca_week.columns
camera_km = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer']]           





camera_km.head()

camera_km['lag_1_gmv'] = camera_km['gmv'].shift(-1)
### Checking NaN



camera_km.isnull().sum()
camera_km = camera_km.fillna(0)
camera_km.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



### Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()



### Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer','lag_1_gmv']



### Scale these variables using 'fit_transform'

camera_km[varlist] = scaler.fit_transform(camera_km[varlist])
camera_km.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = camera_km.drop('gmv',axis=1)

y = camera_km['gmv']



camera_train_km = camera_km
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

km = LinearRegression()



###  Fit a line

km.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(km.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA4_listed_price',

       'MA2_discount_offer', 'lag_1_gmv'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ### forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
### Import statsmodels

import statsmodels.api as sm  



### Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



### Fitting the model with selected variables

km1 = sm.OLS(y, x_rfe1).fit() 



print(km1.params)
print(km1.summary())
### Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
### Model Evaluation on testing data

x_2 = x[features]





### Predicition with selected features on the test data

y_pred = km1.predict(sm.add_constant(x_2))

### Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse
### Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(km.coef_)

coef

### Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(km,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(km1,camera_train_km)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(camera_train_km[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
ca_week.columns
camera_dlm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',

       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',

       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           





camera_dlm.head()

camera_dlm['lag_1_gmv'] = camera_dlm['gmv'].shift(-1)

camera_dlm['lag_2_gmv'] = camera_dlm['gmv'].shift(-2)

camera_dlm['lag_3_gmv'] = camera_dlm['gmv'].shift(-3)

### Checking NaN



camera_dlm.isnull().sum()
camera_dlm = camera_dlm.fillna(0)
camera_dlm.head()
camera_dlm.columns
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',

       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',

       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']





###  Scale these variables using 'fit_transform'

camera_dlm[varlist] = scaler.fit_transform(camera_dlm[varlist])
camera_dlm.head()
###  Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = camera_dlm.drop('gmv',axis=1)

y = camera_dlm['gmv']



camera_train_dlm = camera_dlm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

dlm = LinearRegression()



###  Fit a line

dlm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',

       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',

       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],

                     threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###  forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###  backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###  use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###  null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

dlm1 = sm.OLS(y, x_rfe1).fit() 



print(dlm1.params)
print(dlm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('discount', axis = 1, inplace = True)
### 2
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('product_procurement_sla', axis = 1, inplace = True)

### 3
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_listed_price', axis = 1, inplace = True)

### 4
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]



###  Predicition with selected features on the test data

y_pred = dlm1.predict(sm.add_constant(x_2))
###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse

###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

###  Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(dlm1,camera_train_dlm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(camera_train_dlm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
ca_week.columns
camera_dlmm = ca_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',

       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',

       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           





camera_dlmm.head()

camera_dlmm['lag_1_gmv'] = camera_dlmm['gmv'].shift(-1)

camera_dlmm['lag_2_gmv'] = camera_dlmm['gmv'].shift(-2)

camera_dlmm['lag_3_gmv'] = camera_dlmm['gmv'].shift(-3)

### Checking NaN



camera_dlmm.isnull().sum()
camera_dlmm = camera_dlmm.fillna(0)
### Applying Log 

camera_dlmm=np.log(camera_dlmm)



camera_dlmm = camera_dlmm.fillna(0)

camera_dlmm = camera_dlmm.replace([np.inf, -np.inf], 0)
camera_dlmm.head()
camera_dlmm.columns
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',

       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',

       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']





###  Scale these variables using 'fit_transform'

camera_dlmm[varlist] = scaler.fit_transform(camera_dlmm[varlist])

camera_dlmm.head()
###  Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = camera_dlmm.drop('gmv',axis=1)

y = camera_dlmm['gmv']



camera_train_dlmm = camera_dlmm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

dlm = LinearRegression()



###  Fit a line

dlm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday','heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm',  'MA4_listed_price',

       'MA2_discount_offer', 'lag_1_listed_price','lag_1_discount',

       'lag_2_discount','lag_3_discount','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],

                     threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###  forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###  backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###  use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###  null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

dlm1 = sm.OLS(y, x_rfe1).fit() 



print(dlm1.params)
print(dlm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_gmv', axis = 1, inplace = True)
### 2
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_2_discount', axis = 1, inplace = True)

### 3
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('Special_sales', axis = 1, inplace = True)

### 4
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_listed_price', axis = 1, inplace = True)

### 5
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_discount', axis = 1, inplace = True)

### 6
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_2_gmv', axis = 1, inplace = True)

### 7
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_2_NPS', axis = 1, inplace = True)

### 8
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('sla', axis = 1, inplace = True)

### 9
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]



###  Predicition with selected features on the test data

y_pred = dlm1.predict(sm.add_constant(x_2))
###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse

###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

###  Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(dlm1,camera_train_dlm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(camera_train_dlm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
###  Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
ga_week.columns
gaming_lm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']]

                            

    

gaming_lm.head()
### Checking NaN

gaming_lm.isnull().sum()
gaming_lm.fillna(0, inplace = True)
### Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



### Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





### Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']

                                      



### Scale these variables using 'fit_transform'

gaming_lm[varlist] = scaler.fit_transform(gaming_lm[varlist])
gaming_lm.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = gaming_lm.drop('gmv',axis=1)

y = gaming_lm['gmv']



gaming_train_lm = gaming_lm
print("x dataset: ",x.shape)

print("y dataset: ",y.shape)
###  Instantiate

lm = LinearRegression()



###  Fit a line

lm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(lm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

lm1 = sm.OLS(y, x_rfe1).fit() 



print(lm1.params)
print(lm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]





###  Predicition with selected features on the test data

y_pred = lm1.predict(sm.add_constant(x_2))

###  Mean square error (MSE)



mse = np.mean((y_pred - y)**2)

mse
###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(lm.coef_)

coef

### Mean Square Error 

###  Using K-Fold Cross validation evaluating on selected dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(lm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#    features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(lm1,gaming_train_lm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(gaming_train_lm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
ga_week.columns
gaming_mm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']]         



gaming_mm.head()

### Applying Log 

gaming_mm=np.log(gaming_mm)



gaming_mm = gaming_mm.fillna(0)

gaming_mm = gaming_mm.replace([np.inf, -np.inf], 0)
gaming_mm.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()



###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']      







### Scale these variables using 'fit_transform'

gaming_mm[varlist] = scaler.fit_transform(gaming_mm[varlist])
gaming_mm.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split



x = gaming_mm.drop('gmv',axis=1)

y = gaming_mm['gmv']



gaming_train_mm = gaming_mm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)

### Instantiate

mm = LinearRegression()



### Fit a line

mm.fit(x,y)

### Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(mm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
### Import statsmodels

import statsmodels.api as sm  



### Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)





### Fitting the model with selected variables

mm1 = sm.OLS(y, x_rfe1).fit() 



print(mm1.params)
print(mm1.summary())
### Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('order_payment_type', axis = 1, inplace = True)
### 2
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

mm1 = sm.OLS(y, x_rfe1).fit()   

print(mm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('snow_on_grnd_cm', axis = 1, inplace = True)

### 3
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

mm1 = sm.OLS(y, x_rfe1).fit()   

print(mm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
### Model Evaluation on testing data

x_2 = x[features]





### Predicition with selected features on the test data

y_pred = mm1.predict(sm.add_constant(x_2))

### Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse
### Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(mm.coef_)

coef

### Mean Square Error 

###  Using K-Fold Cross validation evaluating on selected dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(mm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#     features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(mm1,gaming_train_mm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(gaming_train_mm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
ga_week.columns
gaming_km = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price']]           





gaming_km.head()

gaming_km['lag_1_gmv'] = gaming_km['gmv'].shift(-1)
### Checking NaN



gaming_km.isnull().sum()
gaming_km = gaming_km.fillna(0)
gaming_km.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



### Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()



### Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_gmv']



### Scale these variables using 'fit_transform'

gaming_km[varlist] = scaler.fit_transform(gaming_km[varlist])
gaming_km.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = gaming_km.drop('gmv',axis=1)

y = gaming_km['gmv']



gaming_train_km = gaming_km
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

km = LinearRegression()



###  Fit a line

km.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(km.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'lag_1_gmv'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ### forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
### Import statsmodels

import statsmodels.api as sm  



### Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



### Fitting the model with selected variables

km1 = sm.OLS(y, x_rfe1).fit() 



print(km1.params)
print(km1.summary())
### Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
### Model Evaluation on testing data

x_2 = x[features]





### Predicition with selected features on the test data

y_pred = km1.predict(sm.add_constant(x_2))

### Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse
### Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(km.coef_)

coef

### Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(km,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(km1,gaming_train_km)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(gaming_train_km[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
gaming_dlm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           





gaming_dlm.head()

gaming_dlm['lag_1_gmv'] = gaming_dlm['gmv'].shift(-1)

gaming_dlm['lag_2_gmv'] = gaming_dlm['gmv'].shift(-2)

gaming_dlm['lag_3_gmv'] = gaming_dlm['gmv'].shift(-3)

### Checking NaN



# gaming_dlm.isnull().sum()
gaming_dlm = gaming_dlm.fillna(0)
gaming_dlm.head()
gaming_dlm.columns
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']





###  Scale these variables using 'fit_transform'

gaming_dlm[varlist] = scaler.fit_transform(gaming_dlm[varlist])
gaming_dlm.head()
###  Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = gaming_dlm.drop('gmv',axis=1)

y = gaming_dlm['gmv']



gaming_train_dlm = gaming_dlm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

dlm = LinearRegression()



###  Fit a line

dlm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],

                     threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###  forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###  backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###  use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###  null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

dlm1 = sm.OLS(y, x_rfe1).fit() 



print(dlm1.params)
print(dlm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_3_Stock_Index', axis = 1, inplace = True)
### 2
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_3_NPS', axis = 1, inplace = True)

### 3
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('sla', axis = 1, inplace = True)

### 4
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]



###  Predicition with selected features on the test data

y_pred = dlm1.predict(sm.add_constant(x_2))
###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse

###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

###  Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(dlm1,gaming_train_dlm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(gaming_train_dlm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
gaming_dlmm = ga_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           





gaming_dlmm.head()

gaming_dlmm['lag_1_gmv'] = gaming_dlmm['gmv'].shift(-1)

gaming_dlmm['lag_2_gmv'] = gaming_dlmm['gmv'].shift(-2)

gaming_dlmm['lag_3_gmv'] = gaming_dlmm['gmv'].shift(-3)

### Checking NaN



gaming_dlmm.isnull().sum()
gaming_dlmm = gaming_dlmm.fillna(0)
### Applying Log 

gaming_dlmm=np.log(gaming_dlmm)



gaming_dlmm = gaming_dlmm.fillna(0)

gaming_dlmm = gaming_dlmm.replace([np.inf, -np.inf], 0)
gaming_dlmm.head()
gaming_dlmm.columns
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']





###  Scale these variables using 'fit_transform'

gaming_dlmm[varlist] = scaler.fit_transform(gaming_dlmm[varlist])

gaming_dlmm.head()
###  Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = gaming_dlmm.drop('gmv',axis=1)

y = gaming_dlmm['gmv']



gaming_train_dlmm = gaming_dlmm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

dlm = LinearRegression()



###  Fit a line

dlm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],

                     threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###  forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###  backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###  use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###  null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

dlm1 = sm.OLS(y, x_rfe1).fit() 



print(dlm1.params)
print(dlm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_3_Stock_Index', axis = 1, inplace = True)
### 2
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_3_discount_offer', axis = 1, inplace = True)

### 3
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_2_NPS', axis = 1, inplace = True)

### 4
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_2_discount_offer', axis = 1, inplace = True)

### 5
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('Online_marketing_ads', axis = 1, inplace = True)

### 6
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_listed_price', axis = 1, inplace = True)

### 7
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_discount_offer', axis = 1, inplace = True)

### 8
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_3_NPS', axis = 1, inplace = True)

### 9
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('Stock_Index', axis = 1, inplace = True)

### 10
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('order_payment_type', axis = 1, inplace = True)

### 11
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_gmv', axis = 1, inplace = True)

### 12
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_1_Stock_Index', axis = 1, inplace = True)

### 13
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('TV_ads', axis = 1, inplace = True)

### 14
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]



###  Predicition with selected features on the test data

y_pred = dlm1.predict(sm.add_constant(x_2))
###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse

###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

###  Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(dlm1,gaming_train_dlm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(gaming_train_dlm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
###  Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
ha_week.columns
home_lm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']]

                            

    

home_lm.head()
### Checking NaN



home_lm.isnull().sum()
home_lm.fillna(0, inplace = True)
### Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



### Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





### Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']

                                      



### Scale these variables using 'fit_transform'

home_lm[varlist] = scaler.fit_transform(home_lm[varlist])
home_lm.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = home_lm.drop('gmv',axis=1)

y = home_lm['gmv']



home_train_lm = home_lm
print("x dataset: ",x.shape)

print("y dataset: ",y.shape)
###  Instantiate

lm = LinearRegression()



###  Fit a line

lm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(lm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
### Import statsmodels

import statsmodels.api as sm  



### Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



### Fitting the model with selected variables

lm1 = sm.OLS(y, x_rfe1).fit() 



print(lm1.params)
print(lm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]



###  Predicition with selected features on the test data

y_pred = lm1.predict(sm.add_constant(x_2))

###  Mean square error (MSE)



mse = np.mean((y_pred - y)**2)

mse
###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(lm.coef_)

coef
### Mean Square Error 

###  Using K-Fold Cross validation evaluating on selected dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(lm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#    features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(lm1,home_train_lm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(home_train_lm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
ha_week.columns
home_mm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']]         



home_mm.head()

### Applying Log 

home_mm=np.log(home_mm)



home_mm = home_mm.fillna(0)

home_mm = home_mm.replace([np.inf, -np.inf], 0)
home_mm.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()



###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']      







### Scale these variables using 'fit_transform'

home_mm[varlist] = scaler.fit_transform(home_mm[varlist])
home_mm.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split



x = home_mm.drop('gmv',axis=1)

y = home_mm['gmv']



home_train_mm = home_mm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)

### Instantiate

mm = LinearRegression()



### Fit a line

mm.fit(x,y)

### Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(mm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
### Import statsmodels

import statsmodels.api as sm  



### Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)





### Fitting the model with selected variables

mm1 = sm.OLS(y, x_rfe1).fit() 



print(mm1.params)
print(mm1.summary())
### Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
### Model Evaluation on testing data

x_2 = x[features]





### Predicition with selected features on the test data

y_pred = mm1.predict(sm.add_constant(x_2))

### Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse
### Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(mm.coef_)

coef

### Mean Square Error 

###  Using K-Fold Cross validation evaluating on selected dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(mm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#     features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(mm1,home_train_mm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(home_train_mm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
ha_week.columns
home_km = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price']]           





home_km.head()

home_km['lag_1_gmv'] = home_km['gmv'].shift(-1)
### Checking NaN



home_km.isnull().sum()
home_km = home_km.fillna(0)
home_km.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



### Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()



### Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_gmv']



### Scale these variables using 'fit_transform'

home_km[varlist] = scaler.fit_transform(home_km[varlist])
home_km.head()
### Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = home_km.drop('gmv',axis=1)

y = home_km['gmv']



home_train_km = home_km
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

km = LinearRegression()



###  Fit a line

km.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(km.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price', 'lag_1_gmv'],

                       threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ### forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
### Import statsmodels

import statsmodels.api as sm  



### Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



### Fitting the model with selected variables

km1 = sm.OLS(y, x_rfe1).fit() 



print(km1.params)
print(km1.summary())
### Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
### Model Evaluation on testing data

x_2 = x[features]





### Predicition with selected features on the test data

y_pred = km1.predict(sm.add_constant(x_2))

### Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse
### Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(km.coef_)

coef

### Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(km,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(km1,home_train_km)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(home_train_km[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
home_dlm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           





home_dlm.head()

home_dlm['lag_1_gmv'] = home_dlm['gmv'].shift(-1)

home_dlm['lag_2_gmv'] = home_dlm['gmv'].shift(-2)

home_dlm['lag_3_gmv'] = home_dlm['gmv'].shift(-3)

### Checking NaN



home_dlm.isnull().sum()
home_dlm = home_dlm.fillna(0)
home_dlm.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']





###  Scale these variables using 'fit_transform'

home_dlm[varlist] = scaler.fit_transform(home_dlm[varlist])
home_dlm.head()
###  Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = home_dlm.drop('gmv',axis=1)

y = home_dlm['gmv']



home_train_dlm = home_dlm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

dlm = LinearRegression()



###  Fit a line

dlm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],

                     threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###  forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###  backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###  use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###  null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

dlm1 = sm.OLS(y, x_rfe1).fit() 



print(dlm1.params)
print(dlm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('sla', axis = 1, inplace = True)

# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
x_rfe1.drop('lag_1_discount_offer', axis = 1, inplace = True)
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
x_rfe1.drop('product_procurement_sla', axis = 1, inplace = True)
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlm1 = sm.OLS(y, x_rfe1).fit()   

print(dlm1.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]



###  Predicition with selected features on the test data

y_pred = dlm1.predict(sm.add_constant(x_2))
###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse

###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(dlm.coef_)

coef

###  Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(dlm1,home_train_dlm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(home_train_dlm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()
home_dlmm = ha_week[['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']]           





home_dlmm.head()

home_dlmm['lag_1_gmv'] = home_dlmm['gmv'].shift(-1)

home_dlmm['lag_2_gmv'] = home_dlmm['gmv'].shift(-2)

home_dlmm['lag_3_gmv'] = home_dlmm['gmv'].shift(-3)

### Checking NaN



home_dlmm.isnull().sum()
home_dlmm = home_dlmm.fillna(0)
### Applying Log 

home_dlmm=np.log(home_dlmm)



home_dlmm = home_dlmm.fillna(0)

home_dlmm = home_dlmm.replace([np.inf, -np.inf], 0)
home_dlmm.head()
###  Import the StandardScaler()

# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



###  Create a scaling object

# scaler = StandardScaler()

scaler = MinMaxScaler()





###  Create a list of the variables that you need to scale

varlist = ['gmv', 'discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday']





###  Scale these variables using 'fit_transform'

home_dlmm[varlist] = scaler.fit_transform(home_dlmm[varlist])
home_dlmm.head()
###  Split the train dataset into X and y

from sklearn.model_selection import train_test_split

x = home_dlmm.drop('gmv',axis=1)

y = home_dlmm['gmv']



home_train_dlmm = home_dlmm
print("X = Independent variable & Y = Target variable")

print(x.shape,y.shape)
###  Instantiate

dlmm = LinearRegression()



###  Fit a line

dlmm.fit(x,y)

###  Coefficient values



coef = pd.DataFrame(x.columns)

coef['Coefficient'] = pd.Series(dlmm.coef_)

coef

col = x.columns

col
import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
def stepwise_selection(x, y,

                       initial_list=['discount', 'sla','product_procurement_sla', 'order_payment_type',

       'online_order_perc', 'TV_ads','Sponsorship_ads', 'Content_marketing_ads', 'Online_marketing_ads',

       'NPS', 'Stock_Index', 'Special_sales', 'Payday', 'heat_deg_days', 'cool_deg_days', 

       'total_rain_mm', 'total_snow_cm','snow_on_grnd_cm', 'MA2_listed_price', 'MA4_listed_price','lag_1_listed_price','lag_1_discount_offer',

       'lag_2_discount_offer','lag_3_discount_offer','lag_2_NPS','lag_3_NPS','lag_1_Stock_Index',

       'lag_2_Stock_Index','lag_3_Stock_Index','lag_1_Special_sales','lag_2_Special_sales','lag_3_Special_sales',

       'lag_1_Payday','lag_2_Payday','lag_3_Payday'],

                     threshold_in=0.01,threshold_out = 0.05, verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        ###  forward step

        excluded = list(set(x.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                

                

        ###  backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()

        ###  use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() ###  null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included

import statsmodels.api as sm  



final_features = stepwise_selection(x, y)



print("\n","final_selected_features:",final_features)
###  Import statsmodels

import statsmodels.api as sm  



###  Subsetting training data for 15 selected columns

x_rfe1 = x[final_features]



x_rfe1 = sm.add_constant(x_rfe1)



###  Fitting the model with selected variables

dlmm1 = sm.OLS(y, x_rfe1).fit() 



print(dlmm1.params)
print(dlmm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_rfe1.drop('lag_2_Stock_Index', axis = 1, inplace = True)

# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlmm1 = sm.OLS(y, x_rfe1).fit()   

print(dlmm1.summary())
x_rfe1.drop('lag_2_NPS', axis = 1, inplace = True)
# Refitting with final selected variables

x_rfe1 = sm.add_constant(x_rfe1)



# Fitting the model with final selected variables

dlmm1 = sm.OLS(y, x_rfe1).fit()   

print(dlmm1.summary())
###  Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()



vif['Features'] = x_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_rfe1.values, i) for i in range(x_rfe1.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
features = list(x_rfe1.columns)

features.remove('const')

features
###  Model Evaluation on testing data

x_2 = x[features]



###  Predicition with selected features on the test data

y_pred = dlmm1.predict(sm.add_constant(x_2))
###  Mean square error (MSE)

mse = np.mean((y_pred - y)**2)

mse

###  Coefficient values



coef = pd.DataFrame(x_rfe1.columns)

coef['Coefficient'] = pd.Series(dlmm.coef_)

coef

###  Using K-Fold Cross validation evaluating on whole dataset



# lm = LinearRegression()

fold = KFold(10,shuffle = True, random_state = 100)



cv_scores = -cross_val_score(dlm,x[features],y,cv=fold,scoring='neg_mean_squared_error')



print("Neg. of MSE:",cv_scores,"\n")

print("Mean of 5 KFold CV - MSE:",cv_scores.mean())
def elasticity(model,x):

    

    features_df = pd.DataFrame(model.params)

    features_df = features_df.rename(columns={0:'coef'})

    

    features_df['imp_feature'] = model.params.index

    features_df = features_df[features_df.imp_feature != 'const'][['imp_feature','coef']]

    features_df.index = range(len(features_df))

#      features



    elasticity_list = list()

    

    for i in range(len(features_df)):

        elasticity_list.append(((features_df.coef[i] * np.mean(x[features_df.imp_feature[i]])) / np.mean(x['gmv'])))



    features_df['elasticity'] = np.round(elasticity_list,3)

    

    sns.barplot(x='elasticity',y='imp_feature',data=features_df)

    plt.show()

    

    return features_df

    
elasticity(dlmm1,home_train_dlm)
# Plotting y and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y, y_pred)

fig.suptitle('y vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)  
# Figure size

plt.figure(figsize=(8,5))



# Heatmap

sns.heatmap(home_train_dlm[features].corr(), cmap="YlGnBu", annot=True)

plt.show()