import pandas as pd



#load file

nyc = pd.read_csv('../input/nyc-rolling-sales.csv', sep =',')



#show dataset details

print('number of entries:',nyc.shape)

nyc.dtypes
nyc = nyc.rename(columns={'SALE PRICE': 'SALEPRICE'})
# Look for irrelevant data to drop

nyc['SALEPRICE'].value_counts().head(10)
# Drop irrelevant transactions (0$, 1$, 10$)



nyc = nyc[nyc.SALEPRICE != '0']

nyc = nyc[nyc.SALEPRICE != '10']

nyc = nyc[nyc.SALEPRICE != '1']

nyc = nyc[nyc.SALEPRICE != ' -  ']

print('number of entries:',nyc.shape)
nyc['SALEPRICE'] = pd.to_numeric(nyc['SALEPRICE'])

nyc.dtypes
# Dicard rows with empty (NaN) fields



nyc.isnull().any()

nyc = nyc.dropna()

nyc.shape
#Set SALE DATE as index ID for sorting purposes



nyc['SALE DATE'].dtype

pd.to_datetime(nyc['SALE DATE'])
# Check how the DF looks like

nyc.set_index('SALE DATE', inplace=True)
# Change borough index to borough real name



nyc['BOROUGH'][nyc['BOROUGH'] == 1] = 'Manhattan'

nyc['BOROUGH'][nyc['BOROUGH'] == 2] = 'Bronx'

nyc['BOROUGH'][nyc['BOROUGH'] == 3] = 'Brooklyn'

nyc['BOROUGH'][nyc['BOROUGH'] == 4] = 'Queens'

nyc['BOROUGH'][nyc['BOROUGH'] == 5] = 'Staten Island'

#Date time index applied



nyc.index
#Cleaned data preview



nyc.head(25)
#Set index data dype to date



nyc.index = pd.to_datetime(nyc.index) 

nyc.index
#Divide cleaned data into 2017 monthly data sets



jan = nyc.loc['2017-1-1':'2017-1-31']

feb = nyc.loc['2017-2-1':'2017-2-28']

mar = nyc.loc['2017-3-1':'2017-3-31']

apr = nyc.loc['2017-4-1':'2017-4-30']

may = nyc.loc['2017-5-1':'2017-5-31']

jun = nyc.loc['2017-6-1':'2017-6-30']

jul = nyc.loc['2017-7-1':'2017-7-31']

aug = nyc.loc['2017-8-1':'2017-8-31']

print(jan.shape)

print(feb.shape)

print(mar.shape)

print(apr.shape)

print(may.shape)

print(jun.shape)

print(jul.shape)

print(aug.shape)
#Let's see which building class makes the most of market share:

nyc['BUILDING CLASS CATEGORY'].value_counts().head(10)
#Let's investigate that deeper and split it into monthky transtactions.



jan_dwellings = jan['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

JAN = jan[jan_dwellings]

feb_dwellings = feb['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

FEB = feb[feb_dwellings]

mar_dwellings = mar['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

MAR = mar[mar_dwellings]

apr_dwellings = apr['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

APR = apr[apr_dwellings]

may_dwellings = may['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

MAY = may[may_dwellings]

jun_dwellings = jun['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

JUN = jun[jun_dwellings]

jul_dwellings = jul['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

JUL = jul[jul_dwellings]

aug_dwellings = aug['BUILDING CLASS CATEGORY'].str.contains('01 ONE FAMILY DWELLINGS')

AUG = aug[aug_dwellings]

print('Number of one family dwellings transactions per month')

print('JAN:', len(JAN))

print('FEB:', len(FEB))

print('MAR:', len(MAR))

print('APR:', len(APR))

print('MAY:', len(MAY))

print('JUN:', len(JUN))

print('JUL:', len(JUL))

print('AUG:', len(AUG))
print('JAN:')

print(JAN['BOROUGH'].value_counts())

print('FEB:')

print(FEB['BOROUGH'].value_counts())

print('MAR:')

print(MAR['BOROUGH'].value_counts())

print('APR:')

print(APR['BOROUGH'].value_counts())

print('MAY:')

print(MAY['BOROUGH'].value_counts())

print('JUN:')

print(JUN['BOROUGH'].value_counts())

print('JUL:')

print(JUL['BOROUGH'].value_counts())

print('AUG:')

print(AUG['BOROUGH'].value_counts())
#Let's build a data frame showing number of transactions in Boroughs in different months



JAN_list = JAN['BOROUGH'].value_counts().tolist()

FEB_list = FEB['BOROUGH'].value_counts().tolist()

MAR_list = MAR['BOROUGH'].value_counts().tolist()

APR_list = APR['BOROUGH'].value_counts().tolist()

MAY_list = MAY['BOROUGH'].value_counts().tolist()

JUN_list = JUN['BOROUGH'].value_counts().tolist()

JUL_list = JUL['BOROUGH'].value_counts().tolist()

AUG_list = AUG['BOROUGH'].value_counts().tolist()
#list of Boroughs for our table

BOROUGH = ['Queens','Staten Island','Brooklyn','Bronx ','Manhattan']
#building the aggregated table from all monthly lists

borough_sales = pd.DataFrame(

    {'BOROUGH': BOROUGH,

     'JANUARY': JAN_list,

    'FEBRUARY': FEB_list,

    'MARCH': MAR_list,

    'APRIL': APR_list,

    'MAY': MAY_list,

    'JUNE': JUN_list,

    'JULY': JUL_list,

    'AUGUST': AUG_list,

    }, columns = ['JANUARY','FEBRUARY','MARCH','APRIL','JUNE','JULY','AUGUST','BOROUGH'])

borough_sales.set_index('BOROUGH', inplace=True)

borough_sales.head()
JAN[JAN['BOROUGH']=='Manhattan'].sort_values(by='SALEPRICE', ascending=0)
# total value of tranactions per building category



jan.groupby('BUILDING CLASS CATEGORY').SALEPRICE.sum()
# Top Boroughs



jan['BOROUGH'].value_counts()
# Top Neighborhoods



jan['NEIGHBORHOOD'].value_counts().head(20)
# Top Building categories



jan['BUILDING CLASS CATEGORY'].value_counts().head(20)
# Top 10 most expensive properties in January 2017

jan.sort_values('SALEPRICE', ascending=False).head(10)

