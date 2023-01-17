import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn import preprocessing as pro



from scipy.stats import skew

from scipy.stats import kurtosis



#colour palette for graphics

current_palette = sns.color_palette("GnBu_r")

sns.set_palette(current_palette)



# load data in pandas data frame

data = pd.read_csv('../input/MELBOURNE_HOUSE_PRICES_LESS.csv')



# add new attribute columns for price in millions of dollars and log of price

data['PriceMillions'] = data['Price']/1000000

data['LogPrice'] = data['Price'].apply(np.log)
data.describe()
data.mode()
data['Propertycount'].head()



# check data types

data.dtypes
# as the above analysis shows date is stored incorrectly. Therefore, I used the Pandas .to_datetime() 

# function to convert the date from an object to a date time

data['Date'] = pd.to_datetime(data['Date'])



# I've also conducted some feature engineering by adding month and year as attributes 

# add month column to data frame

data['Month'] = data['Date'].dt.month



# add year column to data frame 

data['Year'] =  data['Date'].dt.year          
# convert postcode from int to string

data['Postcode'] = data['Postcode'].apply(str)



# check conversions successful 

data.dtypes
# label encorder variable 

label = pro.LabelEncoder() 



# columns to be encoded

values = ('TypeEncoded', 'MethodEncoded', 'SellerEncoded', 'CouncilEncoded', 'SuburbEncoded', 'RegionEncoded')



# creation of new encoded variables 

data['TypeEncoded'] = data['Type']

data['MethodEncoded'] = data['Method']

data['SellerEncoded'] = data['SellerG']

data['CouncilEncoded'] = data['CouncilArea']

data['SuburbEncoded'] = data['Suburb']

data['RegionEncoded'] = data['Regionname']



# label encoding for-loop 

for i in values:

    label.fit(list(data[i].values)) 

    data[i] = label.transform(list(data[i].values))
# Check for Missing Values 

data.isna().sum()
# dropping missing price values 

data.dropna(0, inplace=True)
# Check for missing values 

data.isna().sum()
# create data frame of values which are outliers for price

outliersprice = data.loc[data['Price'] > 2120000]



# return sorted list from large to small of outliers

outliersprice.sort_values(by=['Price'], ascending=False)
# locate values which are lowerbound outliers for Rooms

lowerrooms = data.loc[data['Rooms'] < 1.5]



# I don't think this value is useful in determining whether the number of bedrooms is an outlier as it would 

# exclude all one bedroom homes. Consequently, I have not used this in my analysis.
# Create data frame of values which are upperbound outliers for 'Rooms'

upperrooms = data.loc[data['Rooms'] > 5.5]



# Return sorted list from large to small of outliers

upperrooms.sort_values(by=['Rooms'], ascending=False)
# Create data frame of values which are upperbound outliers for 'Propertycount'



upperpropertycount = data.loc[data['Propertycount'] > 19473]



# Again not a very useful indicator. The number listed for the suburb of Resevior is correct. 

# Consequently, I have not used this upperbound in my analysis."""  
# Create data frame of values which are upperbound outliers for 'Distance'



upperdistance = data.loc[data['Distance'] > 31.25]



# Return sorted list from large to small of outliers

upperdistance.sort_values(by=['Distance'], ascending=False)
# set 35 Bevis St to correct sale price ##

data.at[61142, 'Price'] = 904500



# set 5 Cottswold Place bedroom value to 4 ##

data.at[55467, 'Rooms'] = 4



# set 507 Orrong Rd Arrandale bedroom value to 6 ##

data.at[7825, 'Rooms'] = 6 



# set 1 Beddoe Ave bedroom value to 8 ##

data.at[49016, 'Rooms'] = 8 



# deletion of 20 Harrison St Mitcham listing ##

data.drop(59741, inplace=True)



# deletion of 213 station Road Melton ##

data.drop(21337, inplace=True)



# deletion of 84 Flemington Rd Parkville ##

data.drop(29421, inplace=True)



# deletion of 225 McKean St, Fitzroy North ##

data.drop(7452, inplace=True)



# deletion of 445 Warrigal Rd Burwood  ##

data.drop(55673, inplace=True)



# deletion of 10 Berkley St Hawthorn ##

data.drop(39847, inplace=True)



# deletion of 5 Ball Ct, Bundoora

data.drop(27354, inplace=True)





# summary statistics of continuous and discrete data 

# create new data frame for scatterplot containing only continuous and discrete data 



numdata = pd.DataFrame()

numdata['Rooms'] = data['Rooms']

numdata['PriceMillions'] = data['PriceMillions']

numdata['Propertycount'] = data['Propertycount']

numdata['Distance'] = data['Distance']



numdata.describe()

# Histograms of price 

# Normal distribution with same std and mean as price data 

normal = np.random.normal(loc=.9978982,scale=.5934989,size=48433)



# Histogram settings 

plt.figure(figsize = (24, 12))

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne Housing Prices', fontsize=40)

plt.ylabel('Density')

plt.xticks(np.arange(12), ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'))

price = sns.distplot(data['PriceMillions'], bins=25, label='Price Millions', axlabel='Price (Millions AUD)', hist_kws=dict(edgecolor="k", linewidth=2))

normalprice = sns.distplot(normal, hist=False, bins=25, color='#ce1256', label='Normal distribution')

normalprice.tick_params(labelsize=25)

plt.show()



# mean = 0.997898

# std = 0.593499

# observations = 48433
# data['logPrice'] stats

# mean 13.681025

# std 0.497138



normal = np.random.normal(loc=13.681025, scale=0.497138,size=48433)



# Log transformed histrogram

plt.figure(figsize = (24, 12))

plt.rcParams["axes.labelsize"] = 25

plt.title('Melbourne Housing Prices', fontsize=30)

plt.ylabel('Density')

loprice = sns.distplot(data['LogPrice'], bins=25, label='Price Millions', axlabel='Log(Price)', hist_kws=dict(edgecolor="k", linewidth=2))

normallog = sns.distplot(normal, hist=False, color='#ce1256', label='Normal distribution')

normallog.tick_params(labelsize=15)

plt.show()
# Boxplot of price 

plt.figure(figsize = (15, 12))

plt.rcParams["axes.labelsize"] = 15

pricebox = sns.boxplot(y = 'PriceMillions', orient="h", color='#9ecae1', saturation=1, data = data)

plt.xticks(np.arange(12), ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'))

pricebox.set(xlabel='Price (Millions AUD)')

plt.title('Melbourne Housing Prices', fontsize=17)

pricebox.tick_params(labelsize=10)

plt.show()
# Use seaborn pairplot to produce a scatterplot matrix of numeric variables

plt.rcParams["axes.labelsize"] = 12

sns.pairplot(numdata)

plt.show()
# Correlation matrix of price with other attributes 



cormatrix = data.corr()

cormatrix.drop('Price', inplace=True)

cormatrix.drop('PriceMillions', inplace=True)

cormatrix.drop('LogPrice', inplace=True)



cormatrix = cormatrix.sort_values(by=['Price'])



print(cormatrix['Price'])
from scipy.stats import pearsonr 





values = ('TypeEncoded', 'Distance', 'SuburbEncoded', 'CouncilEncoded', 'CouncilEncoded', 'Propertycount',

         'SellerEncoded', 'Month', 'MethodEncoded', 'Year', 'RegionEncoded', 'Rooms')



for i in values:

    print('The correlation and statitstical signifgance between %s and price is %s' %

          (i, pearsonr(data['LogPrice'], data[i])))
plt.figure(figsize = (24, 20))

plt.rcParams["axes.labelsize"] = 35

roomsprice = sns.regplot(x=data['Rooms'], y=data['PriceMillions'], line_kws={"color": "#f768a1"})

plt.title('Melbourne Housing Prices vs Number of Rooms', fontsize=45)

plt.ylabel('Price (Millions AUD)')

roomsprice.tick_params(labelsize=25)
# Boxplot of price by suburb 

plt.figure(figsize = (30, 30))

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne Housing Prices by Council', fontsize=40)

councilprice = sns.boxplot(y = 'CouncilArea', x = 'PriceMillions', orient="h",  data = data, palette=current_palette)

councilprice.set(xlabel='Price (Millions AUD)')

councilprice.set(ylabel='Council')

councilprice.tick_params(labelsize=15)

plt.show()



# Boxplot of price by region name 

plt.figure(figsize = (30, 30))

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne Housing Prices by Region', fontsize=40)

regionprice = sns.boxplot(y = 'Regionname', x = 'PriceMillions', orient="h",  data = data, palette=current_palette)

regionprice.set(xlabel='Price (Millions AUD)')

regionprice.set(ylabel='Region')

regionprice.tick_params(labelsize=18)

plt.show()
# Create data frame of sales per month

count = pd.DataFrame({'Count': data['Month'].value_counts()})

count.sort_index(inplace=True) 

count.columns = ['Count']

months = pd.DataFrame({'Month': ['Januray','Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']})

months.index += 1 



monthcount = count.join(months, how='outer')



# Plot graph of total sale per month 

plt.figure(figsize = (24, 12))

plt.rcParams["axes.labelsize"] = 25

plt.title('Total Sales Per Month', fontsize=30)

plt.ylabel('Count')

plt.xticks(np.arange(13), ('', 'Januray','Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))

totalsales = sns.lineplot(data=count)

totalsales.set(xlabel='Month')

totalsales.tick_params(labelsize=12)

plt.show()
# Create datae frame with average sales price per month

monthmean = data.groupby('Month')['PriceMillions'].mean()



# Lineplot of average sale price per month 

plt.figure(figsize = (24, 12))

plt.rcParams["axes.labelsize"] = 25

plt.title('Average Sale Price Per Month', fontsize=30)

plt.ylabel('Price in Millions (AUD)')

plt.xticks(np.arange(13), ('', 'Januray','Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))

avgsales = sns.lineplot(data=monthmean)

avgsales.set(xlabel='Month')

avgsales.tick_params(labelsize=12)

plt.show()

# Average sale price per seller

sellermean = data.groupby('SellerG')['PriceMillions'].mean()
# Total sales per seller

sellcount = data['SellerG'].value_counts()

print('Total number of unique sellers:'+ ' ' + str(sellermean.count()))
# Boxplot of price by seller

plt.figure(figsize = (12, 100))

plt.rcParams["axes.labelsize"] = 25

plt.title('Sale Price vs Seller', fontsize=30)

boxseller = sns.boxplot(y = 'SellerG', x = 'PriceMillions', orient="h",  data = data, palette=current_palette)

boxseller.set(xlabel='Price (Millions AUD)')

boxseller.set(ylabel='Seller')

plt.show()
# Boxplot of price by suburb 

plt.figure(figsize = (30, 30))

plt.rcParams["axes.labelsize"] = 35

plt.title('Melbourne Housing Prices by Type', fontsize=40)

typeprice = sns.boxplot(y = 'Type', x = 'PriceMillions', data = data, palette=current_palette)

typeprice.set(xlabel='Price (Millions AUD)')

typeprice.tick_params(labelsize=25)

plt.show()