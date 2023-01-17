# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
from bokeh.io import export_png
import matplotlib.pyplot as plt
import matplotlib as matplot


%matplotlib inline
pd.set_option('display.max_columns', None)  
df = pd.read_csv('../input/autos.csv', encoding='latin-1')
det = pd.read_csv('../input/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv')
df.head(5).to_html('table.html')
df.head(5)
df.tail(5).to_html('table1.html')
df.tail(5)
df.columns
# finding rows where seller is not labels 'privat'
df.loc[df['seller'] != 'privat'].shape
df.loc[df['offerType'] != 'Angebot'].shape
df = df.drop(['seller','name'],axis=1)
df.head(3)
df.shape
# Counting the number of rows with NaN values
np.count_nonzero(df.isnull().values)
# there are many ways to find the # of rows with NaN or missing values
# in this case, we have 110572 car sale entries with NaN values
df1 = df[df.isnull().any(axis=1)] # axis=1 specifies rows instead of columns
df1.shape
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df.shape
df.dropna(subset=['model','brand'],axis=0, how='any', thresh=None, inplace=False).head(5)
df.loc[df['offerType'] != 'Angebot','offerType'].values
df = df.loc[df['offerType'] == 'Angebot']
df = df.drop('offerType',axis=1)
df.shape
df = df[df.powerPS != 0]
df.dropna(subset=['powerPS'],axis=0, how='any', thresh=None, inplace=False)
df.shape
df['gearbox']
def remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list
remove(df['gearbox'])
print('gearbox: ', remove(df['gearbox']))
print('vehicle type: ', remove(df['vehicleType']))
df = df.replace(['manuell','automatik','kleinwagen', 'limousine', 'cabrio', 'kombi', 'ander','ja','nein'], 
                     ['manual', 'automatic','hatckback','sedan','convertible','van','other','yes','no']) 
df.head(3)
df2 = df.head(15)
df['yearOfRegistration'].plot.hist(bins=100)
print("max: ", df['yearOfRegistration'].max())
print("min: ", df['yearOfRegistration'].min())
print("mean: ", df['yearOfRegistration'].mean())
print("mode: ", df['yearOfRegistration'].mode())
df['brand'].value_counts().plot(kind='barh')
df['brand'].value_counts().plot(kind='pie')
df['vehicleType'].value_counts().plot(kind='pie')
df['vehicleType'].value_counts()
df['gearbox'].value_counts().plot(kind='pie')
df['gearbox'].value_counts()
len(df[(df['price'] > 500000)])
## df = df[(df['price']< 100000) & (df['price']>0)]
df['price'].max()
len(df)
df[(df['price']< 500000) & (df['price']>0)]['price'].plot.hist(bins=200)
fig,ax = plt.subplots()
ax.set_yscale('log')
df[(df['price']< 500000) & (df['price']>0)]['price'].plot.hist(bins=200)
fig,ax = plt.subplots()
ax.set_yscale('log')
df = df[df['price']< 200000]
df['price'].plot.hist(bins=200)
len(df)
df.head(10)
df.tail(10)