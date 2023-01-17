import numpy as np
import pandas as pd
#visualisation 
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import date
from datetime import datetime
import datetime as dt
import gc
import json

from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
import math

from scipy import stats
from ipywidgets import interact
#for warnings
import warnings
warnings.filterwarnings('ignore')
#download the files and become in dataframe(pandas)
path = '../input/salesdb-grocery/salesdb_grocery_market/'

countries = pd.read_csv(path + 'countries.csv', sep=";")
cities = pd.read_csv(path + "cities.csv", sep=";")
employes = pd.read_csv(path + "employes.csv", sep=";")
products = pd.read_csv(path + "products.csv", sep=";")
categories = pd.read_csv(path + "categories.csv", sep=";")
customers = pd.read_csv(path + "customers.csv", sep=";")
sales = pd.read_csv(path + "sales.csv", sep=";")

#download file created by myself with cities and location 
#latitude and longitude
cities_loc = pd.read_csv('../input/cities-loc-sales-grocery-market/Cities_location')
cities_loc = cities_loc.drop(columns='Unnamed: 0')
print("  -Countries: \nShape:", countries.shape)
print("Head:")
print(countries.head(),"\n")
print("  -Cities: \nShape:", cities.shape)
print("Head:")
print(cities.head(),"\n")
cities_countries=pd.merge(cities, countries, on=['CountryID','CountryID'])
cities_countries.head()
print ("countries:", cities_countries['CountryName'].drop_duplicates().values[0])
#cities_countries.hist(column='CountryName')
cities_countries=cities_countries.drop(columns=['CountryID','CountryCode'])
#cities_countries.head()
print("  -Employes: \nShape:", employes.shape)
print("Head:")
print(employes.head(),"\n")
Gender=employes.groupby('Gender')['Gender'].count()
fig, ax=plt.subplots(figsize=(10,4))
Gender.plot.bar(ax=ax)
plt.title('Employees\' Gender distribution')
plt.show()

# We create new features Age
today=date.today()
Ag=[]
for i in range(0,(employes.shape[0])):
    aux = employes['BirthDate'][i]
    aux = aux[0:10]
    aux = datetime.strptime(aux, '%Y-%m-%d')
    aux = int(((today-aux.date()).days)/365.25)
    Ag.append(aux)
    
employes['Age']=Ag
#we create the histogram
def group_Age(i):
    if i<20: return ('<20')
    elif i<25: return ('20-24')
    elif i<30: return ('25-29')
    elif i<35: return ('30-34')
    elif i<40: return ('35-39')
    elif i<45: return ('40-44')
    elif i<50: return ('45-49')
    elif i<55: return ('50-54')
    elif i<60: return ('55-59')
    elif i<65: return ('60-64')
    else: return ('>65')
employes['Age_aux']=list(map(group_Age, list(employes['Age'])))
fig,ax=plt.subplots(figsize=(10,4))
pd.DataFrame(employes.groupby('Age_aux')['Age_aux'].count()).plot.bar(ax=ax)
plt.title('Employe\'s Ages distribution')
plt.show()
Y=[]
for i in range(0,(employes.shape[0])):
    aux = employes['HireDate'][i]
    aux = aux[0:10]
    aux = datetime.strptime(aux, '%Y-%m-%d')
    aux = int(((today-aux.date()).days)/365.25)
    Y.append(aux)
    
employes['YearsHired']=Y
YearHired=employes.groupby('YearsHired')['YearsHired'].count()
YearHired=pd.DataFrame(YearHired)
YearHired=YearHired.rename(columns={'YearsHired':'#employes'})
print(YearHired.T)
employes['Age_aux']=list(map(group_Age, list(employes['Age'])))
fig,ax=plt.subplots(figsize=(10,4))
YearHired.plot.bar(ax=ax)
plt.title('Number of year that the employees are working in the company distribuion')
plt.show()
employes=employes.drop(columns=['HireDate','BirthDate','FirstName','MiddleInitial','LastName'])
print("  -Products: \nShape:", products.shape)
print("Head:")
print(products.head(),"\n")
print("  -Categories: \nShape:", categories.shape)
print("Head:")
print(categories.head(),"\n")
# Merge these two tables in order to know 
# what type of categories we have.
products =pd.merge(products, categories, on=['CategoryID','CategoryID'])
products = products.drop(columns=['CategoryID','ModifyDate','IsAllergic', 'VitalityDays','Class', 'Resistant'])
products.head()

products[products['CategoryName']=='Shell fish'].head()
#products with price equal 0
print('number of products with price 0:',len(products[products['Price']==0]))
#try to do hitogram
price=pd.DataFrame(products.groupby('Price')['Price'].count())

len(price)
# 5 of the products less expensive and the price
products=products.sort_values(by=['Price'])
products.head(5)
# 5 of the products more expensive and the price
products.tail(5)
#convert Price string to float
products['Price']=products.Price.str.replace(',', '.')
products['Price']=pd.to_numeric(products['Price'])
fig,axes=plt.subplots(nrows=2,figsize=(10,8))
axes[0].set_title('Products\'s Prices Histogram')
products['Price'].plot.hist(bins=20,rwidth=0.9, ax=axes[0])

products['Price'].plot.density(ax=axes[1])
axes[1].set_title('Products\'s Prices Density Function')
plt.show()
distribution='uniform'
min_price=min(products['Price'])
max_price=max(products['Price'])
stats.kstest(list(products['Price'].sort_values()), distribution, args=(0,100))
#Shape of countries
print("- CUSTOMERS:")
print(customers.shape)
print(customers.head())
#merge Customers and cities 
customers=pd.merge(customers, cities, on=['CityID', 'CityID'])
city_customers = customers.groupby('CityName')['CustomerID'].count()
city_customers = pd.DataFrame(city_customers)
city_customers = city_customers.rename(columns={"CustomerID":"# customers"})
city_customers=city_customers.sort_values(by=['# customers'])
print("the 5 cities with the least customers:")
city_customers.head(5)
print("the 5 cities with the most customers:")
city_customers.tail(5)
#histogram with the cities and number of customeres
x=list(np.arange(1,96,6))
city_customers=city_customers.sort_values(by=['# customers'])
city_customers['Range']=range(1,len(city_customers)+1)
aux=city_customers[city_customers['Range'].isin(x)]
labels=list(aux.index)
city_customers=city_customers.drop(columns=['Range'])

fig,ax=plt.subplots(figsize=(10,4))
city_customers.plot.bar(ax=ax)
plt.xticks(x, labels, rotation=100)
plt.ylim(900,1200)
plt.title('# customers per city distribution')
plt.show()
city_cust_mean=city_customers.mean()
city_cust_std=city_customers.std()
print("mean: %f" %city_cust_mean)
print("std: %f" %city_cust_std)
#Crea Z=(x-E(x))/std
city_cust_norm=(city_customers-city_cust_mean)/city_cust_std
fig,axes=plt.subplots(nrows=2,figsize=(10,8))
city_cust_norm.plot.hist(bins=10, ax=axes[0])
axes[0].set_title('Normalised Distribution')
city_cust_norm.plot.density(ax=axes[1])
axes[1].set_title('Density Function')
plt.show
distribution='norm'
stats.kstest(list(city_cust_norm['# customers']), distribution, args=(0,1))
city_customers = customers.groupby('CityName')['CityName'].count()
cities=[[cit, val] for cit,val in city_customers.iteritems()]
scale=0.25
mymap = Basemap(width=10000000,height=6000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)


def cleanloc(string):
    longitude = float(string.split(',')[1][:-1])
    latitude = float(string.split(',')[0][1:])
    return (longitude, latitude)
d = {cities_loc['cities'][i]: cleanloc(cities_loc['loc'][i]) for i in cities_loc.index}
plt.figure(figsize=(19,20))
mymap.bluemarble()
for (city,count) in cities:
    x, y = mymap(*d[city])
    mymap.plot(x,y,marker='o',color='Red',markersize=int(math.sqrt(count))*scale)
plt.show()
#zipcode
print('number of zipcode:',len(customers['Zipcode'].drop_duplicates()))
print('number of cities:', len(customers['CityName'].drop_duplicates()))

# clean data customers
customers=customers.drop(columns=['FirstName','MiddleInitial','LastName','Address',
                       'CityID','Address', 'CountryID', 'Zipcode'])
customers.head()
print("  -Sales: \nShape:", sales.shape)
print("Head:")
print(sales.head(),"\n")
# sales with customer
sales_customer=pd.merge(sales, customers, on=['CustomerID', 'CustomerID'])
sales_customer_prod=pd.merge(sales_customer, products, on=['ProductID', 'ProductID'])
sales_customer_prod=sales_customer_prod.drop(columns=['TotalPrice', 'TransactionNumber',
                                         'CategoryName'])
                          
#PriceTotal=Quantity*Price
sales_customer_prod['PriceTotal']=sales_customer_prod['Price']*sales_customer_prod['Quantity']
#DiscountTotal=Discount*Price
sales_customer_prod['Discount']=sales_customer_prod['Discount'].fillna(0)
sales_customer_prod['DiscountTotal']=sales_customer_prod['Discount']*sales_customer_prod['Price']
#Total=PriceTotal-TotalDiscount
sales_customer_prod['Total']=sales_customer_prod['PriceTotal']-sales_customer_prod['DiscountTotal']
sales_customer_prod=sales_customer_prod.drop(columns=['Discount'])
sales_customer_prod.head()
a=pd.DataFrame(sales_customer_prod.groupby('ProductName')['Quantity'].sum().sort_values())
x=list(np.arange(5,len(a)+1, 40))
a['Range']=np.arange(1,len(a)+1)
labels=a[a['Range'].isin(x)].index

print('the worst seller: %s' %a[a['Quantity']==min(a.Quantity)].index[0] )
print('the best seller: %s' %a[a['Quantity']==max(a.Quantity)].index[0] )
fig,ax=plt.subplots(figsize=(10,4))
pd.DataFrame(a['Quantity']).plot.bar(ax=ax)

plt.xticks(x, labels, rotation=100)
plt.title('number of products are sold')
plt.ylim(180000,200000)
plt.show()
# working with SalesDate. SalesDate contains NaN, so we are going 
# to work with sales_aux dateframe.
sales_aux=sales_customer_prod[sales_customer_prod['SalesDate'].notnull()]
sales_aux['SalesDate'] = pd.to_datetime(sales_aux['SalesDate'], format='%Y-%m-%d %H:%M:%S.%f')
sales_aux['S_Date']=sales_aux['SalesDate'].dt.date
a_1=sales_aux.groupby(['S_Date', 'SalesPersonID'])['SalesID','CustomerID','CityName'].count()
a_2=sales_aux.groupby(['S_Date', 'SalesPersonID'])['CustomerID','CityName'].nunique()
a_2=a_2.rename({'CustomerID':'CustomerUnique','CityName':'CityUnique'}, axis=1)
a_3=sales_aux.groupby(['S_Date', 'SalesPersonID'])['Quantity','Total'].sum()
a=pd.concat([a_1, a_2,a_3], axis=1)
a_original=a
def set_num_employees(num_employees):
    global a
    listemployees=list(np.arange(1,num_employees+1)) 
    a=a_original
    a=a[np.in1d(a.index.get_level_values(1), listemployees)]
def represent_data_1(features):
    if features=='CustomerID':
        fig, axes=plt.subplots(nrows=3, figsize=(10,12))
        print("mean(sales by SalesPersonID and Day): %f" %a[features].mean())
        print("std(sales by SalesPersonID and Day): %f" %a[features].std())
        print("mean(sales by SalesPersonID and Day): %f" %a['CustomerUnique'].mean())
        print("std(sales by SalesPersonID and Day): %f" %a['CustomerUnique'].std())
        a_aux=pd.DataFrame(a[features].unstack())
        plt.suptitle('%s per seller and day'%features, size=20)
        # Plot 1.
        x=list(np.arange(1,len(a_aux)+1, 10))
        a_aux['Range']=np.arange(1,len(a_aux)+1)
        labels_aux=list(a_aux[a_aux['Range'].isin(x)].index.astype(str))
        a_aux=a_aux.drop(columns=['Range'])
        a_aux.plot.line(ax=axes[0])
        labels=axes[0].set_xticklabels(labels_aux, rotation=30)
        # Plot 2.
        if len(a[features].drop_duplicates())>1:
            a[features].plot.density(ax=axes[1])
        # Plot 3.
        a['CustomerUnique'].unstack().plot.line(ax=axes[2])
        axes[2].set_title('customer unique')
        labels=axes[2].set_xticklabels(labels_aux, rotation=30)
        plt.show()
    elif features=='CityName':
        fig, axes=plt.subplots(nrows=3, figsize=(10,12))
        print("mean(sales by SalesPersonID and Day): %f" %a[features].mean())
        print("std(sales by SalesPersonID and Day): %f" %a[features].std())
        print("mean(sales by SalesPersonID and Day): %f" %a['CityUnique'].mean())
        print("std(sales by SalesPersonID and Day): %f" %a['CityUnique'].std())
        a_aux=pd.DataFrame(a[features].unstack())
        plt.suptitle('%s per seller and day'%features, size=20)
        # Plot 1.
        x=list(np.arange(1,len(a_aux)+1, 10))
        a_aux['Range']=np.arange(1,len(a_aux)+1)
        labels_aux=list(a_aux[a_aux['Range'].isin(x)].index.astype(str))
        a_aux=a_aux.drop(columns=['Range'])
        a_aux.plot.line(ax=axes[0])
        labels=axes[0].set_xticklabels(labels_aux, rotation=30)
        # Plot 2.
        if len(a[features].drop_duplicates())>1:
            a[features].plot.density(ax=axes[1])
        # Plot 3.
        a['CityUnique'].unstack().plot.line(ax=axes[2])
        axes[2].set_title('city unique')
        labels=axes[2].set_xticklabels(labels_aux, rotation=30)
        plt.show()
    else:
        fig, axes=plt.subplots(nrows=2, figsize=(10,8))
        print("mean(sales by SalesPersonID and Day): %f" %a[features].mean())
        print("std(sales by SalesPersonID and Day): %f" %a[features].std())
        plt.suptitle('%s per seller and day'%features, size=20)
        a_aux=pd.DataFrame(a[features].unstack())
        # Plot 1.
        x=list(np.arange(1,len(a_aux)+1, 10))
        a_aux['Range']=np.arange(1,len(a_aux)+1)
        labels_aux=list(a_aux[a_aux['Range'].isin(x)].index.astype(str))
        a_aux=a_aux.drop(columns=['Range'])
        a_aux.plot.line(ax=axes[0])
        labels=axes[0].set_xticklabels(labels_aux, rotation=30)
        # Plot 2.
        if len(a[features].drop_duplicates())>1:
            a[features].plot.density(ax=axes[1])
        plt.show()
        
interact(set_num_employees, num_employees=list(np.arange(1,24)))
interact(represent_data_1, features=['SalesID','CustomerID', 'CityName','Quantity', 'Total'])
# Checking the NaN which are in SalesDate
len(sales_customer_prod[sales_customer_prod['SalesDate'].isna()])/len(sales_customer_prod)*100
sales_customer_prod_aux=sales_customer_prod.dropna(subset=['SalesDate'])
sales_customer_prod_aux['SalesDate']=pd.to_datetime(sales_customer_prod_aux['SalesDate'],format='%Y-%m-%d %H:%M:%S.%f')
#Features
sales_customer_prod_aux['s_weekday']=sales_customer_prod_aux['SalesDate'].dt.weekday
sales_customer_prod_aux['s_hour']=sales_customer_prod_aux['SalesDate'].dt.hour
sales_customer_prod_aux['s_day']=sales_customer_prod_aux['SalesDate'].dt.day
sales_customer_prod_aux['s_month']=sales_customer_prod_aux['SalesDate'].dt.month
sales_customer_prod_aux['s_year']=sales_customer_prod_aux['SalesDate'].dt.year
def freq_feature(data, column):
    aux = data.groupby(column)['Quantity'].sum()
    aux = pd.DataFrame(aux)
    return (aux)
names_freq = ['s_hour', 's_day', 's_weekday', 's_month','s_year']
l = []#this is a list of dataframes in order to collect for all frequencies
for i in names_freq:
    aux=freq_feature(sales_customer_prod_aux,i)
    l.append(aux)
#aux.plot.bar()
def represent_data_4(freq):
     for i in l:
            if i.index.name==freq:
                i.plot.line()
                plt.title(freq)
                plt.show()

interact(represent_data_4, freq=['s_hour', 's_day', 's_weekday', 's_month','s_year'])
print (sales_customer_prod_aux['SalesDate'].min())
print (sales_customer_prod_aux['SalesDate'].max())