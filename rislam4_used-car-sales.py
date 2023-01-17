# Importing library
import numpy as np
import pandas as pd
# Reading the csv file
filename = "/kaggle/input/used-cars-database-50000-data-points/autos.csv"
df = pd.read_csv(filename, encoding= 'Windows-1252')
df.head()
print(df.columns)
df.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'abtest', 'vehicle_type', 'registration_year', 
              'gear_box', 'power_PS', 'model','odometer', 'registration_month', 'fuel_type', 'brand','unrepaired_damage',
              'ad_created', 'Num_of_Pictures', 'postal_code', 'lastSeen']
df.columns
# rename method to make specific changes
df.rename({'lastSeen':'last_seen'}, axis= 1, inplace= True)
df.columns
df.info()
df.isnull().sum() * 100 / df.shape[0]
# To see the column datan types
df.dtypes
df.head()
# for the price column
# defining a function
def replace(x):
    x = x.replace('$','')
    x = x.replace(',','')
    return x

# applying that function
df['price'] = df['price'].apply(replace)
df['price'].head()
# For the odometer column 
def replace_odo(x):
    x = x.replace(',','')
    x = x.replace('km','')
    return x

df['odometer'] = df['odometer'].apply(replace_odo) 
# Assigning the data type in a dictionary
dic = {'price': 'float', 'odometer': 'float'}

# Changing the data type
df = df.astype(dic)
df.dtypes
# Changing the name to make it more readable
df.rename({'price':'price_in_dollar', 'odometer':'odometer_in_km'}, axis= 1, inplace= True)
df.columns
# Let's check the unique value in the registration_month column
df['registration_month'].unique()
d = {0:'unknown',1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
column = []
for item in df['registration_month']:
    if item in d:
        column.append(d[item])
df['reg_month_in_words'] = column
df.head(3)
# We could assign all the rearragned column names to the data frame but i am gonna try the drop and insert method here.
# Assigning the columns in a variable
year = df['registration_year']
month = df['registration_month']
month_words = df['reg_month_in_words']

# Dropping all the columns
df.drop(labels=['registration_year'], axis=1, inplace= True)
df.drop(labels=['registration_month'], axis=1, inplace= True)
df.drop(labels=['reg_month_in_words'], axis=1, inplace= True)

# Inserting to our expected position
df.insert(6, 'registration_year', year)
df.insert(7, 'registration_month', month)
df.insert(8, 'reg_month_in_words', month_words)
df.head(3)
df[['date_crawled','ad_created','last_seen']].head()
# Since the first 10 numbers are date, we will just apply the slicing
for item in ['date_crawled','ad_created','last_seen']:
    df[item] = df[item].str[:10]
df.head(3)
# Changing the format of these columns to datetime

df['date_crawled'] = pd.to_datetime(df['date_crawled'], format= "%Y-%m-%d", dayfirst= True )
df['ad_created'] = pd.to_datetime(df['ad_created'], format= "%Y-%m-%d", dayfirst= True )
df['last_seen'] = pd.to_datetime(df['last_seen'], format= "%Y-%m-%d", dayfirst= True )
df.dtypes
# if we want we can just extract month, year or day from these
df['date_crawled'][0].year
df.head(3)
df['price_in_dollar'].value_counts().sort_index(ascending= False).head(20)
# Taking data between 0 to 10 millin
df = df[df['price_in_dollar'].between(0,10000000)]
df.shape
df['registration_year'].value_counts().sort_index(ascending= False).head(20)
df['registration_year'].value_counts().sort_index(ascending= True).head(20)
# Taking the year from 1900 to 2018
df = df[df['registration_year'].between(1900,2018)]
df.shape
df.head(3)
# Determining the top car brand in term of their sales number
top_20 = df['brand'].value_counts(ascending= False).head(20)
top_20
# making a list of top 20 brand to run the loop
brand_list = list(top_20.index )
brand_list
# Empty dictionary
mean_dic = {}

# Running a loop to entry the mean average
for item in brand_list:
    mean = df[df['brand'] == item]['price_in_dollar'].mean()
    mean_dic[item] = mean
        
mean_dic
# Making a series to create the dataframe
mean_series = pd.Series(mean_dic)
mean_series
brand_mean = pd.DataFrame(mean_series, columns=['mean_price'])
brand_mean
# now calculating the avg mileage of those car
dic_mileage = {}
for item in brand_list:
    mean = df[df['brand'] == item]['odometer_in_km'].mean()
    dic_mileage[item] = mean
        
print(dic_mileage)
# Creating series
mileage_series = pd.Series(dic_mileage)

# Dataframe
brand_mileage = pd.DataFrame(mileage_series, columns= ['mileage'])
brand_mileage
brand_count = dict(top_20)
brand_count
# Creating series
count_series = pd.Series(dict(top_20))

# Creating the Dataframe
brand_count = pd.DataFrame(count_series, columns= ['count'])
brand_count
brand_price_mileage_count = pd.concat([brand_mean, brand_mileage, brand_count], axis=1)
brand_price_mileage_count
# Sorting by number of sales unit
brand_price_mileage_count.sort_values(by= 'count', ascending= False)
def green(val):
    color = 'green'
    return 'color: %s' % color
# Coloring certain cell
brand_price_mileage_count[:5].style.applymap(green, subset=pd.IndexSlice[['volkswagen','opel'], ['mean_price','mileage']])
# Coloring certain cell
brand_price_mileage_count[:5].style.applymap(green, subset=pd.IndexSlice['bmw':'audi', ['mean_price','mileage']])
