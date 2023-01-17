# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2017.csv')
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18)) #2 tane future arasında corelation 1 ise buınlar aralarında dogru orantılıdır.#figure size i ayarlarız 
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) #mt ondalık rakam sayısı,annot true üstüne rakamlar yazılsın mı
plt.show()

data.head(10)#first 10 rows
data.columns
# Line Plot

data['Economy..GDP.per.Capita.'].plot(kind = "line", color = "red",label = "Economy.GDP.per.Capita.",linewidth = 1, alpha = 0.6, grid = True,linestyle = ":")
data['Health..Life.Expectancy.'].plot( color = "green",label = "Health..Life.Expectancy.",linewidth = 1, alpha = 0.5, grid = True,linestyle = "-")
plt.legend(loc = "upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()
# Scatter Plot 
data.plot(kind='scatter', x='Economy..GDP.per.Capita.', y='Health..Life.Expectancy.',alpha = 0.7,color = 'orange')
plt.xlabel('Economy..GDP.per.Capita.')              # label = name of label 
plt.ylabel('Health..Life.Expectancy.')
plt.title('Economy..GDP.per.Capita.- Health.Life.Expectancy. Scatter Plot')            # title = title of plot
data['Economy..GDP.per.Capita.'].plot(kind = 'hist',bins = 30,figsize = (12,12))
data_frame = data[['Economy..GDP.per.Capita.']]  # data[['Defense']] = data frame
date_series=data['Economy..GDP.per.Capita.']
print(type(data_frame))
print(type(date_series))
# 1 - Filtering Pandas data frame
x = data['Health..Life.Expectancy.']>0.8
data[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Health..Life.Expectancy.']>0.8) & (data['Economy..GDP.per.Capita.']>1.5)]

# CLEANING DATA 2nd homework :)

data = pd.read_csv('../input/2017.csv')
data.head()  # head shows first 5 rows
# tail shows last 5 rows
data.tail()

# columns gives column names of features
data.columns
# shape gives number of rows and columns in a tuble
data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
# For example lets look frequency of pokemom types
print(data['Whisker.high'].value_counts(dropna =False))  # if there are nan values that also be counted

data.describe() #ignore null entries -only numeric types
data.info()

# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Generosity')
plt.show()
# Firstly I create new data from main data to explain melt nore easily.
data_new=data.head()
data_new
# lets melt
#frame=melt edilecek datamız
# id_vars = what we do not wish to melt/ yani ilgili sütunun yeni melt durumunda değişmeden aynı kalmasını saglamaktır
# value_vars = what we want to melt /değişecek kısım ilki variable ikincisi value yani aslında o satırı melt etmiş eritmiş olduk
melted = pd.melt(frame=data_new,id_vars = 'Country', value_vars= ['Economy..GDP.per.Capita.','Health..Life.Expectancy.'])
melted
# PIVOTING DATA
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Country', columns = 'variable',values='value')
# CONCATENATING DATA
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


data1 = data['Country'].head()
data2= data['Happiness.Score'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
# DATA TYPES
data.dtypes
# lets convert object(str) to categorical and int to float.
data['Country'] = data['Country'].astype('category')
data['Happiness.Rank'] = data['Happiness.Rank'].astype('float')

#new dttypes
data.dtypes
# MISSING DATA and TESTING WITH ASSERT
data.info()
# Lets drop nan values
# data1=data   # also we will use data to fill missing value so I assign it to data1 variable
# data1["Happiness.Rank"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# but not null variable-not avaible
assert  data['Happiness.Rank'].notnull().all() # returns nothing because we dont have nan values
data.columns

# 4. PANDAS FOUNDATION 

# Plotting all data 
data1 = data.loc[:,["Economy..GDP.per.Capita.","Health..Life.Expectancy.","Trust..Government.Corruption."]]
data1.plot()
# it is confusing
# subplots
data1.plot(subplots = True)
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="Economy..GDP.per.Capita.",y = "Trust..Government.Corruption.")
plt.show()
# INDEXING PANDAS TIME SERIES
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)#listeyi date time formatına çevirmemizi saglar
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of our data and add it a time list
data2 = data.head()
date_list = ["2000-01-10","2000-02-10","2000-03-10","2001-03-15","2001-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")#index değiştirmemizi saglar bu metod
data2 
# Now we can select according to our date index
print(data2.loc["2001-03-16"])
print(data2.loc["2000-03-10":"1993-03-16"])
# RESAMPLING PANDAS TIME SERIES
# We will use data2 that we create at previous part
data2.resample("A").mean()
# Lets resample with month
data2.resample("M").sum()
# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")#sayısal kısımlarda boş olan kısımları linear olarak doldur
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").mean().interpolate("linear")

# MANIPULATING DATA FRAMES WITH PANDAS
# read data
data = pd.read_csv('../input/2017.csv')
#data= data.set_index("#")#id futurune u index yaptık yani 0 yerine 1 den baslayacak
data.head()
# indexing using square brackets
data["Happiness.Rank"][1]
# using loc accessor
data.loc[1,["Happiness.Rank"]]
# Selecting only some columns
data[["Happiness.Score","Family"]].head(10)
# SLICING DATA FRAME
# Slicing and indexing series
data.loc[0:10,"Happiness.Score":"Family"]   # 10 and "Defense" are inclusive
# Reverse slicing 
data.loc[10:1:-1,"Happiness.Score":"Family"] 
# From something to end
data.loc[0:10,"Family":] # en sonuncusuna kadar git
# FILTERING DATA FRAMES
# Creating boolean series
boolean = data.Freedom > 0.5
data[boolean]
# Combining filters
first_filter = data.Freedom > 0.4
second_filter = data.Generosity > 0.3 #nokta içeren alanlar için farklı bir gösterim ile seçim yaparız aksi taktirde hata alırız data[hapiness.rank]gibi
data[first_filter & second_filter]
# Filtering column based others
data.Country[data.Freedom<0.2]
# TRANSFORMING DATA
# Plain python functions
def ekstra(n):
    return n*2
data["Economy..GDP.per.Capita."].apply(ekstra)

# Or we can use lambda function
data["Economy..GDP.per.Capita."].apply(lambda n : n*2)
# Defining column using other columns
data["total_freedom_generosity"] = data.Freedom + data.Generosity
data.head()
# HIERARCHICAL INDEXING
# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/2017.csv')
data.head()
# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Country","Happiness.Rank"]) 
data1.head(100)
# data1.loc["Fire","Flying"] # howw to use indexes