# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/world-happiness/2019.csv')
data.info()
data.head(10)
data.tail(10)
data.corr()
#Correlation Map

f,ax = plt.subplots(figsize=(14,14))

sns.heatmap(data.corr(), annot =True , linewidth=.5, fmt='.2f',ax=ax)

plt.show()
data.columns
# Line Plot

data.rename(columns={"GDP per capita":"GDPPerCapita","Healthy life expectancy":"HealthyLifeExpectancy","Social support":"SocialSupport"},inplace=True)



data.GDPPerCapita.plot(kind = 'line', color = 'blue',label = 'GDPPerCapita',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.SocialSupport.plot(color = 'purple',label = 'SocialSupport',linewidth=1, alpha = 0.8,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel=('x axis')              # label = name of label

plt.ylabel=('y axis')

plt.title=('Line Plot')            # title = title of plot

plt.show()

# Scatter Plot

data.plot(kind='scatter',x='GDPPerCapita', y='HealthyLifeExpectancy',alpha=0.5,color='blue',title='Scatter Plot')

plt.xlabel=('GDP per capita')

plt.ylabel=('Healthy life expectancy')

#Histogram Plot

data.GDPPerCapita.plot(kind='hist', bins=45,figsize=(15,15),title='GDP per capita Histogram Plot')

plt.show()
#Comparison Operator

print (3<2)

print (3!=5)

print (3==3)

print (5>4)

print (3<=1)

print (9>=6)



#Boolean Operator

print (True and False)

print (True and True)

print (True or False)

print (True or True)

print (False or False)
# Filtering Pandas Data Frame

a = data['Score']>7.000     # There are only 16 Country or region who have higher Score value than 7.000

data[a]
data['Score']>7.000
# Filtering pandas with logical_and

# There are only 3 Country or region who have higher Score value than 7.000 and higher SocialSupport value than 1.580

data[np.logical_and(data['Score']>7.000, data['SocialSupport']>1.580)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Score']>7.000) & (data['SocialSupport']>1.580)]
# Stay in loop if condition( i is not equal 7) is true

i = 0

while i !=7:

    print("i is : ",i)

    i+=1

print(i," is equal to 7")
# Stay in loop if condition( i is not equal 9) is true

lis = [11,24,38,49,52,66,78,81,93]

for i in lis:

    print("i is: ",i)

print("")

# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9

for index, value in enumerate(lis):

    print(index," : ",value)

print("")

# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'Turkey':'Ankara', 'Uganda':'Kampala','Somalia':'Mogadishu','Romania':'Bucharest'}

for key,value in dictionary.items():

    print(key," : ",value)

print("")

# For pandas we can achieve index and value

for index,values in data [['Country or region']][0:2].iterrows():

    print(index," : ",values)
# example of what we learn above

def tuple_example1():

    x=(1,2,3,4)

    return x

a,b,c,d=tuple_example1() #eleman sayısı kadar değere atama yapmak zorundayız.

print(a,b,c,d)

    
# example of what we learn above

def tuple_example2():

    a=(54,98,82,46,73,37)

    return a

a,b,c,d,e,f = tuple_example2()

print(a,b,c,d,e,f)
# example of what we learn above

def tuple_example3():

    notlar=(65,98,7,45,25,14)

    return notlar

a,b,c,d,e,f = tuple_example3()

print(a,b) # Değerlerin hepsini yazdırmak zorunda değliz. Sadece istediğimiz değereri yazdırma şansımız var.
# example of what we learn above

def tuple_example4():

    sayilar =(11,12,13)

    return sayilar

a,_,_= tuple_example4()  #eğer 3 sayıya da atama yapmak istemiyor isek diğerlerinin yerine alt cizgi koyabiliriz. Başka yerlerde görür iseniz şaşırmayın..

print(a)
# guess print what

degisken = 1

def fonksiyon():

    degisken = 5

    return degisken

print(degisken) # degisken = 1 global scope

print(fonksiyon()) # degisken = 5 local scope

# What if there is no local scope

degisken = 6

def fonksiyon():

    sonuc = degisken * 5  # there is no local scope degisken

    return sonuc

print(fonksiyon())  # it uses global scope degisken

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# Example2

x = 5

def islem():

    sonuclar = x + 9

    return sonuclar

print(islem())
# How can we learn what is built in scope

import builtins

dir(builtins)
#nested function

def fonksiyon():

    """ return square of value """

    def add():

        """ add two local variable """

        a = 4

        b = 5

        sonuc= a + b

        return sonuc

    

    return add()**2



print(fonksiyon())

# default arguments

def elemanlar(x,y=5,z=3):

    sonuc= x + y + z

    return sonuc

print(elemanlar(9))

# what if we want to change default arguments

print(elemanlar(4,5,6))

### Flexible Arguments *args

def fonksiyon(*args):

    for i in args:

        print(i)

fonksiyon(2)



print("----------------------")



# Flexible arguments *args example 2

def fonksiyon2(*args):

    for i in args:

        print (i)

fonksiyon2(1,2,3,4,5,6)

# Flexible arguments **kwargs that is dictionary 

def fonksiyon(**kwargs):

    """ print key and value of dictionary"""

    for key,value in kwargs.items():   # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key," : ",value)

fonksiyon(country = 'Turkey',capital = 'Ankara')

fonksiyon(country = 'England', capital = 'France')

        

        
# Lambda function

hesapla = lambda x: x**2

print(hesapla(5))



print("---------------")



hesapla2 = lambda a,b,c : a + b + c

print(hesapla2(4,5,6))
numara_listesi = [15,20,25,30,35]

fonksiyon = map(lambda x:x**2,numara_listesi)

print(list(fonksiyon))
# iteration example

isim = "HELLO"

hecele = iter(isim)

print(next(hecele))

print("-----")

print(*hecele)

print("-----")



# iteration example2

isim2 = "THATS ALL :D"

hecele2 = iter(isim2)

print(*hecele2)
# zip example

liste1 = [1,2,3,4,5,6]

liste2 = [10,11,12,13,14,15]

zipli_liste = zip(liste1,liste2)

print(zipli_liste)

zipi_cevir = list(zipli_liste)

print(zipi_cevir)
# unzip example

zipi_kaldır = zip(*zipi_cevir)

zipli_liste1,zipli_liste2 = list(zipi_kaldır)  # unzip returns tuble

print(zipli_liste1)

print(zipli_liste2)

print(type(zipli_liste1))

print(type(list(zipli_liste1)))

print(type(list(zipli_liste2)))
# Example of list comprehension

liste1 = [1,2,3,4]

liste2 = [i+1 for i in liste1]

print (liste2)

print("-------------------------------")

döngü_listesi = [2,3,4,5,6,7,8,9]

islem_listesi = [i+5/2 for i in döngü_listesi]

print (islem_listesi)
# Conditionals on iterable

liste1 = [5,10,15,6,8,4,9]

liste2 = [i**2 if i==10 else i-5 if i<7 else i+7 for i in liste1]

print(liste2)
data = pd.read_csv('../input/world-happiness/2019.csv')

data.head(10)  # head shows first 10 rows
data.tail(3)
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
# For example lets look frequency of happy types

print(data["Country or region"].value_counts(dropna=False))  #example 1

print(data["GDP per capita"].value_counts(dropna = False))  # example 2

print(data["Healthy life expectancy"].value_counts(dropna = False)) # example 3
# For example max HP is 255 or min defense is 5

data.describe()
data.Score.describe()
data["GDP per capita"].describe()
data.info()
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='Healthy life expectancy',by='Social support')

plt.show()
# Firstly I create new data from pokemons data to explain melt nore easily.

new_data = data.head(6)

new_data
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted= pd.melt(frame=new_data,id_vars='Country or region',value_vars=['Score','Healthy life expectancy'])

melted

# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index='Country or region',columns='variable',values='value')
# Firstly lets create 2 data frame

data1 = data.head(3)

data2 = data.tail(3)

conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True) # # axis = 0 : adds dataframes in row

conc_data_row



# data_birlestir = pd.concat([data1,data2],axis=1,ignore_index=True)

# data_birlestir





data1=data['Score'].head()

data2=data['Healthy life expectancy'].head()

data_concat=pd.concat([data1,data2],axis=0)

data_concat

data.dtypes
# lets convert object(str) to categorical and int to float.

data['Country or region'] = data['Country or region'].astype('category')

data['GDP per capita'] = data['GDP per capita'].astype('object')
# As you can see Country or region is converted from object to categorical

# And GDP per capita ,s converted from int to object

data.dtypes
data.info()
# Lets chech Type 2

data['Country or region'].value_counts(dropna=False)
data['Generosity'].value_counts(dropna=False)
data['Perceptions of corruption'].value_counts(dropna=False)
data['Social support'].value_counts(dropna=False)
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# assert 25==100  ----> false

assert 25==25   # -----> True
assert data['Social support'].notnull().all()   # returns nothing because we drop nan values
assert data ['Perceptions of corruption'].notnull().all()
data['Perceptions of corruption'].fillna('empty',inplace=True)
assert data ['Perceptions of corruption'].notnull().all()  # returns nothing because we do not have nan values
data.head(10)
# assert data.columns[1]=='Country or region'

# assert data.columns[0]=='Overall rank'

# assert data.columns[2]=='Score'

# assert data.columns[3]=='GDP per capita'

# assert data.columns[4]=='Social support'

# assert data.columns[5]=='Healthy life expectancy'

# assert data.columns[6]=='Freedom to make life choices'

# data frames from dictionary

country = ["İspanya","Fransa","Türkiye","Somali","Afrika"]

population = ["11","12","10","15","11"]

list_label = ["country","population"]

list_column = [country,population]

zipped = list(zip(list_label,list_column))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"]=["madrid","paris","ankara","mogadişu","cape Town"]

df
# Broadcasting

df["income"]=0         #Broadcasting entire column

df

data.head()
data.info()
# Plotting all data 

data1 = data.loc[:,["Score","Social support","Freedom to make life choices"]]

data1.plot()

plt.show()
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot

data1.plot(kind="scatter",x = "Social support",y="Freedom to make life choices")

plt.show()
# hist plot

data1.plot(kind="hist",y="Social support",bins=15,range=(0,1),normed=True)

plt.show()
# histogram subplot with non cumulative and cumulative

fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist" , y="Social support" , bins=20 , range=(0,1) , normed=True , ax=axes [0])

data1.plot(kind="hist" , y="Social support" , bins=20 , range=(0,1) , normed=True , ax=axes[1], cumulative=True)

plt.savefig('graph.png')

plt

plt.show()
data.describe()
time_list=["2020-01-02","2020-01-03","2020-01-04"]

print(type(time_list[1]))       # As you can see date is string

# however we want it to be datetime object

date_time_object=pd.to_datetime(time_list)

print(type(date_time_object))
# In order to practice lets take head of happy data and add it a time list

data2=data.head()

date_list=["2020-01-01","2020-01-02","2020-01-03","2020-01-04","2020-01-05"]

date_object=pd.to_datetime(date_list)

data2["date"] = date_object

# lets make date as index

data2=data2.set_index("date")

data2
# Now we can select according to our date index

print(data2.loc["2020-01-05"])

print(data2.loc["2020-01-01":"2020-01-05"])

# We will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
data=pd.read_csv('../input/world-happiness/2019.csv')

data.head()
data.info()
data.index.names=['#']

data.head()
# indexing using square brackets

print(data["Country or region"][1])

print(data["Country or region"][23])
# using column attribute and row label

data.Score[1]
# using loc accessor

data.loc[1,["Score"]]
# Selecting only some columns

data[["Score","GDP per capita"]]