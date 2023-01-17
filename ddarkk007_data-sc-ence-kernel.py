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
df = pd.read_csv("/kaggle/input/btcusd-dataset/btc.csv") #csv datamızı import ediyoruz.
df.info()
df.columns
df.describe()
df_reserved = df[::-1] #liste tersten yazdırıyoruz.

df_reserved.head(10) #sonrasında ilk 10 indexi inceliyoruz
df.head(10)
df.corr()
#correlation map

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)

plt.show()
# lineplot



df.High.plot(kind='line', color='r', label='High Cost', linewidth=1, alpha=1,grid=True)

df.Low.plot(kind='line', color='b', label='Low Cost', linewidth=1, alpha=1,grid=True)

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('High/Low Cost')

plt.show()
#scatter plot

"""

High ile bid datalarınızı almamızda ki sebep; fiyatın tepe noktasında, fiyatların genel

seviyesi de yükseleceğinden dolayı insanlar daha yüksek bir fiyattan satın almaları gerecektir.

"""



df.plot(kind='scatter',color='orange',x='High', y='Ask',alpha=0.5)

plt.xlabel('High')

plt.ylabel('Ask')

plt.title('High/Ask')

plt.show()
# Histogram



df.Ask.plot(kind = 'hist',bins = 15,figsize = (12,12),color='darkblue')

plt.show()
dictionary = {'NAME': ["enes","serhat","lokman","omer","ibrahim"],

              'AGE': [25,20,15,14],

              'JOBS': ["investor","doctor","trainer","teacher","nurse"]}



# print(dictionary)

print(dictionary.keys())

print(dictionary.values())
dictionary['NAME'] = ["enes","kemal","sahin"]

print(dictionary)



# del dictionary['JOBS']

# print(dictionary)



print("sahin" in dictionary['NAME'])



dictionary.clear()

print(dictionary)
df1 = pd.read_csv('/kaggle/input/btcusd-dataset/btc.csv')
series = df1['High']

print(type(series))

data_frame = df1[['High']]

print(type(data_frame))
#Comparison operator:  ==, <, >, <=



print(3 > 2)

print(3 != 9)

print(4 == 16)



print(True or False)

print(True and True)

x = df1['High']>3000.0

df1[x]
df1[(df1['High'] > 3000.0) & (df1['Ask'] > 1000.0)]
s = 0



while s != 10:

    print('s is: ',s)

    s += 1

print('s degeri 10 degerine ulasti')
yeniliste = [1,2,3,4,5,6,7,8]

for x in yeniliste:

    print('x is: ',x)

print('')



for index, value in enumerate(yeniliste):

    print(index," : ",value)

print('')   



dictionary = {'enes':'investor','kemal':'doctor'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



for index,value in df[['Mid']][0:1].iterrows():

    print(index," : ",value)
#user defined function



def tuble_ex():

    t = (1,2,3)

    return t



a,b,c = tuble_ex()

print(a,b,c)
x = 2 #global variable



def f():

    x = 3

    return x



print(x)

print(f())
x = 5



def f():

    y = 2*x

    return y



print(x)

print(f())
import builtins



dir(builtins)
#nested function



def square():

    """return square of value"""

    def add():

        """add two local variable"""

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2



print(square())
#default and flexible arguments



def f(a,b=1,c=2):

    y = a + b + c

    return y



print(f(5))

print(f(5,4,3))



def s(*args):

    for i in args:

        print(i)

s(1)

print("")



s(1,2,3,4)



def x(**kwargs):

    for key, value in kwargs.items():

        print(key, ": ", value)



x = {"country": "turkey","capitalcity": "ankara","flag": "TR"}

print(x)
#lambda function



square = lambda a: a**2



print(square(5))



tpl = lambda x,y,z: x+y+z

print(tpl(5,5,5))
#anonymous function



yeniliste = [1,2,3]

y = map(lambda x: x**2, yeniliste)

print(list(y))
#iterators



name = "enes"

it = iter(name)

print(next(it))

print(*it)
#zip method



list1 = [1,2,3,4]

list2 = [5,6,7,8]



z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip)

print(un_list1)

print(un_list2)



print(type(un_zip))

print(type(un_list2))
#list comprehension



num1 = [1,2,3]

num2 = [i + 1 for i in num1]

print(num2)
#conditionals on iterrable



num3 = [5,10,15]

num4 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num3]

print(num4)
#pandas comprehension

import pandas as pd



df2 = pd.read_csv('/kaggle/input/btcusd-dataset/btc.csv')



treshold = sum(df2.High)/len(df2.High)

df2['HL_PCT'] = ["High" if i > treshold else 'Low' for i in df2.High]

df2.loc[:10,["HL_PCT","High"]]
num1 = [1,2,3,4]



num2 = [i**2 if i % 2 == 0 else i-i  for i in num1]



print(num2)
#cleaning data



import pandas as pd



df3 = pd.read_csv('/kaggle/input/btcusd-dataset/btc.csv')

df3.head()
df3.tail()
df3.columns
df3.shape
df3.info()
#explore data analysis

#value_counts()

import pandas as pd



df4 = pd.read_csv("/kaggle/input/btcusd-dataset/btc.csv")

print(df4['Low'].value_counts(dropna=False))
df4.describe()
#visual exploratory data analysis

import matplotlib.pyplot as plt



df4.boxplot(column='Low')

plt.show()
#tidy data



df_new = df4.head()

df_new
# Belirli dataları çıkartıp ekleyebilmemezi sağlıyor.



melted = pd.melt(frame=df_new,id_vars='Date',value_vars=['High','Low','Mid'])

melted
#pivoting data; melted datayı eski hale getirme



melted.pivot(index='Date',columns='variable',values='value')
#concatenating data



df0 = df4.head()

df8 = df4.tail()



conc_data_row = pd.concat([df0,df8],axis=0,ignore_index=True) #dataları birleştirme

conc_data_row
df7 = df4['High'].head()

df9 = df4['Ask'].head()



conc_df_row = pd.concat([df7,df9],axis=1)

conc_df_row
#data types



df4.dtypes
df4['Date'] = df4['Date'].astype('category')
df4.dtypes
#missing data



df4.info()
df4["Ask"].value_counts(dropna=False)
data2 = df4

data2["Ask"].dropna(inplace=True)
assert 1==1 #çalıştığını kontrol ediyoruz
assert data2['Ask'].notnull().all() #null olan kısımların hepsi silindi mi kontrol ediyoruz
data2['Ask'].fillna('empty',inplace=True)
assert data2['Ask'].notnull().all() #null olan kısımların hepsi silindi mi kontrol ediyoruz
#Pandas Foundation



import pandas as pd



name = ["enes","ali","omer"]

age = [19,23,35]

list_label = ["name","age"]

list_col = [name,age]

zipped = zip(list_label,list_col)

data_dict = dict(zipped)

df88 = pd.DataFrame(data_dict)

df88
df88['City'] = ["istanbul","rize","hakkari"]

# df44 = df88.drop(columns=['Cit'])

df88
df88['Cost'] = 0

df88
#visual exploratory



df4 = df4.loc[:,['High','Low','Ask']]

df4.plot()
df4.plot(subplots = True)

plt.show()
df4.plot(kind='scatter',x='High',y='Ask',label='High/Ask',color='r')

plt.show()
df4.plot(kind='hist',y='High',bins=50,range=(0,250),color='r',normed=True,grid=True)

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

df4.plot(kind='hist',y='High',bins=50,range=(0,250),color='r',normed=True,grid=True,ax=axes[0])

df4.plot(kind='hist',y='High',bins=50,range=(0,250),color='r',normed=True,grid=True,ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt.show()

#indexing pandas time series



df88 = pd.read_csv('/kaggle/input/btcusd-dataset/btc.csv')

df88.head()
import warnings



warnings.filterwarnings("ignore")



data5 = df88.head()

data5.drop(columns='Date')

date_list = ["2020-05-27","2020-05-26","2020-03-25","2020-02-24","2019-01-23"]

datetime_object = pd.to_datetime(date_list)

data5['Date'] = datetime_object



data5 = data5.set_index("Date")

data5
print(data5.loc['2020-05-27'])

print(data5.loc['2020-05-27':'2020-03-23'])
#resampling pandas time series



data5.resample("M").mean()
data5.resample("M").first().interpolate("linear") #NaN datalarımızı linear olarak doldururu 1,2,3,?,5,6,7 && ? == 4
data5.resample("M").mean().interpolate("linear")
#manipulating data frames with pandas



import pandas as pd



data66 = pd.read_csv("/kaggle/input/btcusd-dataset/btc.csv")

# data66.columns = ['D','High','Low','Mid','Last','Bid','Ask','Volume']

# data66 = data66.set_index('Date')

data66.head()
data66.Mid[1]
data66.iloc[2,2]
data66[["High","Low"]]
#slicing dataframe



data66.loc[1:10,"High":"Mid"]
data66.loc[1:10,"High":]
#filtering data frames



boolean = data66.High > 6500.0



data66[boolean]
first_filter = data66.High > 6500.0

second_filter = data66.Low < 6000.0



data66[first_filter & second_filter]
data66[data66.High > 6000]
data66.Low[data66.High > 6000]
#transforming data



def div(n):

    return n/2

    

data66.High.apply(div)
data66["HIGH-LOW"] = (data66['High']- data66['Low'])

data66.head()
#index objects labeled data



print(data66.index.name)



# data66.index_name = "index_name"

data66.head()
import pandas as pd



data67 = pd.read_csv("/kaggle/input/btcusd-dataset/btc.csv")



data67.index = range(100,2228,1)

data67.head()
data67.tail()
#hierarchical indexing

import pandas as pd



data27 = pd.read_csv("/kaggle/input/btcusd-dataset/btc.csv")

data27.head()
data28 = data27.drop(columns='Date')
data28 = data28.set_index(['High','Low'])
data28.head(100)
#pivoting dataframes



dic4 = {"GENDER": ["MALE" , "FEMALE"],

        "NAME": ["SERKAN","BERNA"],

        "AGE": [19,22]}



data101 = pd.DataFrame(dic4)



data101
#pivoting



data101.pivot(index='NAME',columns='GENDER',values='AGE')
#stacking and unstacking dataframe



data100 = data101.set_index(["NAME","AGE"])

data100
data100.unstack(level=0)
df4 = data100.swaplevel(0,1)

df4
#MELTING DATA FRAMES



data100