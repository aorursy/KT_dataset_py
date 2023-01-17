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
data = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')
data.head()

data.info() #variables
#correlation map  
#featurelar arasındaki ilişkiyi anlamamızı sağlar
#iki feature arasında corolation 1 ise doğru orantılıdır
data.corr()
f, ax = plt.subplots(figsize=(11,11)) #figürümüzn size ını belirledik 18-18
sns.heatmap(data.corr(), annot=True, linewidth=.5, fmt='1f',ax=ax) #fmt 0dan sonra yazdıracağı değer
plt.show()
data.head(100) #default olarak ilk 5 değeri gösterir bizimki 10
data.columns  #sahip olduğu columnların ismini gösterir gösterir
#lineplot
data.High.plot(kind='line',color='blue',label='High',linewidth=1,alpha=0.5,grid=True,linestyle=':')
data.Close.plot(kind='line',color='red',label='Close',linewidth=1,alpha=0.5,grid=True,linestyle='-')
plt.legend(loc='upper right') #legend=puts label into plot
plt.xlabel('x axis')          #name of label
plt.ylabel('y axis')
plt.title('Line Plot')        #title of plot
plt.show()
#Scatter plot
# x= Open y = Close
data.plot(kind='scatter',x='Open',y='Volume_(BTC)',alpha=0.5,color='red')
plt.xlabel('Open')          #name of label
plt.ylabel('Volume_(BTC)')
plt.title('Open-Volume_(BTC) Scatter Plot')
plt.show()
#Histogram
#bins=number od bar in figure
data.Open.plot(kind='hist',bins=50,figsize=(11,11))
plt.show()
# clf() = cleans it up again you can start a fresh 
data.Open.plot(kind='hist',bins=50)
plt.clf()  #çizilen plotu clean eder
#create a dictionary and look its keys and values
dictionary = {'palpable money':'dolar','cripto money':'bitcoin'}
print(dictionary.keys())
print(dictionary.values())
dictionary['palpable money']= "euro"  #update existing entry
print(dictionary)
dictionary['inconsumable']="turkish lira" #add a new entry
print(dictionary)
del dictionary['inconsumable']  # remove entry with key 'inconsumable'
print(dictionary)
print('cripto' in dictionary)  #check include or not
dictionary.clear()     #remove all entries in dict
print(dictionary)
#If you want to delete all dictionary
#del dictionary
print (dictionary)
#series and dataframe

series = data['Open']
print(type(series))
data_frame = data[['Open']]
print(type(data_frame))
# logic = mantık
# control flow =kontrol akış
# filtering = filtre
# Comparison operator: ==, <, >, <= 
# Boolean operators: and, or ,not 
# Filtering pandas
# Comparison operator
print(12 > 8)
print(38!=4)

# Boolean operators
print(True and False) 
print(True or False)
# 1 - Filtering Pandas data frame
x=data['Open']>19000
data[x]
# 2 - Filtering pandas with logical_and
data[np.logical_and(data['Open']>19000, data['Close']>19500 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Open']>19000) & (data['Close']>19500)]
# Stay in loop if condition( i is not equal 5) is true

i = 0
while i !=5 :
    print('i is:',i)
    i+=1
print(i,'is equal to 5')    

# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]     #sırayla listenin içindekileri print ettirir
for i in lis:
    print('i is:',i)
print('')
# Enumerate index and value of list / indexini girerek değerlere erişim sağlayabiliriz
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):
    print(index," : ",value)
print('')    
# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
#Dictionary den keys ve value değerlerini çekebilmek için

dictionary = {'sapain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')    

# For pandas we can achieve index and value

for index,value in data[['Attack']][0:1].iterrows():
    print(index," : ",value)
#PYTHON DATA SCIENCE TOOLBOX
def tuble_ex():        #kind of list
    """return defined t tuble"""
    t=(1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
x = 5
def f():
    x = 12
    return x
print(x)       #x=5 global scope
print(f())     #x=12 local scope
x = 5
def f():
    y=x**2   #there is no local scope
    return y
print(f())   #it uses global scope
import builtins   #python tarafından kullanılan kısaltmalar
dir(builtins)
#NESTED FUNCTION - function inside function

def square():
    def add():
        x=6
        y=9
        z=x+y
        return z
    return add()**2
print(square())









