# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as scs
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/creditcard.csv')
data.info()
#dir(data)
#data.round()
data.head(10)
data.tail(10)
dictionary = {'V1':sum(data['V1']),'V2':sum(data['V2']),'V3':sum(data['V3']),'V4':sum(data['V4']),\
'V5':sum(data['V5']),'V6':sum(data['V6']),'V7':sum(data['V7']),'V8':sum(data['V8']),'V9':sum(data['V9'])}
print(dictionary.keys())
print(dictionary.values())                                                           
dictionary['V1'] = 0
print(dictionary)
dictionary['V10']= 100
print(dictionary)
del(dictionary['V10'])
print(dictionary)
print('V3' in dictionary)
dictionary.clear()
print(dictionary)
series = data['V1']
print(series)
print(type(series))
df = data[['V1','V20']]
x = data['V1']>1
df1 = data[x][['V1']]
print(df1)
data[np.logical_and(data['V1']>1.2, data['V2']<-3.2)]


sum(data[np.logical_and(data['V1']>1.2, data['V2']<-3.2)]['V1'])
data[(data['V1']<3)|(data['V2']>2)]
i = 0
while i != data['Time'][13]:
    print('your time is : ', i)
    i += 1
print('system not found')
for i in data['V1'][0:200000]:
    if i>2.4:
        print('V1 is :', i)
print('')
for index, value in enumerate(data['V1'][0:10]):
    print(index, ':', value)
print('')
dictionary = {'V1':sum(data['V1']),'V2':sum(data['V2']),'V3':sum(data['V3']),'V4':sum(data['V4']),\
'V5':sum(data['V5']),'V6':sum(data['V6']),'V7':sum(data['V7']),'V8':sum(data['V8']),'V9':sum(data['V9'])}
for key, values in dictionary.items():
    print(key, ':', values)
print('')
for index, value in data[['V1']][4:6].iterrows():
    print(index, ':', value)

def tuble_ex():
        liste = list(dictionary)
        return liste
a,b,c,d,e,f,g,h,j = tuble_ex()
print(a,b,c,d,e,f,g,h,j)
liste = [1,2,3]
def f():
        liste = list(dictionary)
        return liste
print(liste)
print(f())
import builtins
def square():
    def add():
        z = data['V1'][1]+data['V2'][1]
        print(data['V1'][1])
        print(data['V2'][1])
        return z
    return add()**2
print(square())
def f(c=5):
    y = data['V1'][1]+c
    return y
print(f())
print(f(4))
def f(*args):
    for i in args:
 
        print(i)
f(1)
print('')
f(1,2,3)
def f(**kwargs):
    for key, value in kwargs.items():
        print(key, ':', value)

f(country = 'Türkiye', city = 'İstanbul', population = '20 million')
square = lambda x: data['V1'][1]**2
print(square(x))

k=1
l=2
m=3
ext = lambda a:(k+l+m)**2
print(ext(a))
liste1 = list(data['V1'][0:10])
y = map(lambda a:a+5,liste1)
print(list(y))
liste1 = list(data['V1'][0:5])
liste2 = list(data['V2'][0:5])
z = zip(liste1,liste2)
z_list = list(z)
print(z_list)

un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
liste = list(data['V1'][0:10])
liste2 = [i+1 for i in liste]
print(liste2)
treshold = sum(data['V1'])/len(data['V1'])
data["V1+"] = ['high' if treshold>i else 'low' for i in data['V1']]
data.loc[:15,["V1+","V1"]]
print(data['V1'].value_counts(dropna=False))
data.describe()
data.info()
data.boxplot(column='V1', by = 'V1+')
plt.show()
data_new = data.head(10)
data_new
melted = pd.melt(frame=data_new,id_vars = 'Amount', value_vars = ['V1','V2','V3'])
melted
melted.pivot(index='Amount', columns='variable', values='value')
data_1=data.head()
data_2=data.tail()
conc_data_row=pd.concat([data_1,data_2],axis=0,ignore_index=True)
conc_data_row
data1=data['V1'].head()
data2=data['V2'].head()
data3=data['Amount'].head()
conc_data_col = pd.concat([data3,data1,data2],axis=1)
conc_data_col
melted1 = pd.melt(frame=conc_data_col,id_vars='Amount',value_vars=['V1','V2'])
melted1
melted1.pivot(index='Amount',columns = 'variable', values= 'value')
data.dtypes
data['V1']=data['V1'].astype('int')
data['V1+'] = data['V1+'].astype('category')
data.head()
data['V1+'].head()
data.info()
data['V1'].value_counts(dropna=False)
data_3 = data
data_3['V1'].dropna(inplace=True) 
assert 'naber' == 'naber'
assert data['V1'].notnull().all()
data['V1'].fillna('empty',inplace=True)
assert data['V1'].notnull().all()
nationality = ['Turkish','German','English']
gdp_per_capita = ['8000','21000','35000']
liste_col= ['nationality','gdp_per_capita']
liste_value = [nationality,gdp_per_capita]
zipped = list(zip(liste_col,liste_value))
dict_zip=dict(zipped)
df=pd.DataFrame(dict_zip)
df
df['gdp'] = ['1 billion','4 billion','17 billion']
df
df['improvement level categories'] = 2,1,1
df
data1=data.loc[:,['V1','V2','V3']]
data1.plot()
plt.show()
data1.plot(subplots=True)
plt.show()
data1.plot(kind='scatter',x='V1',y='V2')
plt.show()
data1.plot(kind='hist',y='V1',bins=50,range=(0,3))
plt.show()
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "V1",bins = 50,range= (0,5),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "V1",bins = 50,range= (0,5),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
time_list = ["1994-01-01","1987-01-12"]
print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore")
data12=data.head()
datetime_list=["1994-01-01","1994-01-02","1994-01-03","1994-01-04","1994-07-05"]
datetime_object1 = pd.to_datetime(datetime_list)
data12["date"]=datetime_object1
data12=data12.set_index("date")
data12
print(data12.loc["1994-01-01"])
print(data12.loc["1994-01-01":"1994-07-05"])
data12.resample("A").mean()
data12.resample("M").mean()
data12.resample("M").first().interpolate("linear")
data12.resample("M").mean().interpolate("linear")
data["V1"][1]
data.V1[1]
data.loc[1,["V1"]]
data[["V1","V2"]].head()
print(type(data["V1"]))
print(type(data[["V1"]]))
data.loc[1:10,"V1":"V5"]
data.loc[10:1:-1,"V1":"V5"]
data.loc[1:5,"V1":]
boolean = data.V1<=-40
data[boolean]
first_filter = data.V1<=-30
second_filter = data.V1>=-45
data_filter = data[first_filter & second_filter]
data_filter.set_index("Time")
data.V1[data.V2<-40]
def div(n):
    return n/2
data.Time.head().apply(div)
data.V1.head().apply(lambda n:n/2)
data["V1+V2"]=data.V1+data.V2
data.head()
print(data.index.name)
data.index.name="#"
data.head()
data33 = data.head().copy()
set_list=[1,2,3,4,5]
data33["no"] = set_list
data33.set_index("no")
data33.index=range(10,15,1)
data33
data21=data.set_index(["V1","V2"])
data21.head(25)
dic = {"name":["Ali","Veli","Kırkdokuz","Elli"],"surname":["Gel","Tut","Elli","Ellibir"],"gender":["M","M","F","F"],"class":[1,2,2,1]}
df=pd.DataFrame(dic)
df
df.pivot(index="gender",columns="class",values="name")
df1=df.set_index(["gender","class"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2=df1.swaplevel(0,1)
df2
pd.melt(df,id_vars="name",value_vars=["gender","class"])
df
df.groupby("gender").mean()


