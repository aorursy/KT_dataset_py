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
data=pd.read_csv('../input/ccdata/CC GENERAL.csv')
data.info()
data.head()
data.columns
data.corr()
##corraletion map
f,ax =plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot=True,linewidths=0.5, fmt='.2f', ax=ax)
plt.show()
data.plot(kind="line",x='ONEOFF_PURCHASES', y='CREDIT_LIMIT',linewidth=1,color='g',grid=True,alpha=0.6)
plt.xlabel('oneoff_purchases')
plt.ylabel('limits')
plt.title('line plot')
plt.show()
data.PURCHASES.plot(kind='line', x='PURCHASES',color='r',linewidth=1.0, grid=True, alpha=1.0,linestyle = ':')
data.CASH_ADVANCE.plot(kind='line', x='CREDIT_LIMIT',color='b',linewidth=1, grid=True,alpha=0.7,linestyle = ':')
plt.xlabel('PURCHASES')
plt.ylabel('CREDIT_LIMIT')
plt.show()
data.CREDIT_LIMIT.plot(kind='hist', bins=50,figsize=(12,12))
plt.show()
data.plot(kind='scatter',x='BALANCE',y='PURCHASES_FREQUENCY', alpha=0.7,grid=True, color='r')
plt.xlabel('BALANCE')
plt.ylabel('PURCHASES')
plt.title('scatter plot')
plt.show()
plt.clf()
dictionary={'turkey':'istanbul','england':'london','germany':'berlin'}
print(dictionary.keys())
print(dictionary.values())
dictionary['turkey']="ankara"
print(dictionary)
dictionary['italy']="rome"
print(dictionary)
del dictionary['turkey']
print(dictionary)
print('italy' in dictionary)
print('russia'in dictionary)
dictionary.clear()
print(dictionary)
series=data['BALANCE']
print(type(series))
data_frame=data[['BALANCE']]
print(type(data_frame))
print(5<10)
print(3!=4)

print(True and False)
print(True or False)
x=data['BALANCE']<1000
data[x]
data[np.logical_and(data['BALANCE']>4000,data['PURCHASES']<200)]
data[(data['PURCHASES']<200) & (data['BALANCE']>4000)]
i=0
while i!=10:
    print('i is:',i)
    i+=2
print(i,'is eqaul to 10')
lis=[1,2,3,4,5]
for i in lis:
    print('i is:',i)
print('')

for index , value in enumerate(lis):
    print(index,":", value)
    print('')

dictionary={'turkey':'istanbul', 'germany':'berlin'}
for key,value in dictionary.items():
    print(key,":",value)
print('')
x=5
def f():
    y= x+2
    return y
print(x)
print(f())
import builtins
dir(builtins)
x=10
def add():
    def drive():
        x=12
        y=4
        z=x/y
        return z
    return drive()+x
print(add())
square= lambda x: x**2
tot= lambda x,y,z: x+y+z
print(square(7))
print(tot(3,5,7))
number_list=[1,2,3]
y=map(lambda x: x**2,number_list)
print(list(y))
city="istanbul"
it=iter(city)
print(next(it))
print(*it)
#zip example
list1=[1,12,3,4]
list2=[5,6,7,8]
z=zip(list1,list2)
print(z)
z_list=list(z)
print(z_list)
#unzip
un_zip=zip(*z_list)
un_list1,un_list2=list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))
num1=[2,4,6]
num2=[i**2 if i==4 else i+2 if i<4 else i-2 for i in num1]

print(num2)
threshold=sum(data.BALANCE)/len(data.BALANCE)
data["balance_amount"]=["high"if i>threshold else "low" for i in data.BALANCE]
data.loc[:20,["balance_amount","BALANCE"]]
print(data['CREDIT_LIMIT'].value_counts(dropna= False))
data.describe()
data.boxplot(column='BALANCE', by='BALANCE_FREQUENCY')
data_new=data.head()
data_new
melted=pd.melt(frame=data_new,id_vars="CUST_ID", value_vars=['CREDIT_LIMIT','PAYMENTS'])
melted
melted.pivot(index = 'CUST_ID', columns = 'variable',values='value')
data1=data.head()
data2=data.tail()
conc_data_row=pd.concat([data1,data2],axis=0, ignore_index=True)
conc_data_row
data["MINIMUM_PAYMENTS"].value_counts(dropna=False)
data1=data
data1["MINIMUM_PAYMENTS"].dropna(inplace=True)
assert data["MINIMUM_PAYMENTS"].notnull().all()
data["MINIMUM_PAYMENTS"].fillna('empty',inplace = True)
assert  data['MINIMUM_PAYMENTS'].notnull().all()
import warnings
warnings.filterwarnings("ignore")
data2=data.head()
date_list=["2020-01-10","2020-02-10","2020-03-10","2020-03-15","2020-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2 
print(data2.loc["2020-03-10":"2020-03-16"])
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data["BALANCE"][1]
#OR 
#data.BALANCE[1]
data[["BALANCE","BALANCE_FREQUENCY"]]
print(type(data["BALANCE"]))     # series
print(type(data[["BALANCE"]]))   # data frames
data.loc[1:10,"BALANCE":"BALANCE_FREQUENCY"]
data.loc[10:1:-1,"BALANCE":"BALANCE_FREQUENCY"]
data=pd.read_csv('../input/ccdata/CC GENERAL.csv')
data.loc[1:10,"CREDIT_LIMIT":]
def div(n):
    return n/2
data.BALANCE.apply(div)
print(data.index.name)
data.index.name="index_name"
data.head()
data3=data.copy()
data3.index=range(100,9050,1)
data3.head()
data1=data.set_index(["BALANCE","BALANCE_FREQUENCY"])
data1.head(10)