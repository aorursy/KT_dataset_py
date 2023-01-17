# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def tuple_ex():

    d = (1,2,3,4)

    return d

a,b,c,d = tuple_ex()

print(a,b,c,d)
x = 2

def deger():

    x=3

    return x

print(x)

print(deger())
x=5

def sharp():

    y=2*x

    return y

print(sharp())
import builtins

dir(builtins)
def square():

    def add():

        x = 4

        y = 8

        z = x * y

        return z

    return add()**2 

print(square())
def sum(a, b=8,c=9):

    z = a + b + c

    return z

print(sum(5))
def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
square = lambda x : 5 ** x

print(square(5))

total = lambda x,y,z : x + y + z

print(total(5,4,3))
number_list = [5,6,7]

y = map(lambda x : x ** 2, number_list)

print(list(y))
# iteration example

name = "blackmamba"

it = iter(name)

print(next(it))    # print next iteration

print(*it)         # print remaining iteration
list1 = [2,4,6,8]

list2 = [3,5,7,9]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuple

print(un_list1)

print(un_list2)

print(type(un_list2))
num1 = [1,2,3]

num2 = [i + 1 for i in num1]

print(num2)
num3= [20,25,30]

num4 = [i ** 2 if i == 10 else i-5 if i<24 else i+5 for i in num3]

print(num4)
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.head(5) 
data.tail(5)
data.columns
data.shape
data.info()
print(data['Country/Region'].value_counts(dropna =False))
1,2,3,4,200
def hesaplama():

    t = (0,1,2,3,4,15)

    return t

a,b,c,d,e,f = hesaplama()

print(a+b+c+d+e+f)
# data.boxplot(column='Recovered', by ='Province/State')
data_new = data.tail(5)

data_new
melted = pd.melt(frame=data_new, id_vars="Country/Region", value_vars=["Deaths","Confirmed"])

melted
#melted.pivot(index = 'Country/Region', columns = 'variable',values='value')

#hocam hata alıyorum bakarsanız memnun olurum
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis=0, ignore_index=True)

conc_data_row
data1 = data["Country/Region"].head()

data2 = data["Province/State"].head()

conc_data_col = pd.concat([data1,data2],axis=1)

conc_data_col
data.dtypes
data['SNo'] = data['SNo'].astype('category')

#data['Speed'] = data['Speed'].astype('float')

data.dtypes
data.head(10)
data["Province/State"].value_counts(dropna =False)
assert  data['Country/Region'].notnull().all()
data1=data

data1["Province/State"].dropna(inplace = True)

data.head()
data.info()
data.head(100)
country = ["Turkey","ABD"]

population = ["80M","3Mi"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
df["capital"] = ["Ankara","Washington DC"]

df
df["income"]=0

df
data1=data.loc[:,["Confirmed","Deaths"]]

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind = "scatter",x="Deaths",y = "Confirmed")

plt.show()
data1.plot(kind = "hist",y = "Confirmed",bins = 50,range= (0,500))
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Deaths",bins = 50,range= (0,250),ax = axes[0])

data1.plot(kind = "hist",y = "Confirmed",bins = 50,range= (0,250),ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
time_list = ["01/22/2020","01/29/2020 	"]

print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))

print(datetime_object)
import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.tail()

date_list = ["01/22/2020","01/23/2020","01/24/2020","01/25/2020","01/26/2020"]

datetime_object = pd.to_datetime(date_list)

data2["ObservationDate"] = datetime_object

# lets make date as index

data2= data2.set_index("ObservationDate")

data2 
print(data2.loc["2020-01-22"])

print(data2.loc["2020-01-22":"2020-01-26"])
data2.resample("A").mean()
data2.resample("M").mean()
#data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

#data= data.set_index("#")

#data.head()

##KeyError: "None of ['#'] are in the columns"
data2["Deaths"][1]

data2.Deaths[1]
data.loc[1,["Deaths"]]
data[["Recovered","Deaths"]]
data.loc[1:20,"Deaths":]
print(type(data["Recovered"]))     # series

print(type(data[["Recovered"]]))
# Reverse slicing 

data.loc[40:1:-1,"Deaths":"Recovered"]
data.loc[1:10,"Confirmed":] 
boolean = data.Confirmed > 400

data[boolean]
first_filter = data.Confirmed > 400

second_filter = data.Deaths > 35

data[first_filter & second_filter]
data.Confirmed[data.Deaths<15]
def div(n):

    return n/2

data.Deaths.apply(div)
data.Deaths.apply(lambda n : n/2)
data["Remaining"] = data.Confirmed - data.Deaths

data.tail()
print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df



df.pivot(index="treatment",columns="gender",values="response")
df1 = df.set_index("treatment","gender")

df1
df1.unstack(level=0)
df1.unstack(level=1)
df
df.groupby("treatment").mean()
df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min() 
df.info




num1 = [6,4,10,12]



num2 = [i**2 if i % 3 == 2 else i-1  for i in num1]



print(num2)


