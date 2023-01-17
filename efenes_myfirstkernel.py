#My First Kernel
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/world-happiness/2015.csv')
data.columns = [c.replace(' ', '_') for c in data.columns]
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(19,19))

sns.heatmap(data.corr(), annot=True, linewidths=.6, fmt= '.2f',ax=ax)

plt.show()
data.head(10)
data.columns
data.Freedom.plot(kind = 'line', color = 'g',label = 'Freedom',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')

data.Family.plot(color = 'r',label = 'Family',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend() 

plt.xlabel('Country')              

plt.ylabel('Freedom and Family')

plt.title('Freedom and Family Overview')            

plt.show()
data.plot(kind='scatter', x='Freedom', y='Family',alpha = 0.4,color = 'green')

plt.xlabel('Freedom')              

plt.ylabel('Family')

plt.title('Freedom and Family Scatter Plot') 

plt.show()
data.Family.plot(kind = 'hist',bins = 40,figsize = (15,15))

data.Generosity.plot(kind = 'hist',bins = 40,figsize = (15,15))

plt.ylabel('Frequency')              

plt.title('Freedom and Generosity Histogram') 

plt.legend()

plt.show()
dict20={"Iceland":2,"Norway":4,"Sweden":8,"Australia":10}

print(["Sweden"])

dict20["Pol"]=15

print(dict20)

del dict20["Pol"]

print(dict20)

print(dict20.keys())

print(dict20.values())

print(dict20.items())

dict20.clear()

print(dict20)
data[(data["Happiness_Score"]>7.3) & (data["Family"]<1.32)]
data[((data["Standard_Error"]<0.03)&(data["Family"]<1.33)) | ((data["Freedom"]>0.6)&(data["Generosity"]<0.6))]
data["RandomCol"]=data["Standard_Error"]*2-data["Family"]+data["Dystopia_Residual"]

data.head()


dict22={"Iceland":2,"Norway":4,"Sweden":8,"Australia":10}

for key,value in enumerate(dict22):

    print(key," : ",value)

    

print("")

    

def tple():

    t = (2,4,8,10)

    return t

Iceland,Norway,Sweden,Australia = tple()

print([Iceland,Norway,Sweden,Australia])
def square():

    def add():

        z = Iceland

        return z

    return add()**2

print("Norway: ",square()) 
def f(a, b = 3, c = 1):

    y = a * b * c

    return y

f(1,2,3)
def f(**kwargs):

    for key, value in kwargs.items():              

        print(key, " ", value)

f(country = 'Norway', capital = 'oslo', population = 635000)
exp11 = lambda n,q3: n+q3

print([exp11(1,5)])
number = [1,5]

y = map(lambda x:x**2,number)

print(y)

list1 = ["n","i","s","d"]

list2 = ["p","k","v","y"]

q0 = zip(list1,list2)

print(q0)

threshold = sum(data.Family)/len(data.Freedom)

data["Family_level"] = ["high" if i > threshold else "low" for i in data.Family]

data.loc[:10,["Family_level","Family"]]
print(data['Region'].value_counts(dropna =False))
data.describe()
data.tail().boxplot(column='Generosity',by = 'Family',figsize=(8,8))

plt.show()
data_new = data.head()

data_new

melted = pd.melt(frame=data_new,id_vars = 'Country', value_vars= ['Happiness_Rank','Freedom'])

melted
melted.pivot(index = 'Country', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) 

conc_data_row
data1 = data['Country'].head()

data2= data['Region'].head()

conc_data_col = pd.concat([data1,data2],axis =1) 

conc_data_col
data['Family'] = data['Family'].astype('float')
data.dtypes
data["Region"].value_counts(dropna =False)
data1 = data.loc[:,["","Happiness_Score","Standard_Error"]]

data1.plot()

plt.show()
data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x="Happiness_Score",y = "Standard_Error",color='red')

plt.show()
data1.plot(kind = "hist",y = "Happiness_Score",bins = 10,range= (2,8),normed = True)

plt.show()
time_list = ["2022-03-08","2022-04-12"]

print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")

data2 = data.head()

date_list = ["2022-01-10","2022-02-10","2022-03-10","2022-03-15","2022-12-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data["Family"][5]
data.loc[5,["Family"]]
data[["Family","Freedom"]]
print(data["Family"])
print(data[["Family"]])
print(type(data["Family"]))

print(type(data[["Family"]]))
data.loc[1:10,"Family":"Freedom"]   
data.loc[10:1:-1,"Family":"Freedom"]
data.Freedom[data.Family<1.23]
def div(n):

    return n/2

data.Family.apply(div)
data.Family.apply(lambda n : n/2)
data["Power"] = data.Family + data.Freedom

data.head()
print(data.index.name)

data.index.name = "Number"

data.head()
data.index = data["Happiness_Rank"]

data.head()
data.unstack(level=0)
data.groupby("Family").mean()
data.groupby("Family").Freedom.max()
data.groupby("Family")[["Freedom","Generosity"]].min() 