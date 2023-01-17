# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015ddd/2015d.csv')
data.info()
data.head(20)
data.tail()
data.columns
data.corr()
f,ax = plt.subplots(figsize = (15,15))

sns.heatmap(data.corr(), annot = True, linewidth = 0.5, fmt='.2f', ax=ax)

plt.show()
data.Economy.plot(kind = 'line', color = 'g', label = 'Economy', linewidth = 1, alpha = 1, grid = True, linestyle= ':')

data.Health.plot(kind = 'line', color = 'b', label = 'Health', linewidth = 1, alpha = 1, grid = True, linestyle = '-.')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.legend()

plt.title('Line plot')

plt.show()
data.plot(kind = 'scatter', x = 'Happiness_Score', y = 'Economy', color = 'b', alpha = 0.5)

plt.xlabel('Happiness Score')

plt.ylabel('Economy')

plt.title('Happiness Score - Economy Scatter Plot ')

plt.show()
plt.bar(data.Country[145:157], data.Happiness_Score[145:157], width =  0.5, color= 'r', alpha = 0.7)

plt.show()
def tuple_ex():

    t = (5,6,'a')

    return t    #return defined tuple -t- .

x,y,z = tuple_ex()

print(x,y,z)

x = 5

def f():

    x = 3

    y = x * 2

    return y

print(x)

print(f())
def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(1,2,3))
def f(*args):

    for i in args:

        print(i)

f(1)

f(1,2,3,4,'as',67)



def f(**kwargs):

    for key , value in kwargs.items():

        print(key," ",value)

        

f(country = 'spain', capital = 'madrid' , population = 123421)
addition = lambda x,a,b : x+a+b

addition(1,9,3)

square = lambda x : x**2

square(7)
number_list = [1,2,3]

y = map(lambda x : x**2 , number_list)

print(list(y))
name = "abdullah"

k = iter(name)

print(next(k))

print(*k)
list1 = [1,2,3,4]

list2 = [5,6,'a',8]

z = zip(list1,list2)

print(z)

z_list = list(z)

type(z_list)

print(z_list)
un_zip = zip(*z_list)

un_list1 , un_list2 = list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list2))
num1 = [1,2,3]

num2 = [i+1 for i in num1]

print(num2)
num1 = [5,10,15]

num2 = [i-5 if i < 7 else i**2 if i == 10 else i+5 for i in num1]

print(num2)
#  <6low        >6 high
data["Happiness_level"] = ["high" if i > 6 else "low" for i in data.Happiness_Score]

data.loc[:10 , ["Country","Happiness_level", "Happiness_Score"]]
data.head()
data.tail() #last 5
data.columns

data.shape
data.info()
print(data['Region'].value_counts(dropna = False)) #count non-null variables too.
data.describe() #ignore null entries
data.boxplot(column = 'Economy', by = 'Happiness_level')

plt.show()
data_new = data.head()

data_new
melted = pd.melt(frame = data_new, id_vars = 'Country',value_vars =['Happiness_Rank','Happiness_Score'])

melted
melted.pivot(index = 'Country',columns = 'variable',values = 'value')
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis = 0,ignore_index = False) # axis = 0 add in row

conc_data_row

data0 = data['Country'].head()

data1 = data['Happiness_Score'].head()

data2 = data['Economy'].head()

conc_data_col = pd.concat([data0,data1,data2],axis = 1)

conc_data_col
data.dtypes
data['Country'] = data['Country'].astype('category')

data['Happiness_Rank'] = data['Happiness_Rank'].astype('float')
data.dtypes
data.info()
data["Region"].value_counts(dropna = False)
data["Region"].dropna(inplace = True) #inplace = True means do not assign new variable.
assert data["Region"].notnull().all()
#data frame creating from dictionary

mountains = ["Rocky","Everest"]

altitude = ["4401","8848"]

list_label = ["Mountain","Altitude"]

list_col = [mountains,altitude]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
df["Location"] = ["Nort America","Himalayas"]

df
df["broadcasting"] = 0

df
#plotting all data

data1 = data.loc[:,["Happiness_Score","Economy","Health"]]

data1.plot()

plt.show()
#subplots

data1.plot(subplots = True)

plt.show()
#Scatter plot

data.plot(kind = "scatter",x = "Economy",y = "Happiness_Score")

plt.show()
#Histogram plot

data1.plot(kind = "hist",y = "Happiness_Score",bins = 50,range = (0,10),normed = True)

plt.show()
fig,axes = plt.subplots(nrows = 2,ncols = 1)

data1.plot(kind = "hist",y = "Happiness_Score",bins = 50,range = (0,10),normed = True, ax = axes[0])

data1.plot(kind = "hist",y = "Happiness_Score",bins = 50,range = (0,10),normed = True, ax = axes[1]

          ,cumulative = True)

plt.show()
time_list = ["1981-04-17","1981-03-12"]

print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list) #changing type, str to datetime

data2["date"] = datetime_object             #adding to dataframe

data2 = data2.set_index("date")             # as an index !

data2
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = data.set_index("Happiness_Rank")  #indexing with happiness_rank

data.head()
data["Happiness_Score"][1]     #using square brackets
data.Happiness_Score[1]       #using column attribute and row label
data.loc[1,["Happiness_Score"]]  #using loc
data[["Happiness_Score","Economy"]]      #selecting only precise columns
data.loc[1:10,"Region":"Family"]    # ROWS --->  1 TO 10

                                    # COLUMNS -> Region to Family
data.loc[10:1:-1,"Region":"Family"]    #reverse slicing
data.loc[1:10,"Economy":]
boolean = data.Happiness_Score > 6

data[boolean]
filt1 = data.Happiness_Score > 6

filt2 = data.Economy > 1.25

data[filt1 & filt2]
data.Region[data.Happiness_Score > 6.8]
def div(m):

    y = m/2

    return y

data.Happiness_Score.apply(div)
data.Happiness_Score.apply(lambda m:m/2)
data["new_column"] = data.Happiness_Score - data.Standard_Error

data.head()
#We can learn our index name :

print(data.index.name)
data.index.name = "index_name"

data.head()
data3 = data.copy()

data3.index = range(100,258,1)

data3.head()
data3 = data.set_index(["Region","Happiness_level"])

data3.head(150)
dic = {"chemical" : ["X","Y","X","Y"],"reactivity" : ["low","high","low","high"],"price" : [34,14,56,78],

       "config" : [3,7,7,3] }

df = pd.DataFrame(dic)

df
#pivoting

df.pivot(index = "chemical",columns = "config",values = "price")
df1 = df.set_index(["chemical","config"])    

df1
df1.unstack(level = 0)  #level : position of unstacked index
df1.unstack(level = 1)
df2 = df1.swaplevel(0,1)

df2
df
pd.melt(frame = df,id_vars = "chemical",value_vars = ["reactivity","price"])
df
df.groupby("chemical").mean()
df.groupby("chemical").max()
df.groupby("chemical").price.max()
df.groupby("chemical")[["price","reactivity"]].min()