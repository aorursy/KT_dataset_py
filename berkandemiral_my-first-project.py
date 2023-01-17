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
data = pd.read_csv('../input/fifa19/data.csv') # pokemon.csv'nin içinde bu konuma ulaşabilriz.

# artık numpy,pandas veya matplotlib metotları kullanırken data. diyerek direrk bu veriler üzerinde işlem yapabiliriz.
data.head()
data.info() # achives basic information with ".info()"
data.describe() # achives istatistical data ---> ".describe()"
data.corr()

# corraletion değeri orantıyı ifade eder. Mesela HP-HP aynı oldukları için corr=1'dir.

# Değerler 0'a yaklaştıkça araraındaki doğru orantı azalır, 0'ın altına düşerse de Ters orantı başlar
#correlation map

f,ax = plt.subplots(figsize=(30, 30))

sns.heatmap(data.corr(), annot=True, linewidths=.10, fmt= '.1f',ax=ax)

plt.show()
data.columns # achives colums ---> if want column count information, we can use len(data.columns)
# Line Plot

data.BallControl.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

# BallControl degerini çiz(tür:line,0.5 saydamlıkta)

data.ShotPower.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')



plt.legend(loc='upper right')     # labelların görünmesi için

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()



# !!!!!!!!!!!!!!!!! line çizimi genelde x yerine zaman parametresi varsa tercih edilir.

# eğer line çizimi yapılacksa, her line için ayrı ayrı çizim yapılır.
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='BallControl', y='ShotPower',alpha = 0.5,color = 'orange')

plt.xlabel('BallControl')              # label = name of label

plt.ylabel('ShotPower')

plt.title('Top Kontrol ve Şut Gücü')            # title = title of plot



x = np.arange(1,120)

plt.plot(x,x,color="blue")

plt.show()
fig = plt.figure(figsize=(8,6))

axes = fig.add_axes([0,0,1,1])

plt.grid()

axes.plot(data.BallControl,data.ShotPower,lw=2,ls="--",marker="o",markersize=4,

         markerfacecolor="black",markeredgecolor="yellow")

plt.show()
# Histogram

data.ShotPower.plot(kind = 'hist',bins = 70,figsize = (12,12),grid=True)

plt.show()
mydict= {"Ball":"178₺","Desk":"98₺","Table":"76₺"}

print(mydict.keys())

print(mydict.values())
print(mydict)

mydict["Ball"] = "192₺"

print(mydict)

mydict["sofa"] = "427₺"

print(mydict)

del mydict["Table"]

print(mydict)

print("Ball" in mydict)

mydict.clear()

print(mydict)
# In order to run all code you need to take comment this line

del mydict         # delete entire dictionary     

print(mydict)       # it gives error because dictionary is deleted
def square(edge):

    area = edge*edge

    return area



print(f"area of this square: {square(5)}")
def totalofnumbers(*args):

    total = 0

    for i in args:

        total+=i

    return total

print(f"total of args: {totalofnumbers(2,4,6,8,12)}")
def f(**args):

    for key,value in args.items():

        print(key,":",value)



f(car="mercedes",model="benz",model2="CLA120")
total = lambda x,y,z:x+y+z

print(total(3,7,21))
my_list = [-2,4-8,11,-21]

funct = map(lambda x:x**(1/2),my_list)

new_list = list(funct)



zipped_List = list(zip(my_list,new_list))

print(zipped_List)
# Sayıların 1 fazlası



num1 = [3,7,11]

num2 = [i+1 for i in num1]



print(num1)

print(num2)
# birthdate --> 2010,1998,1968,1988, how old are they now? --> and do it zipped

birthdate = [2018,1998,1968,1988]

ages= [2020-i for i in birthdate]



newList = list(zip(birthdate,ages))

print(newList)
# multiply by 2 if number is odd else, add 50 to number

numbers = [1,3-2,5,7,21,35,40,50,23,47]

query = [i*2 if i%2!=0 else i+50 for i in numbers ]



print(numbers)

print(query)
data[(data['ShotPower']>92)]     # There are only 3 footballer who have shotpower value than 92
data[(data['ShotPower']>85) & (data['Balance']>85)] # There are only 6 footballer who have shotpower value than 85 and balance value than 85
firs5data = data.head() # --> first 5 row 

last5data = data.tail() # --> last 5 row

newData = pd.concat([firs5data,last5data],axis=0)

# --> we can conbine 2 DataFrame

newData 
newData.sort_values(["Release Clause"],ascending=False,inplace=True)

# we can sorting DataFrame according to the value we want
newData
newData.set_index(["Unnamed: 0"],inplace=True)
newData
#--> Drop the columns(Photo,Flag)

newData.drop(["Photo","Flag"],axis=1,inplace=True)
newData.drop(["Club Logo"],axis=1,inplace=True)

newData
treshold = sum(data.Potential)/len(data.Potential)

data["PotantialDegree"] = ["good" if treshold>55 else "not good" for i in data.Potential]

data.loc[:12,["Name","Age","Potential","PotantialDegree"]]
data.info()
data.Age.value_counts()

data.describe()
data.columns
melted = pd.melt(frame=data,id_vars="Name",value_vars=["Height"])

melted.head()
data1 = data.Age.head()

data2 = data.BallControl.head()

concat = pd.concat([data1,data2],axis=1)



concat
data.dtypes
assert  data['Age'].notnull().all() # returns nothing because we drop nan values

# dataframes from dictionary

country = ["Amerika","Türkiye","Fransa"]

population = [10,22,13]

list_label = ["country","population"]



list_col = [country,population]

print(list_col)



zipped = list(zip(list_label,list_col))

print(zipped)



data_dict = dict(zipped)

print(data_dict)



df = pd.DataFrame(data_dict)

df
df["capital"] = ["wachintondc","Ankara","Paris"]

df
df["incomde"] = 0

df
data1 = data.loc[:20,["Penalties","Marking","Special"]]

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="Penalties",y="Marking")

plt.show()
# hist plot  

data1.plot(kind="hist",y="Special",bins = 50,normed=True)

plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind = "hist",y = "Special",bins = 50,normed = True,ax = axes[0])

data.plot(kind = "hist",y = "Special",bins = 50,normed = True,ax = axes[1],cumulative = True)

plt.show()
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
print(data2.loc["1992-01-10":])
data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")



# değerleri regresyon ile otomatik olarak bir ortalama alarak hesaplıyor ve uyguluyor
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
data2[["Name","Age","Overall","Potential"]]
data.loc[5:15,["Name","Age","Overall","Potential"]]
data.loc[5:15,"Potential":]
boolean = data.Potential > 80

data[boolean]
def div(n):

    return n/2

data.Potential.apply(div)
# Or we can use lambda function

data.Potential.apply(lambda n : n/2)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1
df2 = df.set_index(["gender","treatment"])

df2
df.groupby("treatment")[["age","response"]].min() 