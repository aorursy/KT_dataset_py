# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





#wordcloud

from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# First we read data

data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

data.head(2)
# Then we take some columns which we use in there.

data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive','country':'Country_ID','targsubtype1':'Target_ID'},inplace=True)

data=data[['Year','Month','Day','Country_ID','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_ID','Target_type','Weapon_type','Motive']]

data.head(2)
# Create two dataframes for Attack type:

data_attck_bomb=data[data.AttackType=='Bombing/Explosion']

data_attck_facility=data[data.AttackType=='Facility/Infrastructure Attack']
type(data_attck_bomb)
#data preparetion

countries_list = data.Country

plt.subplots(figsize=(15,12))



wordcloud = WordCloud(

                        background_color = "white",

                        width = 450,

                        height = 384

                        ).generate(" ".join(countries_list))

plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("graph.png")



plt.show()
# desciribe dataframe

data_attck_facility.describe()
data.corr()
# correlation map

f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
data.columns
#line plot

data_attck_bomb.Killed.plot(kind = 'line',color = 'g',label='bomb_killed',linewidth=1,alpha=0.5,grid=True,linestyle='--')

data_attck_facility.Killed.plot(color='r',label='facility_killed',linewidth=1,alpha=0.5,grid=True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line plot')

plt.show()
# line plot2

plt.plot(data_attck_bomb.Killed,data_attck_bomb.Wounded,color='r',linewidth=1,alpha=0.5,label='bomb_kill_wound')

plt.plot(data_attck_facility.Killed,data_attck_facility.Wounded,color='g',linewidth=1,alpha=0.5,label='facility_kill_wound')

plt.xlabel("Killed")

plt.ylabel("Wounded")

plt.legend()

plt.title("Line plot 2")

plt.show()
# line plot3

data.Wounded.plot(kind = 'line',color = 'b',label='Wounded',linewidth=1,alpha=0.5,grid=True,linestyle='--')

data.Killed.plot(color='r',label='Killed',linewidth=1,alpha=0.5,grid=True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line plot')

plt.show()
# Histogram

#bins number of figure

data.Year.plot(kind='hist',bins=70,figsize=(15,15))

plt.show()
# Countplot

plt.subplots(figsize=(17,7))

sns.countplot('Year',data=data,palette='vlag',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Every Year Terrorism Countplot')

plt.show()
# Countplot 2

plt.subplots(figsize=(15,5))

sns.countplot('AttackType',data=data,palette='rocket',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('AttackType Countplot')

plt.show()
# 1- Filtering Pandas DataFrame

x = data['Country_ID']>1003

x.sum() # There are only 225 countries where have higher ID value than 1003
# 2 - Filtering pandas with logical_and

# There are only 2 countries where have higher country_id value than 1003 and higher target_id value than 100

data[np.logical_and(data['Country_ID']>1003,data['Target_ID']>100)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Country_ID']>1003) & (data['Target_ID']>100)]
data1=data.Country.unique()

data1.size

print(type(data1))
# Created new dictionary

data_dict={}

b=0

for i in data1:

    data_dict[b]=i

    b+=5

    if b>=data1.size:

        break
data_dict
print(50 in data_dict)

print(data_dict[50])
del data_dict[50]

print(50 in data_dict)
# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

for key,value in data_dict.items():

    print(key," : ",value)

print('')

# For pandas we can achieve index and value

for index,value in data[['Country']][0:3].iterrows():

    print(index,": ",value)

# Example user defined function

def tupple_square():

    t = (2,4,6)

    return t

a,b,c = tupple_square()

print(a,b,c)
# guess print what

x = 4

def func():

    x = 8

    return x

print(x)

print(func())
# What if there is no local scope

x = 10

def fun():

    y=(x**2)/4

    return y



print(fun())
# How can we learn what is built in scope

import builtins

dir(builtins)
# nested function

def nest():

    "return square of value"

    def an():

        y = 25

        x = 13

        z = (y - x)*(y + x)

        return z

    return (an()**2)



print(nest())
#default arguments

def f(a, b, c=34, d=23):

    y = (a*d)+b-c

    return y

print(f(1,2))

print(f(1,2,3,4))
# flexible arguments *args

def flex(*args):

    s = 1

    for i in args:

        s=s*i

    return s

print(flex(1,2,3))

print(flex(5,6,7,8,9))

# flexible arguments **kwargs that is dictionary

def flex2(**kwargs):

    for key,value in kwargs.items():

        print("{} key values is {}".format(key,value))

        

flex2(country = "spain",capital = 'madrid',best_team = 'Barcelona')
# lambda function

squaret = lambda x: x**0.5     # where x is name of argument

print(squaret(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(9,25,3))
# lambda function

squaret = lambda x: x**0.5

print(squaret(169))

tot = lambda x,y,z: x*y-z

print(tot(4,5,6))
# iterator example



name='messi'

it = iter(name)

print(next(it))

print(next(it))

print(next(it))

print(*it)

# zip example

list1=[1,2,3,4]

list2=[11,12,13,14]

z = zip(list1,list2)

print(z)

z = list(z)

print(z)
# unzip example

unzip = zip(*z)

print(unzip)

unlist1,unlist2 = list(unzip)

print(unlist1)

print(unlist2)

type(unlist1)
# Example of list comprehension

num1 = [1,2,3,4,5,6,7,8,9]

num2 = [i**2 for i in num1]

print(num1)

print(num2)
# Conditionals on iterable

num1 = [1,2,3,4,5,6,7,8,9]

num2 = [i*10 if i // 3==0 else i*5 if i % 3 ==2 else i*4 for i in num1]

print(num2)
data2 = data[['Country_ID','Country']]

average = sum(data2.Country_ID)/len(data2.Country_ID)

data2['new_calc'] = ["C_ID is high" if i > average else "C_ID is low" for i in data2.Country_ID]

data2.head(10)
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive','country':'Country_ID','targsubtype1':'Target_ID'},inplace=True)

data=data[['Year','Month','Day','Country_ID','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_ID','Target_type','Weapon_type','Motive']]

data.head() # head show first 5 rows
data.tail() # tail show last 5 rows
#shape gives number of columns and rows in tupple

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
# Forx example lets look Attack of terrorist types

print(data['AttackType'].value_counts(dropna=False)) # if there are nan values that also be counted

# As it can be seen below there are 42669 Armed Assault terror type or 19312 Assassination terror
data.describe()
data.boxplot(column='Target_ID',by='Killed')
data_new=data.head()

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Country',value_vars= ['city','latitude'])

melted
# Index is country

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index='Country',columns='variable',values='value')
# Firstly lets create 2 data frame

data1=data.head()

data2=data.tail()

conc_dat_rows=pd.concat([data1,data2],axis=0,ignore_index=True) # axis = 0 : adds dataframes in row

conc_dat_rows
data.dtypes
# lets convert object(str) to categorical and int to float.

data['Day']=data['Day'].astype('float')

data['Target_type']=data['Target_type'].astype('category')
# As you can see Target_type is converted from object to categorical

# And Day ,s converted from int to float

data.dtypes
# lets look at does terrorism dataframe have nan value

data.info()
# lets ckeck motive

data['longitude'].value_counts(dropna=False)

# as you can see there're 4557 NAN value
# lets drop NaN values

data1=data 

data1['longitude'].dropna(inplace=True)
#  Lets check with assert statement what is this worked or not

# Assert statement:

assert 1==1
assert data1['longitude'].notnull().all()
    # # With assert statement we can check a lot of thing. For example

    # assert data.columns[1] == 'Name'

    # assert data.Speed.dtypes == np.int
# data frames from dictionary

country=data['Country'].unique()

country_id=data['Country_ID'].unique()

list_label=["country","country_id"]

list_col=[country,country_id]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df = pd.DataFrame(data_dict)

df1 = df.head()

df1
# Add new columns

df1["capital"]=["Santo Domingo","Mexico City","Manila","Athens","Tokyo"]

df1
# broadcasting

df1["income"]=100

df1
# Plotting all data

data1 = data.loc[:,["Killed","Wounded"]]

data1.plot()

plt.show()
# subplots

data1.plot(subplots=True)

plt.show()
# scatter plot

data1.plot(kind="scatter",x="Killed",y="Wounded")

plt.show()
#hist plot

data.plot(kind="hist",y="Target_ID",bins = 50,range = (0,120),normed=True)

plt.show()
##### histogram subplot with non cumulative and cumulative

fig,axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind="hist",y = "Target_ID",bins=50,range=(0,120),normed=True,ax=axes[0])

data.plot(kind="hist",y = "Target_ID",bins=50,range=(0,120),normed=True,ax=axes[1],cumulative=True)

plt.savefig("praph.png")

plt
df1.describe()
#close warnings

import warnings

warnings.filterwarnings("ignore")



date_list=["1989-03-15","1989-03-16","1989-03-17","1990-06-18","1990-06-19"]

dtime_object=pd.to_datetime(date_list)

df1["date"]=dtime_object

# lets make date as index

df1 = df1.set_index("date")

df1
# Now we can select according to our date index

print(df1.loc["1989-03-16"])

print(df1.loc["1989-03-17":"1990-06-19"])
# We will use df1 that we create at previous part

df1.resample("A").mean()
# lets resample with month

df1.resample("M").mean()
# We can interpolete from first value

df1.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

df1.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive','country':'Country_ID','targsubtype1':'Target_ID'},inplace=True)

data=data[['Year','Month','Day','Country_ID','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_ID','Target_type','Weapon_type','Motive']]
data1=data.head(100)

data2=data.tail(100)

data_new=pd.concat([data1,data2],axis=0,ignore_index=True)

data_new.head()
# add new columns

#i=0

TID=[]

for i in range(len(data_new)):

    TID = TID+[i+1]

data_new["ID"]=TID    

data_new = data_new.set_index("ID")

data_new.head()
# using column attribute and row label

data_new.Country[20]
# indexing using square brackets

data_new["Country"][20]
# using loc accessor

data_new.loc[20,["Country"]]
# Selecting only some columns

data_new[["Country","city"]]
# Difference between selecting columns: series and dataframes

print(type(data_new["city"]))

print(type(data_new[["city"]]))
# Slicing and indexing series

data_new.loc[1:10,"Country":"city"]
# Reverse slicing

data_new.loc[10:1:-1,"Country":"city"]
# Creating boolean series

boolean = data_new.Country_ID>400

data_new[boolean]
# Combining filters

first_one = data_new.Country_ID>300

second_one = data_new.Target_ID>40

data_new[first_one & second_one]
# Filtering column based others

data_new.city[data_new.Target_ID>100]
# Plain python functions

def div(n):

    return n/2

data_new.Country_ID.apply(div)
# Or we can use lambda function

data_new.Country_ID.apply(lambda x : x/2)
# Defining column using other columns

data_new["unreasonable"]=data_new.Country_ID+data_new.Target_ID

data_new.head()
# our index name is this:

print(data_new.index.name)

# lets change it

data_new.index.name="index_id"

data_new.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data_new.head()

data3 = data_new.copy()



data3.index = range(100,300,1)

data3.head()
data_new.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive','country':'Country_ID','targsubtype1':'Target_ID'},inplace=True)

data=data[['Year','Month','Day','Country_ID','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_ID','Target_type','Weapon_type','Motive']]

data.head()
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Country","city"])

data1.head(20)
dict = {"treatment":["A","A","B","B","C","C"],"gender":["F","M","F","M","F","M"],"response":[10,45,5,9,24,30],"age":[15,4,72,65,21,38]}

data_dict = pd.DataFrame(dict)

data_dict
# pivoting

data_dict.pivot(index="treatment",columns="gender",values="response")
data_dict = data_dict.set_index(["treatment","gender"])

data_dict

# lets unstack it
# level determines indexes

data_dict.unstack(level=0)
data_dict.unstack(level=1)
# change inner and outer level index position

data2 = data_dict.swaplevel(0,1)

data2
data_dict
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(data_dict,id_vars="treatment",value_vars=["age","response"])
# We will use data_dict

data_dict.groupby("treatment").mean()
# we can only choose one of the feature

data_dict.groupby("treatment").age.min()
# Or we can choose mutliple columns

data_dict.groupby("treatment")[["age","response"]].max()
data_dict.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#data_dict["gender"] = data_dict["gender"].astype("category")

#data_dict["treatment"] = data_dict["treatment"].astype("category")

#data_dict.info()
