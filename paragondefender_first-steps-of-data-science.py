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
data_2015=pd.read_csv('../input/2015.csv')

data_2016=pd.read_csv('../input/2016.csv')

data_2017=pd.read_csv('../input/2017.csv')

data_2015.info()

print("****************************************")

data_2016.info()

print("****************************************")

data_2017.info()

data_2015.corr()

data_2016.corr()

data_2017.corr()

#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data_2015.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

plt.show()

data_2015.head(3)



data_2016.head(3)
data_2017.head(3)
data_2015.columns





data_2016.columns

data_2017.columns

data_2015.columns=[each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data_2015.columns]

data_2016.columns=[each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data_2016.columns]

data_2015.rename(columns = {"Economy_(GDP": "Economy","Health_(Life": "Health","Trust_(Government": "Trust"},inplace = True)

data_2016.rename(columns = {"Economy_(GDP": "Economy","Health_(Life": "Health","Trust_(Government": "Trust"},inplace = True)

data_2015.columns



    

    

    
data_2017.columns=[each.replace(".","_") for each in data_2017.columns]

data_2017.columns
### Line Plot

data_2015.Happiness_Score.plot(kind='line',color='y', label='Happiness Score 2015',linewidth=1.5,alpha=0.9,grid=True,linestyle='--')

data_2016.Happiness_Score.plot(kind='line',color='b', label='Happiness Score 2016',linewidth=1.5,alpha=0.9,grid=True,linestyle='--')

data_2017.Happiness_Score.plot(kind='line',color='g', label='Happiness Score 2017',linewidth=1.5,alpha=0.9,grid=True,linestyle='--')

plt.legend(loc='upper right')

plt.xlabel('Rank')

plt.ylabel('Score')

plt.title('Happiness Score Line Plot')

plt.show()

# Scatter Plot 

# x = Economy, y = Happiness Score

data_2015.plot(kind='scatter', x='Economy', y='Happiness_Score',alpha = 0.5,color = 'orange')

data_2016.plot(kind='scatter', x='Economy', y='Happiness_Score',alpha = 0.5,color = 'black')

data_2017.plot(kind='scatter', x='Economy__GDP_per_Capita_', y='Happiness_Score',alpha = 0.5,color = 'purple')

plt.xlabel('Economy')              # label = name of label

plt.ylabel('Happiness Score')

plt.title('Economy Happiness Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data_2015.Happiness_Score.plot(kind = 'hist',bins = 10,figsize = (10,10))

plt.show()
#create dictionary and look its keys and values

dictionary={'Name':'Murat', 'Height':'180','Weight':'70'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Name']="Ahmet"        #update existing entry

print(dictionary)

dictionary['Country']="Turkey"    # add new entry

print(dictionary)

del dictionary['Weight']          #remove 'Weight'

print(dictionary)

print('Height' in dictionary)     #check include or not

dictionary.clear()                #remove all 

print(dictionary)

data=pd.read_csv('../input/2015.csv')
series=data['Country']           #data['Country'] = series

print(type(series))

data_frame=data[['Country']]     #data[['Country']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1- Filtering Pandas data frame 

meanHappiness_Score2015=data_2015.Happiness_Score.mean()

meanHappiness_Score2016=data_2016.Happiness_Score.mean()

meanHappiness_Score2017=data_2017.Happiness_Score.mean()



x=data_2015['Happiness_Score']>meanHappiness_Score2015*1.33

data_2015[x]

# 2- Filtering pandas with logical

data_2016[(data_2016['Happiness_Score']>meanHappiness_Score2016) & (data_2016.Economy.mean()>data_2016['Economy'])]

### or

#data_2016[np.logical_and(data_2016['Happiness_Score']>meanHappiness_Score2016), (data_2016.Economy.mean()>data_2016['Economy'])]
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['Country']][0:1].iterrows():

    print(index," : ",value)
# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# guess print what

x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope
# What if there is no local scope

x = 5

def f():

    y = 2*x        # there is no local scope x

    return y

print(f())         # it uses global scope x

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
#nested function

def ort():

    """ return square of value """

    def topla():

        """ add two local variable """

        vize = 20

        final = 80

        topl = vize + final

        return topl

    return topla()/2

print(ort())    
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,"murat","ali")





# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(4))



#long way

def Square(x):

    return x**2

print(Square(5))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))



#long way



def Tot(x,y,z):

    return x+y+z

print(Tot(1,5,9))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
# iteration example

name = "murat"

it = iter(name)

print(next(it))    # print next iteration

print(next(it))

print(*it)         # print remaining iteration



#population=985614

#it2=iter(population)

#print(next(it2))

#print(*it2)



#'int' object is not iterable



# Example of list comprehension

num1 = [1,2,3]

num2 = [i**2 for i in num1 ]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
threshold=sum(data_2015.Happiness_Score)/len(data_2015.Happiness_Score)

data_2015["Happiness_Level"]=["high" if i > threshold else "low" for i in data_2015.Happiness_Score]

data_2015.loc[:5,["Happiness_Level","Happiness_Score"]]



data=pd.read_csv('../input/2015.csv')

data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Region'].value_counts(dropna =False))  # if there are nan values that also be counted
data.describe()
# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='Happiness Score',by = 'Region',rot=90, fontsize=8)
data_new=data.head()

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted=pd.melt(frame=data_new,id_vars='Country',value_vars=['Happiness Score','Freedom'])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Country', columns = 'variable',values='value')
 # Firstly lets create 2 data frame

data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)  # axis = 0 : adds dataframes in row

conc_data_row

data11=data['Generosity'].head()

data22=data['Freedom'].head()

conc_data_col=pd.concat([data11,data22],axis=1)

conc_data_col
data.dtypes
data['Happiness Rank']=data['Happiness Rank'].astype('float')

data['Region']=data['Region'].astype('category')
data.dtypes

#data frames from dictionary

name=["murat","volkan","sedat"]

province=["kocaeli","erzurum","ankara"]

list_label=["name","province"]

list_col=[name,province]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
#add new columns

df["age"]=[21,28,32]

df
#broadcasting

df["height"]=180

df
# Plotting all data 

data1=data.loc[:,["Family","Freedom","Generosity"]]

data1.plot()

# it is confusing
#subplots

data1.plot(subplots=True)

plt.show()
#scatter plot

data1.plot(kind="scatter",x="Freedom",y="Generosity")

plt.show()
#hist plot

data1.plot(kind="hist",y="Freedom",bins=50,range=(0,1),normed=True)

plt.show()

#histogram subplot with non cumulative and cumulative

fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="Freedom",bins=50,range=(0,1),ax=axes[0])

data1.plot(kind="hist",y="Freedom",bins=50,range=(0,0.8),ax=axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
data.Region.unique()





data.Region.value_counts()
time_list=["1997-08-12","2019-01-21"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))

### close warning

import warnings

warnings.filterwarnings("ignore")

data2=data.head()

date_list=["1997-01-02","1997-08-12","1997-07-19","2008-07-12","2012-10-23"]

datetime_object=pd.to_datetime(date_list)

data2["date"]=datetime_object

# lets make date as index

data2=data2.set_index("date")

data2
# now we can select according to our date index

print(data2.loc["1997-01-02"])

print(data2.loc["1997-01-01":"2010-05-23"])
# we will use data2 that we create at previous part

data2.resample("A").mean()
# lets resample with month

data2.resample("M").mean()
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data_t=pd.read_csv('../input/2015.csv')

data_t=data_t.set_index("Happiness Rank")

data_t.head()
#indexing using square brackets

data_t["Family"][4]
# using columns attribute and row label

data_t.Family[4]
#using loc accessor

data_t.loc[1,["Family"]]
# selecting only some columns

data_t[["Family","Freedom"]]
# difference between selecting columns: series and dataframes

print(type(data_t["Family"])) #series

print(type(data_t[["Family"]])) #data frames
# slicing and indexing series

data.loc[1:9,"Happiness Rank":"Family"]
#reverse slicing

data.loc[10:1:-1,"Happiness Rank":"Family"]
data.loc[1:10,"Standard Error":]
boolean=data.Family>1.31

data[boolean]
#combining filters

first_filter=data.Family>1.31

second_filter=data.Freedom>0.64

data[first_filter&second_filter]
#filtering column based others

data.Region[data.Generosity<0.1]
# plain python functions

def div(n):

    return n*10

data.Generosity.apply(div)
# or we can use lambda function

data.Generosity.apply(lambda n:n*10)
data["my_rank"]=data.Freedom+data.Family+data.Generosity

data=data.set_index("my_rank")

data.head()
#our index name is this:

print(data.index.name)

#lets change it

data.index.name="index_name"

data.head()
#overwrite index

#if we want to modify index we need to change all of them

data.head()

#first copy of our data to data3 then change index

data3=data.copy()

# lets make index start from 100. it is not remarkable change but it is just example

data3.index=range(100,258,1)

data3.head()
data4=pd.read_csv('../input/2015.csv')

data4.head()
data5=data4.set_index(["Region","Family"])

data5.head(30)
dic={"lesson":["Math","Math","Phy","Phy"],"gender":["F","M","F","M"],"vize":["55","40","70","40"],"final":["60","70","60","80"]}

df=pd.DataFrame(dic)

df
#pivoting

df.pivot(index="lesson",columns="gender",values="final")
df1=df.set_index(["lesson","gender"])

df1

#lets unstack it
#level determines indexes

df1.unstack(level=0)

df1.unstack(level=1)
#change inner and outer level index position

df2=df1.swaplevel(0,1)

df2
df
#df.pivot(index="lesson",columns="gender",values="final")

pd.melt(df,id_vars="lesson",value_vars=["vize","final"])
dic={"lesson":["Math","Math","Phy","Phy"],"gender":["F","M","F","M"],"vize":[55,40,70,40],"final":[60,70,60,80]}

df3=pd.DataFrame(dic)

df3
df3.groupby("lesson").mean()
df3.groupby("lesson").vize.max()
df3.groupby("lesson")[["vize","final"]].min()
df3.info()
WestEU=data_2015[data_2015.Region=="Western Europe"]

mean_WestEU=WestEU.Happiness_Score.mean()

NorthA=data_2015[data_2015.Region=="North America"]

mean_NorthA=NorthA.Happiness_Score.mean()

Aust=data_2015[data_2015.Region=="Australia and New Zealand"]

mean_Aust=Aust.Happiness_Score.mean()

MiddAfrica=data_2015[data_2015.Region=="Middle East and Northern Africa"]

mean_MiddAfrica=MiddAfrica.Happiness_Score.mean()

Latin=data_2015[data_2015.Region=="Latin America and Caribbean"]

mean_Latin=Latin.Happiness_Score.mean()

Asia=data_2015[data_2015.Region=="Southeastern Asia"]

mean_Asia=Asia.Happiness_Score.mean()

EastEU=data_2015[data_2015.Region=="Central and Eastern Europe"]

mean_EastEU=EastEU.Happiness_Score.mean()

EastAsia=data_2015[data_2015.Region=="Eastern Asia"]

mean_EastAsia=EastAsia.Happiness_Score.mean()

SubAf=data_2015[data_2015.Region=="Sub-Saharan Africa"]

mean_SubAf=SubAf.Happiness_Score.mean()

SouthAsia=data_2015[data_2015.Region=="Southern Asia"]

mean_SouthAsia=SouthAsia.Happiness_Score.mean()



dictionary={"Region":["Western Europe","North America","Australia and New Zealand",

                      "Middle East and Northern Africa","Latin America and Caribbean",

                      "Southeastern Asia","Central and Eastern Europe","Eastern Asia","Sub-Saharan Africa","Southern Asia"],

            "Mean":[mean_WestEU,mean_NorthA,mean_Aust,mean_MiddAfrica,mean_Latin,mean_Asia,mean_EastEU,mean_EastAsia,mean_SubAf,mean_SouthAsia]}

dataframe1=pd.DataFrame(dictionary)

dataframe1
plt.barh(dataframe1.Region,dataframe1.Mean)

plt.show()