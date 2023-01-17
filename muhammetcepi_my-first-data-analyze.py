# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plot graphs to analyze the data

from __future__ import print_function

import csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv")

data.info()
data.head()
data.corr()
data.Overall.plot(kind='line',color="red",label="Overall",alpha=0.5,grid=True,linestyle="-",linewidth=5)

data.Potential	.plot(kind ='line',color='green',label='Potential',alpha=0.5,grid=True,linestyle='-.',linewidth=5)

plt.legend(loc="upper right")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Overall Potential Line Plot")

data.plot(kind="scatter",x="Overall",y="Potential",alpha=0.5,color='g')

plt.xlabel("Overall")

plt.ylabel("Potential")

plt.title("Overall Potential Scatter Plot")

plt.scatter(data.Overall,data.Potential,alpha=0.5,color='green')
data.Overall.plot(kind='hist',bins=20)

plt.show()
Dictionary={}

file = '../input/data.csv'

with open(file) as fh:

    rd = csv.DictReader(fh) 

    for row in rd:

        keys=row.keys()

        values=row.values()

        for key in row.keys(): 

            Dictionary[key]=row[key]

             

print(Dictionary)    


#del Dictionary['Flag']    #deletes key 'Flag'

#Dictionary.clear()     #clears the dictionary.

#del Dictionary        #deletes entire Dictionary

#print(Dictionary)       #you cannot print the Dictionary after it's deleted. It will give an error.

#I prefer not to print the dictionary it increases the run time
data=pd.read_csv("../input/data.csv")
series=data["ID"]

print(type(series))

print(series)

data_frame=data[["ID"]]

print(type(data_frame))

print(data_frame)

print(3>2)

print(3<2)

print(3==3)

print(True and False)

print(True or False)
#Filtering with Pandas to Data Frame

x=data['Overall']>90 #with this statement I search for Player who have a higher Overall value than 90

data[x]
#Filtering with Pandas wih Logical_and

data[np.logical_and(data['Overall']>85,data['Potential']>90)] 

#with this statement I look for players who have higher Overall value than 85 'and' players who have higher Potential value than 90
i=0

while i != 5:

    print("i is",i)

    i+=1
my_list=[1,2,3,4,5]

for i in my_list:

    print("i is",i)

    

print(" ")   

#For in Dictionaries

for key,value in Dictionary.items():

    print(key,":",value)
def avg_number(x, y):

    print("Average of ",x," and ",y, " is ",(x+y)/2)

avg_number(3, 4)



def my_function():

    t=(1,2,3,4,5,6)

    return t

a,b,c,d,e,_=my_function()  #we use _ for the variable we don't actually want to use but in any case you can use it as below.

print(a,b,c,d,e,_)
x=5    #x=5 is a global scope

def func():

   # x=x*x if you use this here you will get an UnboundLocalError: local variable 'x' referenced before assignment

    x=3   #=3 is a local scope.

    x=x*x

    return x

print(x)   #this is printing the global scope

print(func()) #and this is printing the local scope.

print(x)  #as yu can see below the global scope didn't change.

#As I mentioned above in desciption of Scope.There are 3 types of scopes.1.Global 2.Local 3.Built-in

#So how can we learn what is a built-in scope

import builtins

dir(builtins)

    
def multiply(x,y): 

    def add():

        summation=0

        for i in range(0,y):

            summation=summation+x

        return summation

    print(add())



multiply(3,5)
#Default Arguments

def func(a,b=1,c=3):

    print(a,b,c)

func(5)

func(5,3,2)# What if I want to change the arguments? 
#Flexible Arguments

#Args

def func(*args):

    summation=0

    for i in args:

        summation+=i

    return summation

func(7,9,15,63,123)#there is no limit of parameters

#kwargs

#don't forget the dictionary I defined in the first part. I am going to use it.

def func(**kwargs):

    for key,value in kwargs.items():

        print(key,":",value)

        

func(country='turkey',name='muhammet',lastname='cepi')

print("----------------------")

func(**Dictionary)
def multiplying(x,y):   #I used 4 lines to multiply two numbers. Instead I can use only one line to create the sam function.

    multiplied=0

    multiplied=x*y

    return multiplied



multiply=lambda x,y:x*y

print(multiplying(3,5),"the answer of the normal function")

print(multiply(5,3),"the answer of lambda funciton")

mylist=[1,2,3,4]

myslist=[3,4,5,6]

y=map(lambda x,z:z*x,mylist,myslist)

print(list(y))
h_letters = []



for letter in 'human':

    h_letters.append(letter)



print(h_letters)



list1=[1,2,3]

list2=[i+1 for i in list1]

print("list2=",list2)
data=pd.read_csv('../input/data.csv')

data.head()

#we would likely use data.ClubLogo but as you can see below the column name is Club Logo.

#There is a space. So we should use data["Club Logo"]

data.tail()
data.columns #to see the columns we use it.
print(data["Weak Foot"].value_counts(dropna=False)) #dropna: if there are any nan values. Count them too.
data.describe()
data.boxplot(column="Overall",by="Potential")

plt.show()

#it's a good Idea to plot Overall by Potential but the plot is not looking ordinary.

#so I am going to use non-sense columns for an ordinary plot.
data.boxplot(column="Weak Foot",by="Skill Moves")

plt.show()
data_new=data.head()  #first I created a new data to explain in easily.

data_new
#So I am going to melt the data

melted=pd.melt(frame=data_new,id_vars="Name",value_vars=["Overall","Potential"])

melted

#melt is a sort of a bridge between pandas and seaborn.
melted.pivot(index="Name",columns="variable",values="value")
#first we need two dataframes

data1=data.head()

data2=data.tail()

pd.concat([data1, data2],ignore_index=True,axis=0) #you shoul not use axis=0 if ypu want to concat the data in rows.
data1=data.Overall.head()

data2=data.Potential.head();

pd.concat([data1, data2],ignore_index=True,axis=1) #but you must use axis=1 if you want to concat the column
#data.dtypes 
data["Potential"]=data["Potential"].astype("float")

data["Name"]=data["Name"].astype("category")

data.dtypes

#as you can see below the data types of the name and Potential have changed.
data.head(10) #as you can see below by the column "Loaned From" there are NaN values. These are missing datas.

#There are several ways to deal with them.
df=data

#df.dropna(inplace=True) #this deletes all data that has one or more missing values.

#But sadly it deleted everything because obviusly every data had one or more missing value.    :(

df.tail(10)
df['Loaned From'] = df['Loaned From'].fillna("Real Madrid")

data.head(10) #so they are all loaned from Real Madrid
data1=data

data1.LS.dropna(inplace=True)
assert data1.LS.notnull().all() #it gives us nothing so it worked.
footballer=["Messi","Ronaldo"]

countries=["Argentina","Portugal"]

list_label=["footballer","countries"]

list_col=[footballer,countries]

zipped=list(zip(list_label,list_col))

dictio=dict(zipped)

dataf=pd.DataFrame(dictio)

print(dataf)

#So above you see how to create two lists and concat them and then turn to zipped form and then to dictionary and then to dataframe wow that's a lot fo things.

#If you don't have a dictionary already you have to make the thing above.

#But let's use my Dictionary that I created at the top of my Kernel. I had a litle bit problems with it but I was able to solve it.

#Using only one statement wil do the job for you ıf you have a Dictionary created already.

df=pd.Series(Dictionary).to_frame()

df
# I am going to add a new column but not to dataframe based on the dictionary at the top of my kernel.

dataf["Overall"]=[94,93]

dataf
#Broadcasting

dataf["potential"]=95 #95 will be the value of the entire column

dataf
data1=data.loc[:,["Overall","LongShots","Acceleration"]]

data1.plot() # it's confusing
data1.plot(subplots=True)

plt.show()
#scatter plot

data.plot(kind="scatter",x="Overall",y="Potential")
#histogram

data.plot(kind="hist",y="Overall",bins=50,range=(0,250),normed=True)
fig,axes=plt.subplots(nrows=2,ncols=1)

data.plot(kind="hist",y="Overall",bins=50,range=(0,250),normed=True,ax=axes[0])

data.plot(kind="hist",y="Overall",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig("mygraph.png")

plt
time_list=["2019-02-24","2019-02-25"]

print("type of time_list=")

print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)

print("type of datetime_object")

print(type(datetime_object))
#close warnings

import warnings

warnings.filterwarnings("ignore")

#as we have no date time index we should add it to our data frame

data2=data.head()

date_list=["2019-02-24","2019-03-25","2019-04-26","2020-05-27","2019-02-28"]

datetime_object=pd.to_datetime(date_list)

data2["date"]=datetime_object

data2=data2.set_index("date")

data2
#now we select the row we want by datetime

data3=data2.loc["2019-02-28"]

print(data3)

print("\n")

data4=data2.loc["2019-02-24":"2019-05-27"]

print(data4)
#list resample by year

data2.resample("A").median()
#list resample by month

data2.resample("M").median()
#there are Nan values and these must be filled.I am going to use interpolate. If you write to google pandas interpolate you will see the methods.



data2.resample("M").first().interpolate("from_derivatives")
#I used here another method. There are 7 methods.

data2.resample("M").mean().interpolate("linear")
data=data.set_index("Unnamed: 0") #with this statement you bring the column you want to row.

#data.index=data["Unnamed: 0"]

data.head()
#data[:] to call everything

#data[:][0:1] # to call one row

data["Nationality"][1] #to call the nationality of one player.
data.iloc[0]   #row selection

#data.iloc[:,5] #column selection

#data.iloc[0:5,2:5] #multiple rows and columns selection. 
data.iloc[0,5]#row and column selection. Exact value selection
data.loc[0:5,"Age":"Nationality"]
boolean=data.Overall>90

data[boolean]
#we can combine two boolean series

boolean_first=data.Overall>90

boolean_second=data.Potential>92

data[boolean_first&boolean_second]
#confusing thing just look at the example.

data.Potential[data.Overall>90]
def div(n):

    return 100-n

data.Overall.apply(div).head(10)
data.Overall.apply(lambda n:100-n).head(10) #the same fucniton as above
data["Overtial"]=data.Overall+data.Potential

data.Overtial.head()

#or you can use data.head() and take a look to the rightest column
#our index name:

print(data.index.name)

#we can change it this way

data.index.name="MyID"

data.head()
#data.info() I used to count the entries.

#I can change the range of the Index

data.index=range(100,18307,1)

data.head()
#I read the data again. It can be confusing for me if I continue with same data as above

data=pd.read_csv("../input/data.csv")

data.head()
data1=data.set_index(["Overall","Potential"])

data1.head()