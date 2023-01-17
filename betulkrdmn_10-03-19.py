# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # visualization tool

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data.csv')
data.info()
data.columns
data.head(8)
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax)





data.Overall.plot(kind="line",color="r",label="Overall",linewidth=1,alpha=1,grid=True,linestyle=':')



data.plot(kind='scatter',x= 'Age',y='Overall',alpha=0.5,color='red')

plt.legend("upper-right")

plt.xlabel("Age")

plt.ylabel("Value")
data.columns
data.Age.plot(kind='hist', bins=40,figsize=(15,15),color='r')
dictionary={'turkey':'Ankara','Germany':'Berlin'}

dictionary['France']='Paris' #add france

print(dictionary.keys())
print(dictionary.values())
del dictionary["Germany"] #delete Germany

print(dictionary)
del dictionary #del dictionary for alan kazanmak için
values=data['Value']

print(values)

print(type(values))
x=data['Age']>25

data[x]
data[np.logical_and(data['Age']>30,data['Overall']>90)]
data.info()
Dictionary={'Dinner':'Sushi','Drink':'Coke','Dessert':'Cheescake'}

print(Dictionary)
print(Dictionary.keys())
print(Dictionary.values())
Dictionary['soup']="Tomatoes Soup"
print(Dictionary.values())
Dictionary["Drink"]="Coffee"
del Dictionary['Dinner'] #just dinner data
del Dictionary #all Dictionary delete
def sum():

    a=5

    b=4

    z=a+b

    return z



print(sum())

a=7   #If "a" is defined in the function it is used

#isn't defined used to this 

def sum():

    b=3

    z=a+b

    return z

        

print(sum())













a=4

def sum():

    a=10

    b=1

    return a+b



print(sum())

def Multi():

    def sum():

        a=4

        b=2

        return a+b

    c=2

    return sum()**c

print(Multi())
def trial(a,b=2): #defaul b value is 2 if we write diffirent  value b is change

    y=a+b

    return y

print(trial(7))
def trial(a,b=2):

    y=a+b

    return y

print(trial(7,4))

def args1(*args): #if the number of values is not certain use to

    for i in args:

        print(i)

args1(1,2,3,44,54,66)        

#if args use for Dictionary >**kwargs
def trial_2(**kwargs): #for Dictionary

    for key, value in kwargs.items():

         print("{0} = {1}".format(key, value))

    

trial_2(dessert="Tiramisu")







        
square=lambda x:x**2 #lambda is easier than user define function

print(square(4))
number_list=[24,5,96,7] 

y=map(lambda x:x**2,number_list) #for list use to map(lambda) functionn

print(list(y))
data.boxplot(column='Overall',by='Potential')

plt.show()
data.melt=data.head()

melted=pd.melt(frame=data,id_vars='Name',value_vars=['Overall','Potential']) 

#id_vars is constant value

#overrall and potential count is value

melted

data.info()
#pivoting is restore of melt

melted.pivot(index = 'Name', columns = 'variable', values = 'value')
#concatenating data

data1=data.head(5) #first 5

data2=data.tail(5) #last 5



conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

#axis ->virticule

#ignore_index -> for new index

conc_data_row
data1=data['Overall'].head(10)

data2=data['Potential'].head(10)

conc_data_col=pd.concat([data1,data2],axis=1)

#axix=1 for horizontal
conc_data_col
#value counts

data.dtypes
#for convert object(str) to categorical and it to float

data['Overall']= data['Overall'].astype('category')

data.info()
data['Club'].value_counts(dropna=False)

#dropna? false for to see the null value
data1 = data

data1['Club'].dropna(inplace = True)

data1
assert data['Overall'].notnull().all() # returns nothing because we don't have nan values

#if assert is true return nothing 
data['Club'].fillna('empty',inplace=True) #To fill in missing data
data['Club']
Dessert =["Tiramisu","Cheescake"]

specification=["Coffee","Cheese"]

list_label=["Dessert","specification"] #add dessert and spesicification

list_col=[Dessert,specification] #add column



zipped=list(zip(list_label,list_col)) #do zip

data_dict=dict(zipped) #to make zip file dictionary

df=pd.DataFrame(data_dict) #to make dictionary DataFrame

df
#Add new column

df["price"]=["11$","10$"]
df["Time"]=40   #same Time values
df
data.plot(kind="hist",y="Overall", bins = 50,range = (0,150), normed = True)

#range for y axis

    
data.info()


data.plot(kind= "hist",y="Overall", bins = 50,range = (0,150),normed=True,cumulative=True)

#cumuative is addand show

data2=data.head(2)

time_list=["1996.03.12","1996.04.12"]

datetime_object = pd.to_datetime(time_list)

data2["date"]=datetime_object

data2=data.set_index("date")

data2
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
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

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean() #mean year
data2.resample("M").mean() #mean mounth
data2.resample("M").first().interpolate("linear") #we can interpolete 
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
data.info()
data.columns
data.head()


data=pd.read_csv('../input/data.csv')

data= data.set_index("Unnamed: 0")

data.head()
data.Name[0] #0. index element
data["Name"][0] #other way
data.loc[0,["Name"]] #The intersection of 0.index and Name Column
data [["Name","Age"]] #show age and Name column
print(type(data["Name"])) #Series

print(type(data[["Name"]])) #Data Frames
data.loc[1:4,"Name":"Nationality"] # between 1 and 4 index and values between age and name columns
data.loc[0:4,["Name"]]
data.loc[10:1: -1,"Name":"Nationality"]
data.loc[1:10,"Age":] #between 1 and 10 index and between age and end of series
deger=data.Age>42

data[deger]
filter_first=data.Age>35

filter_second=data.Overall>65

data[filter_first & filter_second] #gives the intersection of two filters
data.Age[data.Overall<50] #nested filters Overall<50
def div(n):

    return n*2



data.Age.apply(div) #new age = age*2
data.Age.apply(lambda k:k+10) #other function way
data["Overall_Age"]=data.Overall+data.Age

data.head()
print(data.index.name)
data.index.name = "#"

print(data.index.name) #change index_name
data.index = range(10,90,1)
data1 = data.set_index(["Name","Age"]) 
data1.head()
dic = {"treatment":["A","B","C","D"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df.pivot(index="treatment",columns="gender",values="response")
df1=df.set_index(["treatment","gender"]) #two index

df1
df1.unstack(level=0) #first index
df1.unstack(level=1) #second index
df2=df1.swaplevel(0,1) #change  first and second index

df2
pd.melt(df,id_vars='treatment',value_vars=["Age","Response"])
#Categorical and groupby

df.groupby("treatment").mean() #treatment mean

# # mean is aggregation / reduction method

    
df.groupby("treatment").age.max() #max age value for treatment
df.groupby("treatment")[["age","response"]].min() 