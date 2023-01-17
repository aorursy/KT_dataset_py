# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.info()
data.head(30)
data.columns
data.corr() #gives the relationship between numerical data
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show  # a kind of relation map, which shows the most related data in light colors 
#Annual Income and Age plot of customers
data.Age.plot(kind = 'line', color = 'r',label = 'Age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data['Annual Income (k$)'].plot(kind = 'line', color = 'g',label = 'Annual Income',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.title('Line Plot')            # title of the plot
plt.show()
data['Annual Income (k$)'].plot(kind = 'hist',bins = 50 ,figsize = (11,11))
plt.show()  #histagram diagram of Annual Income of customers
data.head(10)
dict1=data.iloc[1:3,1:3] #selecting from 1 to 3rd rows and columns 1 to 2
dictionary = dict1.to_dict("list")
print(dictionary)  #  creating dictionary from dict1
series = data['Age']>20       # creating series from Age column
print(type(series))
print(series)


x=data[np.logical_and(data['Age']>35, data['Annual Income (k$)']>100)]
print(x)  #filtering the customers elder than 35 and annaul income bigger then 100K$
lis=data['Age'].tolist()
print(type(lis))
print(lis)       #creating list 
count=0   #sum the values in list
for i in lis:
    count=count+i
print(count)
max_value = 1
for i in lis:
    if i > max_value: 
        max_value = i   
print(max_value)
#Tupple examples: tupples are very similar to lists, the only difference is tupples cannot be changed or updated
dataset=data["Age"].iloc[0:5]
def tuple1(x): 
    return [tuple(x) for x in dataset.to_numpy()]
tuple1()  #####bakılacakkkk
#Scope: global value
x=5                # x is define globallay
def f():           #No local value is identified
    y = (x**2)-(x)       
    return y*3
print(f())         #result will be calculated by the global value of x

def g(): 
    x=3            #x is locally identified
    y=x**2
    return y
print(g())  


#Nested Function
dataset=data["Age"].iloc[0:5].tolist()  
def multiply():
    """ return multiply of value """
    def add():
        b=sum(dataset)  #sum of the values in dataset
        return b
    return add()*2
print(multiply())
print(dataset)
#Default Arguments:
dataset=data["Age"].iloc[0:5].tolist()
def f(a,b=5,c=4,d=1):        #default values are: b,c,d
    y=(a*b)+c-d
    return y
print(f(dataset[0]))         #assigning the 0th index of dataset to 'a'
#Flexible Arguments
dataset=data["Age"].iloc[0:5].tolist()
def f(*args):
    for i in args:
        print(i)
print(f(dataset))   #print the values in the list dataset 




def f(**kwargs):  #kwargs used for dictionaries
    """ print key and value of dictionary"""
    for key, value in kwargs.items():               
        print(key, " ", value)
print(f(Gender= ['Male', 'Female'], Age= [21, 20]))
# Lambda function
a=data["Age"].iloc[0:5].tolist()
square = lambda x: x**2
b= [square(i) for i in a]

print(b)

# Map Funtion
a=data["Age"].iloc[0:5].tolist()
b= map(lambda x: x**2,a) #map function works as iterator in a list

print(list(b))  
a=data["Age"].iloc[0:5].tolist()
b=data["Age"].iloc[5:10].tolist()
print('a:',a)
print('b:',b)
z=list(zip(a,b))  #zip used to join the lists
print('a+b=z:',z)    
unzip=zip(*z)
a,b=list(unzip)
print(a)
print(b)
data.info() #gives general info about the dataframe
data.tail() #brings the last 5 row
data.head() #bring the first 5 rows
data.shape # gives the #of columns and rows

print(data['Gender'].value_counts(dropna =False)) #gives the # of the categories in Gender column including the non-values
data.describe()
data.columns
data.boxplot(column='Annual Income (k$)',by = 'Gender') # drawa boxplot for annual income according to gender
subdata=data.head(10) #takes the first 10 row of the data
print(subdata)
meltsubdata=pd.melt(frame=subdata,id_vars = 'CustomerID', value_vars= ['Age','Gender'])
meltsubdata
meltsubdata.pivot(index = 'CustomerID', columns = 'variable',values='value')
#create 2 different dataframe and then we will concatenate them vertically
data1 = data.head(10)
data2= data.tail(10)
concatdata = pd.concat([data1,data2],axis =0,ignore_index =True) 
print(concatdata) 
#we will take 2 different columns from data and then concatenate them horizontally 
data1 = data['Gender'].head(10)  
data2 = data['Age'].head(10)
concatdata2 = pd.concat([data1,data2],axis =1) 
concatdata2
data.dtypes  #gives us the columns by their type
# lets convert object(str) to categorical and int to float.
data['Gender'] = data['Gender'].astype('category')
data['Annual Income (k$)'] = data['Annual Income (k$)'].astype('float')
data.dtypes
#what do we do with missing data, first lets check the data
data.info()
#In out data there is no missing value but if we had,
#we would fill it with fillna() or drop them or fill the with the mean
#the can count the non values
data['Gender'].value_counts(dropna =False) #counts the different values

data["Gender"].isnull().sum().sum() #counts only non values
#here you count the different values of Age column
data['Age'].nunique() 
#gives us 51 different values

#assert check the statement you write and gives true/false
assert  data['Gender'].notnull().all() #returns nothing because we dont have any nan values
data.head()
# we will create a dataframe from dictionary according to our data
country = "Turkey"
city = ["Istanbul","Kocaeli","Adapazarı","Bursa","Çanakkale"]
list_label = ["country","city"]
list_column = [country,city]
ziplist = list(zip(list_label,list_column))
dictionary = dict(ziplist)
dataf = pd.DataFrame(dictionary)
dataf
dataf["Age Group"]=['young','young','young','young','young'] #creating a new data
dataf
#dataf["Age Group"]="young" #better way of giving the same values to the entire column

data.columns
# Plotting first 100 data 
datag = data.loc[0:100,["Annual Income (k$)","Spending Score (1-100)"]]
datag.plot()

datag.plot(subplots=True)#you can see the 2 columns in separate graphs
plt.show()  
# scatter plot  
datag.plot(kind = "scatter",x="Annual Income (k$)",y = "Spending Score (1-100)")
plt.show()

fig,axes=plt.subplots(nrows=2,ncols=2) #we will place the graps,2 rows, 2 columns

datag.plot(kind = "hist",y = "Annual Income (k$)",bins = 50,range= (0,80),ax=axes[0,0])

datag.plot(kind = "hist",y = "Annual Income (k$)",bins = 50,range= (0,80),ax=axes[0,1],cumulative = True)

datag.plot(kind = "hist",y = "Spending Score (1-100)",bins = 50,range= (0,100),ax=axes[1,0])

datag.plot(kind = "hist",y = "Spending Score (1-100)",bins = 50,range= (0,100),ax=axes[1,1],cumulative = True)

plt.savefig('graph.png')
plt.show
time_list = ["2020-01-01","2020-01-15","2020-04-01","2020-02-15","2020-06-01"] #we crate a list of date
datetime_object=pd.to_datetime(time_list) #convert teh string to dates
data1=data.head() #we create a new dateframe from first 5 rows
data1["Date"]=datetime_object #add the dates to the new dataframe
data1=data1.set_index("Date")
data1
print(data1.loc['2020-01-01']) #filtering according to date
print(data1.loc['2020-01-01':'2020-02-15'])
data1.resample("A").mean() #we calculate the mean of the data by year

data1.resample("M").mean()
data1.resample("M").first().interpolate("linear") #we fill the missing months
data.columns
data=pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data=data.set_index('CustomerID') # we set the customerid as a index
data.head()
data['Gender'][3] #will give us the Gender column 3rd row
data.Gender[3] #gives the same as previous
data.info()
data.loc[1:5,["Gender"]] #give us first 5 row of the Gender column
data[["Gender","Age","Annual Income (k$)"]] #calling the columns
print(data[["Age"]])  #dataframe:two dimension, dataframe has a column name/header
print(data["Age"]) #series :one dimension,dont have any column name
data.loc[1:15,"Gender":"Annual Income (k$)"] #gives rows 1 t0 15 of columns from Gender to Annual Income 
data.loc[15:1:-1,"Gender":"Annual Income (k$)"]  #same as previous but reverse
# From something to end ,first 10 rows
data.loc[1:10,"Gender":]
#Filtereing Data
boolean = data.Age >67
data[boolean]
#For 2 or more conditions
filter1= data.Age >67
filter2= data.Gender =='Male'
filter3=data['Annual Income (k$)']>45
data[filter1&filter2&filter3]
#Filtering data column according to another condition
data.Age[data.Gender=='Female']
data['Annual Income (k$)'].loc[data.Age>68]
def multply(m):  #we define a fuction
    return m*1000
data['Annual Income (k$)'].apply(multply) #we apply the function to column Annual Income

data['Annual Income (k$)'].apply(lambda n : n*1000)
# Defining column using other columns
data["Unitscore"] = data['Spending Score (1-100)'] /data['Annual Income (k$)']
data.head()
print(data.index.name) #finding the index column
data3=data.copy()
data3.index=range(100,300,1) #index will:100 to 300, increasing 1 by 1
data3.head()
# hierarchical indexing
dataX=data.set_index(["Gender","Age"])
dataX.head()


#dictionary
data.head()
predict=data.iloc[0:5,0:4]        #we create a dictionary from data
dictionary=predict.to_dict("list")
df=pd.DataFrame(dictionary)
df
#pivoting the dataframe
df.pivot(index="Gender",columns ="CustomerID",values="Annual Income (k$)")
# multi indexing
df1=df.set_index(["Gender","CustomerID"])
df1
df1.unstack(level=0)  #level=0 will unstack the Gender index
df1.unstack(level=1) #level=0 will unstack the CustomerID index
#reverse of pivot
#df.pivot(index="Gender",columns ="CustomerID",values="Annual Income (k$)")
pd.melt(df,id_vars="Gender",value_vars=["CustomerID","Annual Income (k$)"])
df

df.groupby("Gender").mean() #we gruop by gender and take the mean of numeric columns
#another example of group by this time will take the max
df.groupby("Gender").max()
#group by gender and min of age and annual income columns
df.groupby("Gender")[["Age","Annual Income (k$)"]].min()
