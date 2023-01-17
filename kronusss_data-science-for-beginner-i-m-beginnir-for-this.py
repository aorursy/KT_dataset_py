

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


data = pd.read_csv("../input/data.csv")
data.info()
data.corr()
f,ax = plt.subplots(figsize=(32, 32))
sns.heatmap(data.corr(), annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()
data.head(20)
data.fractal_dimension_mean.plot(kind="line", color = "r", label="fractal_dimension_mean", linewidth= 1, alpha = 0.5, grid = True,linestyle = "-")
data.radius_mean.plot(color ="b", label = "radius_mean",linewidth=1,alpha=0.5,grid=True,linestyle=":")
plt.legend()
plt.xlabel("fractal_dimension_mean")
plt.ylabel("radius_mean")
plt.title("Line Plot")
plt.show()
#Scatter Plot
#x = smoothness_se
#y = radius_se
data.plot(kind="scatter", x = "smoothness_se", y = "radius_se",alpha=0.7,color="green")
plt.xlabel("smoothness_se")
plt.ylabel("radius_se")
plt.title("Smoothness Radius Scatter Plot")

#Histogram
#bins = number of bar in figure
data.texture_worst.plot(kind ="hist",bins = 40,figsize=(13,13))
plt.show()
                        
#clf() = cleans it up again you can start a fresh
data.texture_worst.plot(kind="hist",bins= 30)
plt.clf()
#create dictionary and look its keys and values

dictionary = {'spain':'madrid','england':'london'}
print(dictionary.keys())
print(dictionary.values())
#Keys have to be imutable object like string, boolean, float, integer or tubles
#list is not immutable
#keys are unique
dictionary['spain'] = "barcelona" #update existing entry
print(dictionary)
dictionary['german'] = "berlin" # add new entry
print(dictionary)
del dictionary['spain']     # remove entry with key 'spain'
print(dictionary)
print('german' in dictionary)  #check include or not
dictionary.clear()            #remove all entries in dict
print(dictionary)
#In order to run all code you need to take comment this line
#del dictionary      # delete entire dictionary
print(dictionary)  #it gives error because dictionary is deleted

data = pd.read_csv('../input/data.csv')
series = data['perimeter_mean']  #data['perimeter_mean'] = series
print(type(series))
data_frame = data[['perimeter_mean']] #data[['perimeter_mean']] = data frame
print(type(data_frame))
# Comparisonoperator

print(3>2)
print(3!=2)
#boolean operators
print(True and False)
print( not False)
print(True or False)

# 1 - Filtering Pandas data frame
x = data['perimeter_mean']>150 # there are only 13 results who higher perimeter_mean value than 150
data[x]
#2 - Filtering pandas with logical_and
#there are only 5 results who have higher perimeter_mean value than 150 and higher radius_mean value than 25

data[np.logical_and(data['perimeter_mean']>150,data['radius_mean']>25)]
# This is also same with previous code line. Therefore wecanalso use '&' for filtering
data[(data['perimeter_mean']> 150) & (data['radius_mean']> 25)]
#stay in loop if condition(i is not equal 5) is true
i=0
while i != 5:
    print('i is: ', i)
    i +=1
print(i,' is equal to 5')
#stay in loop if condition(i is not equal 5)is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ', i)
print('')

#Enumerate indexand value of list
#index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')

#for dictionaries
# we can use for loop to achive key and value of dictionary. We learn key and value at dictionary part.

dictionary = {'spain':'madrid','france':'paris','turkey':'ankara'}
for key,value in dictionary.items():
    print(key, " : ",value)
print('')

#for pandas we can achieve index and value
for index,value in data[['radius_mean']][0:3].iterrows():
    print(index," : ",value)
#example of what learn above
def tuble_ex():
    """return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
#guess print what

x=2
def f():
    x = 3
    y=2*x  # there is no local scope
    return y
print(x) # x= 2 global scope
print(f())   #it uses global scope x=3 local scope
#First local scope searched , if two of them cannot be found lastly duilt in scope searched.
# how can we learn what is luilt in scope
import builtins
dir(builtins)
def square():
    """return square of value"""
    def add():
        """add two local variable"""
        x=2
        y=3
        z = x + y
        return z
    return add()**2
print(square())
# default arguments

def f (a, b = 1, c = 2):
    y = a + b + c
    return y

print(f(5))# what if we want to change default arguments
print(f(5,4,3))
# flexible arguments *arg
def f(*args):
    for i in args:
        print(i)
        
f(1)
print("")
print(1,2,3,4,5,6,7,8)

#flexible arguments **kwargs that is dictionary

def f(**kwargs):
    """print key and value of dictionary"""
    for key,value in kwargs.items():
        print(key," ",value)
        
f(country = "spain", capital = "madrid", population = 123765)
# lambda function
square = lambda x: x**3 # where x is name of argument
print(square(5))
tot = lambda x,y,z : x*y/z # whre x,y,z are names of arguments
print(tot(4,5,8))
number_list = [1,2,3,4,5,6,7]
y = map(lambda x:x+x**2,number_list)
print(type(y))
print(list(y))
#iteration example
name = "Semih"
it = iter(name)
print(next(it)) # print next iteration
print(*it) # print remaining iteration
#zip example
list1 = [1,2,3,4,5,6,7]
list2 = [8,9,10,11,12,13,14]
z = zip(list1,list2)
print(type(z))
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(un_zip)
print(type(un_list1))
num1 = [1,2,3]
num2 = [i+1 for i in num1]
print(num2)
# Conditionals on iterable
num1 = [5,10,20]
num2 = [i**2 if i == 20 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
#lets return our csv and make on more list comprehension example
sum_radius_mean = sum(data.radius_mean)/len(data.radius_mean)
print(sum_radius_mean)
data["radius_mean_level"] = ["high" if i > sum_radius_mean else "low" for i in data.radius_mean]
data.loc[:20,["radius_mean_level","radius_mean"]] 
data = pd.read_csv('../input/data.csv')
data.head() # head shows first 5 rows
data.tail() # tail shows last 5 rows
data.columns # columns gives column names of features
data.shape #shape give number of rows and columns in a tuble
data.info() # info gives data type like dataframe, number of sample or row, number offeature or column, feature types and memory usage
print(data.diagnosis.value_counts(dropna=False)) # if there are nan values that also be counted
#as it can be seen below there are 357 B feature and 212 M feature

data.describe()
#For example max smoothness_mean is 0.1634 and average of texture_mean is 19.289649
#Black line at top is max
#Blue line at top is 75%
#Red line is median(50%)
#Blue line is bottom is 25%
#Black line at bottom is min
#there are no outliers
data.boxplot(column='perimeter_worst',by ='diagnosis')

#Firstly I create new data to explain melt more easily.
data_new = data.head()  # I only take 5 rows into new data
data_new
#lets melt
#id_vars = what we do not wish to melt
#value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars='id',value_vars=['compactness_mean','radius_se'])
melted
#Index is name
# I want to make that columns are variable
#finally values in columns are value
melted.pivot(index='id',columns='variable',values='value')
#Firstly lets create 2 data frame
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis=0, ignore_index=True) # axis=0 : adds dataframes in row
conc_data_row
data1 = data.radius_mean.head()
data2 = data['area_mean'].head()
conc_data_col = pd.concat([data1,data2],axis=1) # axis = 1 : adds dataframes in column
conc_data_col
data.dtypes

#lets convert object(str) to categorical an int to float
data.diagnosis = data['diagnosis'].astype('category')
data['symmetry_worst'] = data.symmetry_worst.astype('int')
data.dtypes
# Lets look at does this data have nan value
#As you can see there are 600 entries but we do not have nan value
data.info()
data.diagnosis.value_counts(dropna=False)
#lets drop nan values
#for example area_worst column have nan values
#data.area_worst.dropna(inplace = True) #incplace = True means we do not assing it to new variable.Changes automatically assigned to do
#lets check with assert statement
#Assert statement:
assert 1==1 #return nothing because it is true
assert 2==3 # return error because it is false
assert data.diagnosis.notnull().all() # returns nothing because we do not have nan values
#data.diagnosis.fillna('empty',inplace=True) # returns error because we do not have nan values
# With assert statement we can check a lot of thing.For Example
#assert data.columns[2] == 'Name'
#assert data.area_se.dtypes == np.int
#data frame from dictionary
country = ["Germany","France","Turkey"]
population = ["15","17","43"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
#Add new columns
df["capital"] = ["Berlin","Paris","Ankara"]
df
#Broadcasting
df["income"] = 0 #broadcasting enrite column
df
#plotting all data
data1 = data.loc[:,["texture_se","compactness_se","symmetry_se"]]
data1.plot()
plt.show() # it is confusing
#subplot
data1.plot(subplots= True)
plt.show()
#scatter plot
data1.plot(kind="scatter",x='texture_se',y='symmetry_se')
plt.show()
#histogram plot
data1.plot(kind='hist',y='texture_se',bins=50,range=(0,300),normed= True) # do not work normed because it is already normed
#histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind='hist',y = 'texture_se',bins=50,range=(0,500),ax = axes[0])
data1.plot(kind='hist',y ='texture_se',bins=50,range=(0,500),ax=axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe()
time_list = ['1992-03-08','1992-06-15']
print(type(time_list[1])) #as you can see date is string
#however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
#In order to practice lets take head of data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2['date'] = datetime_object
data2 = data2.set_index("date")
data2
#Now we can select according to our date index
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
#we will use data2 that we create at previous part
data2.resample("A").mean()
#lets resample with month
data2.resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months
# In real life(data is real. Not created from us like data2) we can solve this problem with interpolete
#we can interpolete from first value
data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()
data2.resample("M").mean().interpolate("linear")
#read data

data = pd.read_csv("../input/data.csv")
data = data.set_index('id')
data.head()
#indexing using square brackets
data.radius_mean[842302] #841302 we use because we setted index id
data["radius_mean"][842302]
data.loc[842302,["radius_mean"]]
#selecting only some columns
data[["area_mean","compactness_mean"]]
#Difference betwenn selecting columns: series and dataframes
print(type(data["compactness_mean"]))
print(type(data[["compactness_mean"]]))
#Slicing and indexinfg series

data.loc[:846226,"compactness_mean":"concave points_mean"]
#Reverse slicing

data.loc[846226:843786:-1,"compactness_mean":"concave points_mean"]
#From something to end

data.loc[:846226,"compactness_worst":]
#Creating boolean series

boolean = data.concavity_worst > 1
data[boolean]
#Combining filters
first_filter = data.concavity_worst > 1
second_filter = data.area_worst > 500
data[first_filter & second_filter]
#Filtering column based others
data.area_worst[data.concavity_worst > 1]
# Plain python functions

def div(n):
    return n/2
data.radius_mean.apply(div)
#or we can use lambda function
data.radius_mean.apply(lambda n: n/2)
#Defining column using other columns
data["total_radius_mean_and_radius_worst"] = data.radius_mean+data.radius_worst
data.head()
# our index name is this:
print(data.index.name)
#lets change it
data.index.name =  "index_name"
data.head()
#Overwrite index
#if we want to modify index we need to change all of them
data.head()
#first copy of our data to data3 then change index
data3 = data.copy()
#lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,669,1)
data3.head()
#lets read data frame one more time to start from beginning
data = pd.read_csv("../input/data.csv")
data.head()
#setting index : perimeter_mean is outer diagnosis is inner index
data1 = data.set_index(["perimeter_mean","diagnosis"])
data1.head(50)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
#pivoting
df.pivot(index="treatment",columns="gender",values="response")
df1 = df.set_index(["treatment","gender"])
df1
#lets unstack it
#level determines indexing
df1.unstack(level=0)
df1.unstack(level=1)
#change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2
df
#df.pivot(index="treatment",columns="gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
#We will use df
df
#according to treatment take means of other features
df.groupby("treatment").mean() #mean is aggregation / reduction method
#there are other methods like sum,std, max or min
#we can only choose one of the feature
df.groupby("treatment").age.max()
#or we can choose multiple features
df.groupby("treatment")[["age","response"]].min()
df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
#df["gender"] = df["gender"].astype("category")
#df["treatment"] = df["treatment"].astype("category")
#df.info()