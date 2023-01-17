# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Create list
num = [1,2,3,4,5,6]
num
#Let's see its attribute
print(dir(num))
#Add new element
num.append("foo") 
num
#Remove element
num.remove('foo') 
num
print("All elements : ", num)
print("First 3 elements : ", num[:3])
print("Except first 3 elements  : ", num[3:])
print("4 to 6 elements: ", num[3:6])
print("5 to 2 elements: ", num[1:-2])
#Creating Tuple
t1 = (6,"str",15,12,"foo",6)
t1
print("count(6)", t1.count(6)) # How many '6' does tuple have?
print("index('foo')", t1.index('foo')) # What's index of 'foo'
dic = {"alice" : 23, "bob" : 12, "steve": 45}
dic
#Get Steve value
dic["steve"]
#Get bob value
dic.get("bob")
print(dic.keys())
print(dic.values())
dic.pop("steve") # remove steve
#Another way to create a dic
x = dict([("Mad Men", "Drama"), ("The Office", "Comedy")])
x
#Create list 
names = ["alice","bob","eva", "steve","jane", "billy"]

#print each element of list
for name in names:
    print(name)
#print each element of string
for i in 'alice':
    print(i)
#split() function is  return a list of word strings. 
for i in "alice and bob".split():
    print(i)
#Lambda Function
foo = lambda x: x**3
foo(3)
#Anonymous function is likely to lambda but it takes more arguments.
numbers = [15,16,20,23,13,18,27]
results = map(foo, numbers)
print(results)
print(list(results))
num1 = [1,2,3,4,5,6,7]
num2 = [5,'foo',8, 1.0]
z = zip(num1,num2)
print(z)
print(list(z))

#Create a dictionary
dic1= { "Name":["alice","bob","clarke", "steve","eva", "jason"],
        "Ages":[15,16,20,23,13,18],
        "Salary":[100,232, 300, 50, 140,500] }

#Increase the age of everyone one year
print(dic1["Ages"])
dic1["Ages"] = [i + 1 for i in dic1["Ages"]] #Increase Ages 1 year
print(dic1["Ages"])

#Creating new column, if salary > avarage write 'high' else 'low'
avg = sum(dic1["Salary"]) / len(dic1["Salary"]) #Avarage
dic1["new_column"] = ["high" if salary > avg else "low" for salary in dic1["Salary"]]
dic1
#Change column name
dic1["Status"] = dic1.pop("new_column")
dic1

a = np.array([1,2,3,4,5,6,7,8,])
a
a = a.reshape(4,2)
a
print("shape : ", a.shape)
print("dimension : ", a.ndim)
print("data type : ", a.dtype.name)
print("size : ", a.size)
print("type", type(a))
#Create 3x4 matris and fill it with zeros
zeros = np.zeros((3,4))
zeros
np.ones((3,4))
#Create a array: range of elements 5 to 20, increase 2
c = np.arange(5,20,2)
c
#Create a array, put into the 20 numbers to between 1 and 10.
np.linspace(1,10,20)
#To sum arrays
a = np.array([1,2,3])
b = np.array([4,5,6])
a**2
np.sin(a)
c > 12
c[c>12]
#Element wise
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,6]])
a*b
#Dot product
a.dot(b.T) #b.T = b's tranpose
#Create random matrix 2x3
np.random.random((2,3))
print("a : ", a)
print("Sum : ", a.sum())
print("Max : ", a.max())
print("Min : ", a.min())
print("Sum rows : ", a.sum(axis=0))
print("Sum columns : ", a.sum(axis=1))
#Indexing and Slicing
a = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
print(a)
print("\n",a[0:2,1:4])
print("\n", a[-1:])
#SHAPE MANIPULATION
b = a.ravel()
b
b = b.reshape(5,3)
b
#ARRAY CONCAT
array1 = np.array([ [1,2], [3,4] ])
array2 = np.array([ [4,5], [6,7] ])
#VERTICAL
np.vstack((array1, array2))
#HORIZONTAL
np.hstack((array1, array2))
#Convert
l = [1,2,3,4]
a = np.array(l)
a
list(a)
#Copy
a = np.array(a)
b = a #Same address a and b
b
a[0] = 5
b
c = a.copy() # New address for c
c
a[0] = 1
c
#Creating a series
my_series = pd.Series([1,3,'string', 'f', 7.1, 10])
my_series
#Reaching values, index and rows
print("First 3 values : ", my_series[0:2].values)
print("Last 2 rows :")
my_series[-2:]
#Changing index names.
indexes = [3, "foo", 5, "float", 1, 10]
my_series.index = indexes
my_series
#Getting the value with index name.
my_series["foo"]
#A dictionary can be put into  to series.
dictionary = {"Alice":25, "Bob" : 15, "Clarke": 19}
new_series= pd.Series(dictionary)
new_series
#Comprehension : new_series > 18 is return true or false
new_series[new_series>18]
#One another way to create series.
indexes = ["str", "float", "int"]
new_series = pd.Series([20,30,40], index=indexes)
new_series
dic1= { "Name":["alice","bob","clarke", "steve","eva", "jason"],
        "Ages":[15,16,20,23,13,18],
        "Salary":[100,232, 300, 50, 140,500] }
#Creating a dataframe
dt= pd.DataFrame(dic1)
dt
for i in dt.index:
    print(i)
#Read Data from csv file
data = pd.read_csv("../input/Iris.csv")

data.info()
#Show columns names
data.columns
#Show first 7 rows
data.head(7)
#Show last 5 rows
#data.tail()
#If numbers are defined as a string, we can change them to numeric
data.SepalLengthCm = pd.to_numeric(data.SepalLengthCm, errors='coerce')
##If there is a null its values returns True
data.isnull().head()
#If there is a null its values returns False
data.notnull().head(5)
#It shows is there any nan values in the columns
data.notnull().all()
#It drops nan values
data.SepalLengthCm.dropna(inplace = True) 
#It fills nan values with 0
data.SepalLengthCm.fillna(0, inplace = True)
data.head()
#Get a column as a series
data["SepalLengthCm"].head()
#Get a column as a dataframe
data[["SepalLengthCm"]].head()
#Get columns
data[["SepalLengthCm", "PetalLengthCm"]].head()
#Set id column as a index.
data = data.set_index('Id')
data.head()
#Change labels index and columns
data.index.names = ['index']
data.head()
#Add one more index
data.set_index('Species', append=True).head()
#Change the index values
#data.index = range(100,len(data.index)+100)
#data.head()
#Changing column names
data = data.rename(columns={'SepalLengthCm':'SLCM','SepalWidthCm':'SWCM', 'PetalLengthCm':'PLCM', 'PetalWidthCm': 'PWCM'})
data.columns
#Indexing and Slicing
data.loc[2:4, ["PLCM"]]
data.loc[2:4, "PLCM"]
data.loc[1:5,"PLCM": ]
data.loc[1:3 , ["PLCM", "Species"]]
data.head()
#Applying lambda function SLCM column
data.SLCM = data.SLCM.apply(lambda n : n/2)
data.head()
# Defining column using other columns
data["total_lengthCM"] = data.SLCM + data.PLCM
data.head(3)
# Creating boolean series
boolean = data.SLCM > 3.5
data[boolean]
#Apply two filters
filter1 = data.SLCM > 3.5
filter2 = data.PLCM > 6.5
data[filter1 & filter2]
data.describe()
#CONCAT ROWS
data1 = data.head()
data2= data.tail()
rows_concat = pd.concat([data1,data2])
rows_concat
#CONCAT COLUMNS
data1 = data['SLCM'].head()
data2= data['PLCM'].head()
cols_concat = pd.concat([data1,data2], axis=1) 
cols_concat
dic = {"treatment":["A","B","B","A"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
#PIVOTING
df.pivot(index="gender", columns="treatment", values=["response", "age"])
df
pd.melt(df, id_vars="treatment", value_vars=["gender","age"])
df
df1 = df.set_index(["treatment","gender"])
df1
#UNSTACKING
# level determines indexes
df1.unstack(level=1)
df1.unstack(level=0)
# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2
data.head()
#Group by according to Species and calculate the mean
data.groupby('Species').mean()
#Groupby and sum
data.groupby('Species').sum()
#print setosa's summed values
data.groupby('Species').sum().loc["Iris-setosa"] #It returns series
data.groupby('Species').sum().loc[["Iris-setosa"]] #It returns dataframe
#print max SWCM, PLCM,PWCM values of versicolor
data.groupby('Species').max().loc[["Iris-versicolor"], "SWCM":"PWCM"]
#print PLCM,PWCM and total_lengthCM values of 1 and 2 rows
data.groupby('Species').max().iloc[[1,2] , -3:]
df
df.groupby("treatment")[["response", "age"]].min()
df.groupby(["treatment", "gender"]).mean()
dic1= { "Name":["alice","bob","clarke", "steve","eva", "jason"],
        "Ages":[25,36,40,44,33,28],
        "Salary":[100,232, 300, 50, 140,500] }
#Creating a dataframe
dt1= pd.DataFrame(dic1)
dt1
dic2= { "Name":["eva", "clarke", "bob", "steve", "jason", "alice"],
         "Experience" : [4, 2, 5, 10, 2, 3 ] }
#Creating a dataframe
dt2= pd.DataFrame(dic2)
dt2
#Merging 2 dataframes according to names
dt = pd.merge(dt1, dt2)
dt