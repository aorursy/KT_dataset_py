# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization tools

import matplotlib.pyplot as plt #visualization tools





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/diamonds.csv")
data.head() #Retrieves the first 5 data in the diamonds.csv
#How to use comparison operators

print(0 > 1)  #False because 0 not greater than 1

print(0 >= 1) #False because 0 is not equal to 1 or 0 is not greater than 1

print(0 == 1) #False because 0 is not equal to 1

print(0 != 1) #True != = is not equal   and 0 is not equal to one therefore result True

print(0 <= 1) #True because 1 greater than 0



#how to use boolean operators



print(True and True) #True because Everything must be true if a 'and' operator is used If one of the values is false return False

print(True or False) #True because Not everything must be true if a one of the value is a true return True



# We use 'not' operators in list section
a = 5

b = 2

# whichever is greater, print it



if a > b: #is true because 5 greater than 2

    

    print(a)

#we say a greater than b what if it wasn't

a = 3

b = 6

if a > b: #is false because 3 not greater than 6

    

    print(a)

else: #If 'if' is not true programs enters else statement

    print(b)
#what if the two variables are equal



a = 3

b = 3



if a > b: #is false because are equal

    print(a)

    

elif a == b: #is true

    print("Are Equals",a,b)

    

else:#programs not enters this statement because first enter if second enter elif 

    print(b)
capital = {"England":"London" ,"Germany":"Berlin"} #We create dictionary 

#keys (England,Germany)

#values (London,Berlin)

print(capital.keys()) #Get keys

print(capital.values()) #Get values
capital["England"] = "Istanbul" #update existing entry

print(capital)
#Istanbul is not the capital of England



#We should update keys

capital["Turkey"] = capital.pop("England")

        #new_key                #old_key

print(capital)
#How to add new entry



capital["Brazil"] = "Brasília"

print(capital)

if ("London" in capital.values()): #false because London not in capital dictionary but if true print(capital["England"]) return KeyError

    print(capital["England"])

elif("Istanbul" in capital.values()):

    print(capital["Turkey"])
#capital["England"] return error
print(capital)

del capital["Brazil"] #remove entry

print(capital)
v = capital.get("Germany")

print(a)
capital.clear() #clear all dictionary
print(capital)
del capital #delete entire dictionary 

#there is no dictionary named 'capital' because it deleted
list1 = [1,2,3,4] #iterable object





for i in list1:

    print(i) #iterator

dictionary = {"LeBron James":"Los Angeles Lakers","Stephen Curry":"Golden State Warriors"}



for k,v in dictionary.items(): #we use two iterators because the dictionary consists of two values

    

    print("Key:{} \nValue:{}".format(k,v))

    



     

    
#def functionname( parameters    (optionals) ):

   #"function_docstring" (optionals)

   #function_suite

   #return [expression]  (optionals)
x = 5

def function1():

    """

    This function takes x's square

    """

    return x ** 2 #get square

print(function1())
a = 5

c = 4



def function2(x,y): #Function Parameters

    """This function x  plus y"""

    return x + y



print(function2(a,c))
b = [1,2,3,4] #list



def function(x):

    

    print("incoming values",x)

    outlist = [i ** 2 for i in x] #list comprehension

    return print("Outgoing values", outlist)



function(b)
def sayhello(helloword):

    

    print(helloword)

    

sayhello("Merhaba")

    
#sayhello() #return error
def function(x,y = 1):

    

    return x * y



print(function(12)) #we give one parameters but y parameter define default = 2 therefore no errors occurred
#what happens we give two parameter

print(function(12,2)) #y value change to 2
def function(*args):

    print("İnput parameter",args)

    

    for i in args:

        print("Output parameter",i + 10)

        

    

function(1,2,3,4,5)
def f(**kwargs): #**kwargs it's dictionary

    """ print key and value of dictionary"""

    for key, value in kwargs.items():

        print(key, " ", value)

f(country = 'Germany', capital = 'Berlin', population = 123456)
list1 = [1,2,3,4]



list2 = [i+1 for i in list1] #i+1 = expression list1 = oldlist list2 = newlist



print(list2)
list1 = [5,11,12,13]



list2 = [i**2 for i in list1 if i > 10] #i**2 = expression if i > 10 condition



print(list2)
list1 = [5,6,7,19,18]



list2 = [i**2  for i in list1 if i < 20 if i > 10]

                              #if block #elif

print(list2)
list1 = [5,6,7,19,18,-2,-5]

                #İf condition       #Elif Expression#Elif Condition    Else Expression

list2 =  [i if i < 18 and 0 < i       else i+2    if i > 17          else i**2                         for i in list1]

            #*****if block****       #*****Elif Block*****          #Else Block



#if condition if i < 18 and 0 < i

#if expression i



#elif condition i > 17

#elif expression i+2



#else expression i**2 





print(list2)
# Example 1



dictionary = {"USA":327,"Russia":144,"China":1386}



maxpopulation = int()

maxpopulationcountry = str()

for iterator,j in dictionary.items():

    

    if j > maxpopulation:

        maxpopulation = j

        maxpopulationcountry = iterator

print("Country {}\nPopulation(Million): {}".format(maxpopulationcountry,maxpopulation))



#But this too long way



#let's learn some trick for take dictionary higher value

#first we need another library



import operator

myTuple = max(dictionary.items() ,key = operator.itemgetter(1))[:] #this line return Max Population and Country ('China',1386)

print(myTuple[0],myTuple[1]) 





# Example 2



evennumbers = [i for i in range(0,100) if i % 2 == 0]

print(evennumbers)

# Example 3

def factorial(x):   

    factorial = 1

    

    for i in range(2,x+1):

        factorial *= i

    return factorial





print(factorial(5))
array1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #1*15 vector

print(array1.shape) #learn array shape

array1 = array1.reshape(5,3) #array reshape 

print(array1.shape)

print(array1)



print(array1.ndim)#learn array dimension

print(array1.size) #array length
array3 = np.empty((2,2)) #Creates an array with a value less than 0

#this 'array.empty' the equivalent null in python

print("Less than 0 array\n",array3)

print("---------------------------------------------------------------------------------")





array2 = np.zeros((2,2))# Creates  array of zeros    (3,4) shape (x,x) 

#list.append()  it consumes too much memory. therefore we use this method

print(array2)



#how to update array variables

array2[0][0] = 1      #columns 1 line 1

array2[1][1] = 4      #columns 2 line 2

print("Update Version\n",array2)
rangearray = np.arange(10,50,5)#10 to 50 step 5 

#10 inclusive   #50 not included

print(rangearray)
linespace = np.linspace(10,50,20000) #Compresses 20000 numbers between 10 and 50

print(linespace)
array1 = np.array([1,2,3])

array2 = np.array([4,5,6])



print(array1+array2) #return new array. collects columns

print(array1.sum())

print(np.square(array1)) # take square
print(np.add(array1,array2)) #other way collects columns
array1 = ([1,2,3,4,5,6,7])



print(array1[0]) #get 0 index

print(array1[0:4]) #Retrieve data from 0 to 4(not included)



reverse_array = array1[::-1]

print(reverse_array) #reverse array
array1 = np.array([1,2,3,4,5,6,7,8,9,10]) 

array1 = array1.reshape(2,5) #two dimensional array

print(array1)

print(array1[1][1]) #column 2 but index number = 1 ,  line 1 

print(array1[0][4])  #column 1 but index number = 0 ,  line 4
print(array1[0][4],array1[1][4]) #It's hard to get the last column with this path.



print(array1[:,-1]) #Therefore we use this way because it's very simple and we may not always know the size of the array





#We can do it the same way take last line. we just need to replace ':' and '-1'

print(array1)

print(array1[-1,:]) #last line
array1 = np.array([1,2,3])

array2 = np.array([4,5,6])



array3 = np.hstack((array1,array2))  #horizontal stack

print(array3)

array3 = np.vstack((array1,array2)) #vertical stack

print(array3)
array1 = np.random.randint(1,100,10) #create random 10 values between 1 and 100

print(array1)
array1 = np.random.rand(5) #create random 5 values between 0 and 1 

print(array1)



array1 = np.random.randn(5) #create Gaussian Distribution values

print(array1)
array1 = np.arange(1,101)

# Take even numbers

filter1 = array1 % 2 == 0

array1[filter1]



#another way

array1[array1 % 2 == 0]



#trick

#array2 = np.arange(0,101,2)

#print(array2)
array1 = np.array([10,20,30,40,50,60,70,71,72,73,74,80,90,100])



#Let's take the numbers between 70 and 80

array1 = array1[(array1 >= 70) & (array1 <= 80 )]

print(array1)
array1 = np.array([1,2,3,4,5,6,7,8,9])



#how to learn max and its index

print("Max:{} Index{}".format(array1.max(),array1.argmax()))



#same way learn min

print("Min:{} Index:{}".format(array1.min(),array1.argmin()))



#collects array1

print(array1.sum())



#Unit Matrix 4x4

print("\n",np.eye(4,4))



dictionary = {"NAME":["James","John","Robert","Michael","Matthew","Mary"],

              "AGE":[15,16,17,33,45,66],

              "Salary":[100,150,240,350,110,220]}





dataframe = pd.DataFrame(dictionary)
dataframe.info()
dataframe.dtypes #Object = string 
print(dataframe.columns)



#we get any column name that way

print(dataframe.columns[0])
print(dataframe.head(2)) #Retrieves the first n(default = 5) data in the dataframe(dictionary)

print("-------------------------------------------------")

print(dataframe.tail(2)) #Retrieves last n(default = 5) data in the dataframe(dictionary)
#to retrieve information for a single column

print(dataframe.AGE)

#or

print(dataframe["NAME"])
print(dataframe.loc[0:3,"NAME"]) #get first four  line and "NAME" column



print(dataframe.loc[0:2,["NAME","AGE"]]) #get first three line  and "NAME" and "AGE " column
reverse = dataframe.loc[::-1,:] #inverts data

reverse
#what difference between iloc and loc



#loc needs column name but iloc index number

print(dataframe)

print(dataframe.iloc[0,1]) #get first line and second column value

dataframe
condition = dataframe.AGE > 30

dataframe[condition]
#or

dataframe[dataframe.AGE > 30]
#how to use two condition

condition1 = dataframe.AGE > 30

condition2 = dataframe.Salary > 300

dataframe[condition1 & condition2]
dataframe
print(dataframe.AGE.max())

print(dataframe.Salary.min())

print(dataframe.AGE.mean())

#How to get the name of the oldest man



dataframe[dataframe["AGE"].max() == dataframe["AGE"]]["NAME"]



#marry age = 66
#How to get the age of the lowest salary



dataframe[dataframe["Salary"].min() == dataframe["Salary"]]["AGE"]



#Name     Age  Salary

#Matthew   45  110
dataframe
#let's create a new column

#column name is Salary level

average_salary = dataframe.Salary.mean()

print(average_salary)

dataframe["salary_level"] = ["High"if each > average_salary else "Lower" for  each in dataframe.Salary]

dataframe
dataframe
dataframe.drop(["salary_level"] , axis = 1,inplace = True) #we destroy salary_level



#inplace = dataframe = dataframe.drop(["...."],...)

dataframe
dataframe = dataframe.drop(dataframe[dataframe.NAME == "James"].index) 

dataframe
df1 = pd.DataFrame(["Toyota","Honda","Nissan"],index = [0,1,2],columns=["Japan"])

df2 = pd.DataFrame(["BMW","Mercedes","Audi"], index = [0,1,2],columns=["German"])

print(df1)

print(df2)
dataconcat = pd.concat([df1,df2],axis = 1) 

dataconcat
data = pd.read_csv("../input/diamonds.csv") 
data.info()
data.head()
data.drop(["Unnamed: 0"],axis  = 1 , inplace = True)

data.head()

#İt's gone 
data.describe() #numeric 
data.loc[(data["x"] == 0) | (data["y"] == 0) | (data["z"] == 0)]

#Notice that instead of using '&' in the code above, '|' because if we used 'and (&)' it would show data where all are 0
#just use len()



len(data.loc[(data["x"] == 0) | (data["y"] == 0) | (data["z"] == 0)])
#drop with three condition

#we use NumPy Logical_and method



data = data[np.logical_and(np.logical_and((data["x"] != 0) , (data["y"] != 0))  , (data["z"] != 0))] 

data.info()
data.describe()
data.head()
data.groupby("cut").mean()
data.groupby("cut").mean()["price"] #just price and cut
data["cut"].value_counts(ascending=True)
data.head()
def seperator(clarity):

    

    if clarity in ["I1","SI2","SI1","VS2"]:

        

        return "Bad Clarity"

    

    elif clarity in ["VS1","VVS2","VVS1"]:

        

        return "Good Clarity"

    

    else:

        

        return "Best Clarity"    

    

data["Clarity_level"] = data["clarity"].apply(seperator)    

    
data.head()
data["Clarity_level"].value_counts()
def seperator2(color):

    

    if color in ["E","F","G","H"]:

        return "Good Color"

    elif color in ["H","I","J"]:

        return "Bad Color"

    else:

        return "Best Color"

    



data["Color_level"] = data.color.apply(seperator2)
data.head()
data["Color_level"].value_counts()
condition1  = data.Color_level == "Best Color"

condition2 = data.Clarity_level == "Best Clarity"



bestdiamonds = pd.DataFrame(data[condition1 & condition2])
len(bestdiamonds.index) #Only 73 diamonds have the best clarity and best color
print(int(bestdiamonds["price"].mean()))

print(int(data["price"].mean()))



#Look at the difference between these two.

arr = np.array([[10,20,np.nan],[5,np.nan,np.nan],[21,np.nan,10]]) #np.nan = NaN Values



df = pd.DataFrame(arr,index = ["İndex1","İndex2","İndex3"], columns = ["column1","column2","column3"])

df.head()
df.info()
#first solution

#drop them with dropna()

print(df.head())

print("--------------------\nAfter Solution\n--------------------")

print(df.dropna(axis = 1)) #if column have ONE NaN Values destroyed all column. but is not so healthy for our data

print(df.head())



print("--------------------\nAfter Solution\n--------------------")

print(df.dropna(axis = 0)) #Healthier than previous solution. Because we will not deal with data with 3 values.

#Line destroy meaning just one object destory. but the collumn destory meaning all data may be corrupted

#second solution

#fill missing value with fillna()

print(df.head())

print("--------------------\nAfter Solution\n--------------------")

print(df.fillna("Empty"))
#third solution

#fill missing values with test statistics like mean

#we write a function for this problem

print(df.head())

print("--------------------\nAfter Solution\n--------------------")



#we use lambda and list comprehension one line 

anonymousfunction = lambda dataframe: [dataframe[cNAME].fillna(value = dataframe[cNAME].fillna(0).mean(),inplace = True) for cNAME in dataframe.columns]

anonymousfunction(df)

print(df.head())

edaanalysıs = pd.read_csv("../input/diamonds.csv")

edaanalysıs.describe()



#example X,Y,Z min = 0 this is a outlier values
#bestdiamonds #diamonds have the best clarity and best color

#data.head #normal data





#                                    Define Figure 1(Normal Diamonds) 





fig, axes = plt.subplots(nrows=1, ncols=2 ,figsize=(15,6)) #we create figure; one rows, two collumn

figure1 = data.boxplot(ax=axes[0],column="price",by=["cut","Clarity_level"],vert=False,fontsize=13,grid=False) #left figure

figure1.set_xlabel("PRİCE")

figure1.set_title("Normal Diamonds")



#                                   Define Figure 1 (Normal Diamonds)



#                                     Define Figure 2 (Best Clarity and Color Diamonds)



figure2 = bestdiamonds.boxplot(ax=axes[1],column="price",by=["cut","Clarity_level"],vert=False,fontsize=13,grid=False) #right figure

figure2.set_xlabel("PRİCE") 

figure2.set_title("Best Diamonds")



#                                     Define Figure 2 (Best Clarity and Color Diamons)





#                                    Plot Settings





plt.tight_layout() #To prevent intertwining

plt.suptitle('Cut and Clarity Level Effect on Price') 



#                                   Plot Settings





plt.show()
#learn  which data types how much are in the dataset

data.dtypes
#we can object to category

#and we can int to float

data["Clarity_level"] = data["Clarity_level"].astype("category")

data["price"]= data["price"].astype("float")
#As you can see Clarity_level is converted from object to categorical

# And price converted from int to float

data.dtypes
data.head()
#Let's add time series to our data

data2 = data.head()





date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")





data2
data2.loc["1992-01-10":"1993-03-16"] #indexing
figure1 = data2.price.plot(kind="line")

figure1.set_xlabel("DATE")

figure1.set_ylabel("PRİCE")

figure1.set_title("diamond price chart by time")

plt.show()
data2.resample("A").mean()



# 'A' = year

# 'M' = month
data2.resample("M").mean()
#first way

data2.resample("M").mean().interpolate("linear")        #depending on the average



#second way



#data2.resample("M").first().interpolate("linear")       #depending on the first value
x = np.arange(1,6)

y = np.arange(2,11,2)

print(x)

print(y)
fig = plt.figure()



axes = fig.add_axes([0.1,0.1,0.8,0.7])

axes.plot(x,y,"red")

axes.set_xlabel("X Values")

axes.set_ylabel("Y Values")

plt.show()

fig , axes = plt.subplots(ncols=2,nrows=1,figsize = (10,5))   

axes[0].plot(x,y,"red",linewidth=5,marker="o",markersize=19,markerfacecolor="black",markeredgewidth=7,markeredgecolor="yellow")

                                                                                

axes[0].set_title("X vs Y Graph")

axes[0].set_xlabel("Y Values")

axes[0].set_ylabel("X Values")



#marker = "o" (add marker any type) 

#https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers can you see all  markers type





axes[1].plot(x**2,y**2,linewidth=5,linestyle="-.")

axes[1].set_title("X Square vs Y Square")

axes[1].set_xlabel("Y Square")

axes[1].set_ylabel("X Square")

plt.tight_layout()

data.head()
plt.scatter(data.price,data.carat,color="red")         #First X (price) , Second Y (carat) , and other properties

plt.xlabel("Price") #X Label

plt.ylabel("Carat") #Y Label

plt.title("scatter plot") #Figure Title

plt.show()
plt.scatter(data.carat,data.price,color="blue")         #First X (carat) , Second Y (price) , and other properties

plt.xlabel("Carat") #X Label

plt.ylabel("Price") #Y Label

plt.title("scatter plot") #Figure Title

plt.show()
data.carat.plot(kind="hist",bins=5,figsize=(10,10),color="red",label="frequency")

plt.legend()

plt.xlabel("CARAT")

plt.show()