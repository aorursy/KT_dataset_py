import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # import matplot library

import seaborn as sns # import seaborn library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.info()
df.head()
df.tail()
df.columns
df.shape
df.corr()
f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, linewidth=.5, fmt = ".1f", ax=ax)
df.chol.plot(kind="line", color="blue", label="Chol", linewidth=1, alpha=.5, grid=True, linestyle="-")

df.age.plot(kind="line", color="red", label="Age", linewidth=1, alpha=.7, grid=True, linestyle="-")

plt.legend(loc="upper right") # Legend position

plt.xlabel("Case ID")           # Label of X axis

plt.ylabel("Level")     # Label of Y axis

plt.title("Cholesterol")      # Title of graph

plt.show()                    #Removes scripting
df.loc[:,["age", "chol"]].plot(subplots=True)

plt.show()
df.plot(kind="scatter", x="age", y="chol", alpha=.7, color="green")

plt.xlabel("Age")

plt.ylabel("Chol")

plt.title("Cholesterol Age Scatter Graphic")

plt.show()
df.age.plot(kind="hist", bins=20, range= (0,100), density=True)   # normed is depracated Use density instead

plt.show()
df.age.plot(kind="hist", cumulative=True)

plt.show()
myDic = { "name": "Adem", "surname": "Gencer", "age": 38}  # Define a dictionary

print(myDic)

print(myDic.keys()) # Get dictionary keys

print(myDic.values()) # Get dictionary values
myDic = { "name": "Adem", "surname": "Gencer", "age": 38}  # Define a dictionary

myDic["name"] = "John"        # Edit existing value

myDic["Location"] = "Turkey"  # Add a new KEY - VALUE pair

print(myDic)

del myDic["name"]             # Delete a KEY - VALUE pair

print(myDic)

myDic.clear()                 # Clear a dictionary

print(myDic)

#del myDic                    # Delete a dictionary

#print(myDic)
myDic = { "name": "Adem", "surname": "Gencer", "age": 38}  # Define a dictionary

print("age" in myDic)      # Search a KEY in a dictionary
print(df["age"])     # series

print("--------------------------------")

print(df[["age"]])   # dataframe
# df.age.head()

# df["age"][1]

# df["age"].head()

# df.age[1]             

# df[["age", "sex"]]       # Selecting columns
dfEx = df.copy()

dfEx = dfEx.set_index("age")          # Change index.

dfEx.head()
dfEx.index.name = "myIndex"

dfEx.head()
dfEx.index = range(1,1515,5)

dfEx.head()
dfEx.set_index(["sex", "cp"]).head()
dfEx.unstack(level=0).head()

df.groupby("sex").mean()
df.groupby("sex")[["age", "chol"]].min()
df.pivot( columns="cp", values="chol").head(10)
df.loc[10:0:-1, "cp":"fbs"]  # Reverse select
xFilter = df["age"] > 70 

# This script will creates a filter based on data value.
df[xFilter]

# This code applies filter to a dataframe and shows only records which has TRUE.
xFilterAge = df["age"] > 60

xFilterChol = df["chol"] > 300
df[xFilterAge].head() # This shows AGE filtered datas
df[xFilterChol].head() # This shows CHOL filtered datas
df[xFilterAge & xFilterChol]  # Apple AGE, CHOL filters with AND operator.



#This code can be written shortly

#df[(df["age"] > 60) & (df["chol"] > 300)]
df.age[xFilterChol].head()    # Filter with xFilterChol but show age column.
def ageM(n):

    return n*12

df.age.apply(ageM).head()



# df.age.apply(lambda n: n*12)          # In a short way...



# This function will transform age cloumn. Age = age*12
df["agelabel"] = df.age * df.sex        # Only experimental result. It is meaningless.

df.head()
df["stayofhospital"] = 0              # Create a new column

df.head()




dfExt = df.head().copy()

dateList = ["2020-01-10","2020-01-12","2020-01-12","2020-01-14","2020-01-15"]

dateTime = pd.to_datetime(dateList)

dfExt["date"] = dateTime

dfExt = dfExt.set_index("date")  # Change index to timeSeries data.

dfExt.head()
dfExt.loc["2020-01-12":"2020-01-14"]   # Use TimeSeries for index.
dfExt.resample("D").mean()
dfExt.resample("D").mean().interpolate("linear")   # Interpolate missing datas with linear function.
for index, value in enumerate(df["age"][0:10]):

    print("index:", index, " value:", value)

for index, value in df[["age"]][10:11].iterrows():

    print("index:",index," value:",value)
x = 5 # This is a global variable

def myFunc():

    """ This is definition of a function"""

    y = 2               # This is a local variable. This variable cannot reached outside.

    result = x*y        # This will print x*y. 

                        # If there is no X value inside function (local) it can be searched globally.

    print("Function returned! Result = ", result)

myFunc()
def myFunc(x, y = 3, z = 5):

    """ You must declare x when you call this function. 

        If you dont declare y, z default variables used in function... """

    print(x+y+z)



myFunc(5)

myFunc(2,1)
def myFunc(*args):

    """ List of arguments. You can pass flexible parameters. """

    for each in args:

        print(each)

    # print("--",args)  # List of variables.

myFunc("first")

print("--------")

myFunc("First", "Second", 5)
def myFunc(**kwargs):

    """ Pass dictionary to a function. """

    for key, value in kwargs.items():

        print("key:", key, " value:", value)

    

myFunc(name = "Adem", age = 38)
multiply = lambda x, y: x*y  # One row function. It returns value.

print(multiply(5,4))
mystr = "TestString"

print(next(iter(mystr)))

print(*iter(mystr))
listKey = {"name", "surname", "age"}

listVal = {"Adem", "Gencer", 38}



zipList = zip(listKey, listVal)

myList = list(zipList)

print(myList)

unZip = zip(*myList)

unList1, unList2 = list(unZip)

print(unList1)        # This is a tuple

print(list(unList2))  # This is a list
myList = [1,2,3,4,5]

revList = [i + 1 for i in myList]  # Do a function for every element.

print(revList)
myList = [1,2,3,4,5]

revList = [0 if i % 2 == 0 else 1 for i in myList]  # Get 0 for even numbers.



print(myList)

print(revList)
cutoff = sum(df.age)/len(df.age)

print("Cutoff value: ",cutoff)



df["ageLevel"] = ["High age" if i > cutoff else "Low age" for i in df.age]

df.loc[:10,["age","ageLevel"]]
df.describe()
df.boxplot(column = "chol", by = "sex")

plt.show()
df_short = df.head()

meltedData = pd.melt(frame=df_short, id_vars = "age", value_vars = ["sex", "chol"])

meltedData
meltedData.pivot(index = "age", columns = "variable", values = "value")
headerData = df.head()

tailData = df.tail()

pd.concat([headerData, tailData], axis=0, ignore_index= True)
df.dtypes
df.ca = df.ca.astype(float)

df.dtypes
df.sex.value_counts(dropna=False)   # Print value counts (drop null data = false)

df.sex.dropna(inplace= True)        # Check data. If value is null drop that raw

assert 1==1                         # Check previous code.
assert df.columns[0] == "age"    # If code is true it doesnot return anything.