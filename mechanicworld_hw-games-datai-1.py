# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # +++ editted  +++

import seaborn as sns  # visualization tool # +++ editted +++



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Let's start with reading csv files;



df = pd.read_csv("../input/videogamesales/vgsales.csv")
# It will give us a first 5 rows of dataset

df.head()
# What are the names of our columns

df.columns
# with this we're going to see columns 

# and data types like a generel information

df.info()

# quick look for correlation

df.corr()
f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(df.corr(),annot=True,cmap="YlGnBu", linewidths=5, fmt=".2f",ax=ax)

plt.show()
# Line plot



# for this one line plot not so much usefull and right

# but learning and applying the code you learn, you have to try it.

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

df.Global_Sales.plot(kind= "line", color= "g", linewidth = 4, alpha = 0.5,grid= True,linestyle = "-.")

plt.legend(loc="upper right") # putting label according to location you want

plt.xlabel("x axis")          # label names

plt.ylabel("y axis")

plt.title("Line Plot")

plt.show()
# Scatter Plot

# x = year, y = Global_Slaes

df.plot(kind= "scatter", x= "Genre",y= "Global_Sales", alpha= 0.5, color= "blue")

plt.xlabel("Genre", rotation="horizontal")

plt.ylabel("Global_Sales")

plt.xticks(rotation=90,ha="right") # with xticks or yticks you can rotate your values for fixing an get good visualization

plt.yticks(rotation=45,ha="right")

plt.title("Global Sales vs Genre Scatter Plot")

plt.show()
# First let's get the number of unique years

print(len(df.Year.unique()))

print(max(df.Year))
#Histogram

# bins = number of bars in figure

a = len(df.Year.unique()) # According to unique values we can arrange the number of bars

df.Year.plot(kind="hist",bins= a,figsize=(8,8),color = "g",alpha = 0.5)

plt.show()

#plt.clf() # this metod use for clearing the graph. If you delete the "#" character and run , It will clear the graph
#We can use same example



dictionary1 = {"Shooter":"Duck Hunt","Puzzle":"Tetris"} # creating a dictionary



print(dictionary1.keys())

print(dictionary1.values())



# Let's add a new element to the dictionary

print(dictionary1)

dictionary1["Racing"] = "Wii Sport" # add a new element

print(dictionary1)

del dictionary1["Shooter"] # delete an element

print(dictionary1)

print("Puzzle" in dictionary1) # search the dictionary Puzzle is in inside , gives a boolean 

dictionary1.clear() # remove all entires in the dictionary1

print(dictionary1)

#del dictionary1 # this command will delete dictionary

a = 2 == 3 # equal

b = 2 >= 3 # bigger and equal

c = 2 <= 3 # smaller and equal

d = 2 != 3 # not equal



print(a,b,c,d)
# Filtering



Most_Valuable_Games = df["Global_Sales"] > 32 # in millions

print(Most_Valuable_Games) # It gives us the boolean results of every row tru eor false according to condition



df[Most_Valuable_Games] # This filtering gives us the table according to condition 

                        # what we have in it. We have 4 games abbove that glabal sales value

    

# with logical_and



df[np.logical_and(df["Global_Sales"] > 32,df["Platform"] == "Wii")]
# with "&"

df[(df["Global_Sales"] > 32) & (df["Platform"] == "Wii")]
series = df['Genre']        # df['Genre'] = series

print(type(series))

data_frame = df[['Genre']]  # df[['Genre']] = data frame

print(type(data_frame))

list1 = ["a","b","c","d","e"]



for each in list1:

    print(each)

    

print("-------")



    # enumerate index and values of the list

    

for index_of_list1, values_of_list1 in enumerate(list1):

    print(index_of_list1," : ",values_of_list1)

    

print("-------")



# Dictionaries

# Lets's find keys and values in the dictinoary with a for loop



dictionary1 = {"Shooter":"Duck Hunt","Puzzle":"Tetris"}



for key, value in dictionary1.items():

    

    print(key," : ",value)

    

print("-------")



# For pandas let's look at 



for index, value in df[["Global_Sales"]][0:1].iterrows():

    print(index," : ",value)

# b and c values are default arguments



def f(a, b = 1, c = 2):

    y = a + b + c

    return y



print(f(5))



print(f(5,3,8))
### flexible arguments *args , *args can be one or more



def f(*args):

    for i in args :

        print(i)

        



print(f(1))

print("------")

print(f(1,2,3,4))



# flexible arguments  **kwargs that is for dictionary 



def f(**kwargs):

    

    """print key and value of dictionary"""

    for key, value in kwargs.items():

        print(key," ",value)

        

f(country = "spain", capital = "madrid", population = 123456)





# lambda function



square = lambda x:x**2



print(square(5))
#Anonymous Function

# More than one arguments

# map(func,seq)



number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(type(y))

print(y)

print(list(y))

print(y)
name = "Baris Dag"

iter_1 = iter(name)

print(next(iter_1)) # print next iteration

print(next(iter_1))

print(next(iter_1))

print(next(iter_1))

print(next(iter_1))

print(*iter_1) # print remaining iteration

# zip --> zipping the lists



list1 = [2,3,4,5]

list2 = [10,11,12,13]

z = zip(list1,list2)

print(z)

print(type(z))

z_list = list(z)

print(z_list)

print(type(z_list))

print(z_list[0][1])

print(type(z_list[0]))
un_zip = zip(*z_list)

print(un_zip)

print(type(un_zip))

un_list1,un_list2 = list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list1))

print(list(un_list1))

# Example



num1 = [1,2,3]

num2 = [i + 1 for i in num1]

print(num2)
# conditionals on iterable



num1 = [5,10,15]

num2 = [ i**3 if i == 5 else i - 5 for i in num1]

print(num2)



list1 = [25,34,22]

list2 = ["It is an even number" if i % 2 == 0 else "It is an odd number" for i in list1]



print(list2)

df.head()
from termcolor import colored



Mean_Global_Sales = sum(df.Global_Sales)/len(df.Global_Sales)

print(Mean_Global_Sales)



df["Sales_Level"] = ["Higher" if i > Mean_Global_Sales else "Lower" for i in df.Global_Sales]

df.loc[:,["Sales_Level","Global_Sales"]]







df["Color_Code"] = ["Blue" if i > 30 else "Green" if i < 20 else "Grey" for i in df.Global_Sales]

df.loc[:25,["Color_Code","Global_Sales"]]
df.head()
print(df["Platform"].unique())

print(df["Platform"].value_counts(dropna = False))
print(df["Genre"].value_counts(dropna = False))

print(df["Publisher"].value_counts(dropna = False))

lis1 =[0,1,2,3,4,15]



print(np.mean(lis1),np.median(lis1))
#print(df)

df.boxplot(column="Global_Sales", by ="Genre")
df_new = df.head()
melted = pd.melt(frame=df_new,id_vars = 'Name', value_vars= ['NA_Sales','EU_Sales'])

melted

melted.pivot(index = 'Name', columns = 'variable',values='value')
df1 = df.head()

df2 = df.tail()

conc_data_row = pd.concat([df1,df2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row = vertical

conc_data_row
df_1 = df["Year"].head()

df_2 = df["Global_Sales"].head()

conc_data_col = pd.concat([df_1,df_2],axis = 1) # axis = 1 , columns = horizontal

conc_data_col
df.head()
print(df["Year"].unique())
df.dtypes
df.head()
Students = ["Ahmet","Can","Berk","Beril","Banu","Gamze","Meltem","Evrim","Murat"]

Heights_of_Students = [156,169,142,153,142,170,156,154,152]

List_of_Column_s_Names = ["Names","Height"]

List_of_Df_s_Colums = [Students,Heights_of_Students]

Z_List = zip(List_of_Column_s_Names,List_of_Df_s_Colums)

Dict_Z_List = dict(Z_List)

print(list(Z_List))

df_students = pd.DataFrame(Dict_Z_List)

df_students

df_students["Age"] = [12,12,11,12,13,12,12,12,12]

df_students
df_students["Weight"] = [50,65,42,40,40,52,42,41,65]

df_students["Grade"] = 6

df_students["Class_NO"] = [1,2,3,4,5,6,7,8,9]

df_students
df_students_plot1 = df_students.iloc[:,[1,2,4]]

df_students_plot1.plot()

plt.show()

df_students_plot1.plot(subplots = True)

plt.show()
df_students.plot(kind ="scatter", x="Height",y="Weight")

plt.show()
df_students_plot1.plot(kind="hist",y ="Height",bins = 8,range = (130,180) , density = True)
fig, axes = plt.subplots(nrows=2,ncols=1)



df_students.plot(kind="hist", y ="Height",bins = 8,range = (130,180),density = True, ax=axes[0])

df_students.plot(kind="hist", y ="Height",bins = 8,range = (130,180),density = True, ax=axes[1],cumulative = True)

plt.savefig("graph1.png")

plt



df_plot1 = df.loc[:250,["EU_Sales","Global_Sales"]]

df_plot1.plot()

plt.show()
df.plot(kind="scatter", x="Global_Sales",y="NA_Sales")
#time_list = ["1992-03-08","1992-05-10","1993-08-22"]

#print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

#datetime_object = pd.to_datetime(time_list)

#df_example["date"] = datetime_object

#print(type(datetime_object))

# If you want to set these values in to index you have yo use set_index method for dataframes

# df = df.set_index("date")
# If you want to use your time series for resampling

 # example

# dataframe_example.resample("M").mean()  # M = Month aylara göre bak ve ortalama al

# dataframe_example.resample("A").mean()  # A = Year aylara göre bak ve ortalama al
# Fill in the blanks

# data2.resample("M").first().interpolate("linear") # aylara göre 

# ortalama alırken boş kalan değerleri lineer ir şekilde, başlangıç ve bitiş aralıklarına göre 

# doldurmak için kullanılır
df_students = df_students.set_index("Class_NO")

df_students
df_students.Names[1]
df_students["Names"][1]
df_students[["Names","Age"]]
df_students.loc[1,["Names"]]
df_students.loc[1,["Names"]]
df_students.loc[1:5,["Height"]]
df_students.loc[9:0:-1,"Height":"Age"]
df_students.loc[9:0:-1,"Height":]
filter1 = df_students.Age < 13

filter2 = df_students.Weight > 50

df_students[filter1 & filter2]
df_students.Names[df_students.Age < 13]
def div1 (x) :

    return x / 2



df_students.Age.apply(div1)

df_students.Age.apply(lambda x : x/2)
df_students["H/W"] = df_students.Height / df_students.Weight

df_students
df_s_copy = df_students.copy()

df_s_copy
df_s_copy.index.name = "Index"

df_s_copy
df_s_copy.index = range(10,100,10)

df_s_copy
# df_s_copy = df_s_copy.set_index("example_column_name")

#df_s_copy.index = df_s_copy["example_column_name"]
df.head()
df_copy = df.copy()
df_copy_1 = df_copy.set_index(["Platform","Genre"])

df_copy_1.head(1000)
df_s_copy["Gender"] = ["M","M","M","F","F","F","F","M","M"]

df_s_copy
df_s_copy_1 = df_s_copy.drop(["H/W"], axis=1)

df_s_copy_1["Hobby"] = ["Music","Basketball","Music","Basketball","Music","Basketball","Music","Basketball","Music"]

df_s_copy_1
dic1 = {"Auto":["On","On","Off","Off"],

        "5S":["Applied","None","Applied","None"],

       "Efficiency":[0.85,0.70,0.60,0.5]}

df_dic = pd.DataFrame(dic1)

df_dic

df_dic.pivot(index="Auto",columns="5S",values="Efficiency")
df_dic_s = df_dic.set_index(["5S","Auto"])

df_dic_s

df_dic_s.unstack(level=0)

df_dic_s.unstack(level=1)
df_dic_s_1=df_dic_s.swaplevel(0,1)

df_dic_s_1
df_dic["Worker"]=[120,110,80,70]

df_dic
pd.melt(df_dic,id_vars="Auto",value_vars=["Efficiency","Worker"])
df_dic.groupby("Auto").mean() 
df_dic.groupby("5S").Efficiency.mean()
df_dic.groupby("5S")[["Efficiency","Worker"]].min()