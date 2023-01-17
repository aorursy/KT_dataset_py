# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/cwurData.csv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
main_data = pd.read_csv('../input/cwurData.csv')
#ees_data = pd.read_csv('../input/education_expenditure_supplementary_data.csv')
eas_data = pd.read_csv('../input/educational_attainment_supplementary_data.csv')
s_c_data = pd.read_csv('../input/school_and_country_table.csv')
shangai_data = pd.read_csv('../input/shanghaiData.csv')
times_data = pd.read_csv('../input/timesData.csv')


main_data.info()
main_data.corr()
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(main_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(shangai_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(times_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
print(main_data.head())
print(eas_data.head())
print(s_c_data.head())
print(shangai_data.head())
print(times_data.head())
print(main_data.shape)
print(eas_data.shape)
print(s_c_data.shape)
print(shangai_data.shape)
print(times_data.shape)
main_data.info()
main_data.publications.plot(kind = 'line', color = 'g',label = 'Yayınlar',linewidth=1,alpha = 0.5,grid = True) 
main_data.quality_of_education.plot(color = 'r',label = 'Quality Of Education',linewidth=1, alpha = 0.5,grid = True) 
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
main_data.plot(kind='scatter', x='publications', y='quality_of_education',alpha = 0.3,color = 'blue', figsize = (15,15))
plt.xlabel('Publications')              # label = name of label
plt.ylabel('Quality_of_education')
plt.title('Publications - Quality_of_education Scatter Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
#main_data.info()
main_data.quality_of_education.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
#create dictionary and look its keys and values
my_dictionary = {'istanbul' : 'istanbul üni','konya' : 'selçuk üni'}
print(my_dictionary.keys())
print(my_dictionary.values())
print(my_dictionary)
my_dictionary['istanbul'] = "boun"
print(my_dictionary)
my_dictionary['ankara'] ="metu"
print(my_dictionary)
del my_dictionary['istanbul']
print(my_dictionary)
print('istanbul' in my_dictionary)
print('ankara' in my_dictionary)
my_dictionary.clear()
print(my_dictionary)
series = main_data['quality_of_education']        # data['Defense'] = series
print(type(series))
data_frame = main_data[['quality_of_education']]  # data[['Defense']] = data frame
print(type(data_frame))
ort_egit_kalitesi = data_frame['quality_of_education'].mean()
print(ort_egit_kalitesi)
ort_egit_kalitesi = main_data['quality_of_education'].mean()
print(ort_egit_kalitesi)

ort_yayin_sayisi = main_data['publications'].mean()
print(ort_yayin_sayisi)
ort_ustunde_egit_kalitesi = main_data['quality_of_education'] > ort_egit_kalitesi
ort_ustunde_yayin = main_data['publications'] > ort_yayin_sayisi
main_data[ort_ustunde_egit_kalitesi & ort_ustunde_yayin]
j = 0
while j != -5 :
    print('j is: ',j)
    j -=1 
print(j,' is equal to -5')
"""
The Fibonacci Function
In : "k" is number of list index and natural number bigger than 1 
Out: [a0,a1,....,ak] = [1,1,...,n]
"""

def func_fibonacci(k=2):
    my_list = [1,1]
    
    for i in range(2,k):
        my_list.append(my_list[i-1]+my_list[i-2])
    return my_list

# example:
func_fibonacci(10)

# [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(my_list):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in main_data[['national_rank']][0:1].iterrows():
    print(index," : ",value)

# User Defined Function

def tuble_2():
    """ return defined t tuble"""
    t = (2,4,6)
    return t
a,b,c = tuble_2()
print(a,b,c)
# Scope

x = 1
def f():
    x = 4
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope
x = 4
def f():
    y = 3*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
import builtins
dir(builtins)
#nested function

def cube():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 4
        y = 5
        z = x + y
        return z
    return add()**3
print(cube())  
# default arguments

def f(a=3, b = 4, c = 2):
    y = a + b + c
    return a,b,c,y
print(f(1))

print(f(5,4,3))
# flexible arguments *args
def f(f_arg, *args):
    print ("first normal arg:", f_arg)
    #for i in args:
    print("another arg through *argv :", args)
f(3)
print("")
f(a,b,c)
# flexible arguments **kwargs that is dictionary

def bar(first, second, third, **kwargs):
    if kwargs.get("action") == "sum":
        print("The sum is: %d" %(first + second + third))

    if kwargs.get("number") == "first":
        return first

result = bar(1, 2, 3, action = "sum", number = "first")
print("Result: %d" %(result))
# lambda function
cube = lambda x: x**3     # where x is name of argument
print(cube(2))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(-1,-2,-3))
number_list = [2,3,5,7,11]
y = map(lambda x:x**2,number_list)
print(list(y))
# iteration example
name = "da vinci"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration
# zip example
list1 = [1,2,3]
list2 = ['a','b','c','d']
z = zip(list1,list2)
print(z)
print(type(z))
z_list = list(z)

print(z_list)
print(list(z))     # Whay this is empty?
print(list(zip(list1,list2)))
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(list(un_list2))

print(type(un_list2))
print(type(list(un_list2)))
# Example of list comprehension
num1 = [3,4,5]
num2 = [2*i + 2 for i in num1 ]  #list comprehension
print(num2)
num3 = [6,8,11]
num4 = [i**2 if i == 7 else i-5 if i < 4 else i+2 for i in num3]
print(num4)
threshold = sum(main_data.publications)/len(main_data.publications)
main_data["publications"] = ["high" if i > threshold else "low" for i in main_data.publications]
main_data.loc[:10,["publications","publications"]] # we will learn loc more detailed later

# In this section, it's running when I try run this code block in first time. But, it is not running after that. I get it this error: TypeError: unsupported operand type(s) for +: 'int' and 'str'
num1 = int(input("What is your first number? "))
num2 = int(input("What is your second number? "))
num3 = int(input("What is your third number? "))
numlist = [num1, num2, num3]
print(numlist)
print("Now I will remove the 3rd number")
print(str(numlist.pop(2)) + " has been removed")
print("The list now looks like " + str(numlist))
main_data.head()
main_data.tail()
main_data.columns
main_data.shape
main_data.info()
print(main_data.country.value_counts(dropna =False))
main_data.describe()
main_data.boxplot(column=['alumni_employment'],by = 'year')
plt.show()

main_data.boxplot(column='quality_of_education',by = 'year')
plt.show()
main_data_new = main_data[main_data.year == 2012].head()    # I only take 5 rows into new data
main_data_new
melted = pd.melt(frame=main_data_new,id_vars = 'institution', value_vars= ['alumni_employment','publications'])
melted
melted.pivot(index = 'institution', columns = 'variable',values='value')
data1 = main_data.head()
data2= main_data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = main_data['alumni_employment'].head()
data2= main_data['publications'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
main_data.dtypes
main_data.institution = main_data.institution.astype('category')
main_data.patents = main_data.patents.astype('float')
main_data.dtypes
main_data.info()
main_data.country.value_counts(dropna =False)
main_data_1=main_data
main_data_1.country.dropna(inplace = True)
assert 1==1
assert  main_data.country.notnull().all()   # return nothing
main_data.country.fillna('empty',inplace = True)
assert  main_data.country.notnull().all()   # return nothing
university = ["METU","BOUN"]
population = ["1100","1200"]
list_label = ["university","population"]
list_col = [university,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df["department"] = ["math", "math"]
df
df["student"] = 0
df
main_data.info()
main_data_1 = main_data.loc[:,["quality_of_faculty","publications","broad_impact"]]
main_data_1.plot()
main_data_1.plot(subplots = True)
plt.show()
main_data_1.plot(kind = "scatter",x="broad_impact",y = "publications")
plt.show()
main_data_1.plot(kind = "hist",y = "broad_impact",bins = 50,range= (0,250),normed = True)
fig, axes = plt.subplots(nrows=2,ncols=1)
main_data_1.plot(kind = "hist",y = "broad_impact",bins = 50,range= (0,250),normed = True,ax = axes[0])
main_data_1.plot(kind = "hist",y = "broad_impact",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
main_data.describe()
time_list = ["2018-03-08","2018-04-12"]
print(type(time_list[1])) 
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore")

main_data_2 = main_data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
main_data_2["date"] = datetime_object
# lets make date as index
main_data_2= main_data_2.set_index("date")
main_data_2 
print(main_data_2.loc["1993-03-16"])
print(main_data_2.loc["1992-03-10":"1993-03-16"])
main_data_2.resample("A").mean()
main_data_2.resample("M").mean()
main_data_2.resample("M").first().interpolate("linear")
main_data_2.resample("M").mean().interpolate("linear")
main_data = pd.read_csv('../input/cwurData.csv')
#main_data= main_data.set_index("#")   # this code doesn't work!
main_data.index +=1   # This code works
main_data.head()
main_data.country[0]
main_data.loc[3, "institution"]
main_data[["institution", "country"]]
print(type(main_data["country"]))     # series
print(type(main_data[["country"]]))   # data frames
main_data.loc[0:10, "world_rank":"quality_of_education"]
main_data.loc[10:0:-1, "world_rank":"quality_of_education"]
main_data.loc[0:10, "quality_of_education":]
boolean = main_data.quality_of_education > 25
main_data[boolean]
first_filter = main_data.quality_of_education > 350
second_filter = main_data.publications > 990
main_data[first_filter & second_filter]
main_data.institution[first_filter & second_filter]
def div(n):
    return n/2
main_data.quality_of_education.apply(div)
main_data.quality_of_education.apply(lambda n : n/2)
main_data["education_score"] = main_data.quality_of_education + main_data.score
main_data.head()
print(main_data.index.name)

main_data.index.name = "index_name"
main_data.head()
main_data.head()

main_data_2 = main_data.copy()
main_data_2.index = range(100,2300,1)
main_data_2.head()
main_data_2.index +=1
main_data_2.index
main_data.head()
main_data_3 = main_data.set_index(["country","institution"]) 
main_data_3.head(100)
main_data= main_data.set_index("#")
dic = {"treatment":["A","A","B","B","C","C"],"gender":["F","M","F","M","F","M"],"response":[10,45,5,9,20,35],"age":[15,4,72,65,58,17]}
df = pd.DataFrame(dic)
df
df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean()
df.groupby("treatment").age.max() 
df.groupby("gender").age.min() 
df.info()
