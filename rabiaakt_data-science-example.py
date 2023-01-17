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
data = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

data.head()
data.info()
data.corr()

male = data[data["Gender"] == 'Male']

male.head()
f, ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(), annot = True, linewidths =.7, fmt= '.7f', ax = ax )

plt.show()
data.columns

data.Age.plot(kind='line', color="red", label="Age graph", linewidth=1 ,alpha = 0.5, grid = True, linestyle = ':')
data.plot(kind="scatter", x='Age', y='Gender', alpha=0.5, color="red")

plt.xlabel("Age")

plt.ylabel("Gender")

plt.title("Age-Gender Scatter Plot")

data.Age.plot(kind="hist", bins=50, figsize=(10,10), color="g")

plt.show()
data[np.logical_and(data["Age"]>20,data["Response"]>0)]
def tuple_ex():

    """This is docstring for writing info about this function"""

    tupl = (1,2,3)

    return tupl

tuple_ex()
x=9

def pout():

    """If there is no local scope (variable name -- x) then it uses global value of the variable"""

    x=5

    print(x)

print(x)

print(pout())
x=7

def mult():

    y = 2*x

    return y

mult()
import builtins

dir(builtins)
def nested():

    

    def nest():

        x=5

        y=7

        z=x+y

        return z

    return nest()**2

nested()
def arg(*args):

    for i in args:

        print(i)

arg(1,2,4,5,6,6,5,650)



def argk(**kwargs):

    for key in kwargs.items():

        print(key)

        

argk(country = 'turkey', capital = 'ankara', population = 426865685)
def squ(x):

    return x**2



square = lambda x:x**2
list_n =[1,4,7,8,10]

func = map(lambda x:x**2,list_n)

list(func)
name="Rabia"

iterator = iter(name)

print(next(iterator))

print(next(iterator))

print(*iterator)
list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip()
num1 = [4,7,8]

num2 = [i+1 for i in num1]

print(num2)
num_list = [4,6,7,8,9,10]

num_list2 = [i+5 if i<5 else i+7 if i>7 else i+1 for i in num_list]

print(num_list2)
threshold = data.Age.mean()

print(threshold)

data["age_level"] = ["higher than mean" if i>threshold else "lower than mean" for i in data.Age]

data.loc[::10,["age_level","Age"]]
data.head()

data.shape
data.columns
data.info
data.info()
print(data["Gender"].value_counts(dropna=False))
data.describe
data.describe()
data.Age.mean()
boxplot = data.boxplot(grid=False, rot=45, fontsize=15)
data.boxplot(column="id")
new_data = data.head()

new_data
melted = pd.melt(frame=new_data,id_vars="id",value_vars=["Gender","Age"])

melted
melted.pivot(index = 'id', columns = 'variable',values='value')
data1 = data.head()

data2 = data.tail()

conc = pd.concat([data1,data2],axis=0)

conc2 = pd.concat([data1,data2],axis=0,ignore_index=True)



conc2
data.dtypes
data["Region_Code"].value_counts(dropna=False)
assert 1==2
assert 1==1
assert data["Age"].notnull().all()
data["Age"].fillna('empty',inplace = True)
country = ["İstanbul","ANkara"]

population = ["10","11"]

population_column = ["Country","Population"]

population_rows = [country,population]

zipped = zip(population_column,population_rows)

dictionary = dict(zipped)

dictionary

df = pd.DataFrame(dictionary)

df
df["Capital"] = ["aa","bb"]

df
df["high"] = 0

df
data_new = data.loc[:,["Driving_License","Region_Code","Annual_Premium","Policy_Sales_Channel","Vintage"]]

data_new.plot()
data_new.plot(subplots=True)

plt.show()
data.plot(kind="scatter",x="Gender",y="Driving_License")

plt.show()
data.plot(kind="hist",y="Age",bins = 50,range= (0,250))

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind="hist",y="Age",bins = 50,range= (0,250),ax=axes[0])

data.plot(kind="hist",y="Age",bins = 50,ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt
import warnings

warnings.filterwarnings("ignore")

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_list = pd.to_datetime(date_list)

data2["Date"] = datetime_list

data2 = data2.set_index("Date")

data2

print(data2.loc["1992-03-10":"1993-03-16"])
time_list = ["1998-05-14","1993-06-11"]

datetimelist = pd.to_datetime(time_list)

datetimelist
data2.resample("A").mean() #resample year
data2.resample("M").mean() #resample month
data2.resample("M").first().interpolate("linear")
data
data = data.set_index("id")
data["Age"][1]
data.Age[1]
data.loc[:,["Age"]]
data[["Age","Gender"]]
data.loc[1:10,"Age":"Vehicle_Damage"]
data.loc[10:1:-1,"Age":"Vehicle_Damage"] 
data.loc[1:10,"Previously_Insured":] 
boolean = data.Age > 76

data[boolean]
boolean1 = data.Age > 50

boolean2 = data.Age <60

data[boolean1 & boolean2]
data.Gender[data.Age > 80]
def mult(a):

    return a*2

data.Age.apply(mult)
data.Age.apply(lambda a:a/2) #using lambda function
data["preInsured+drivingLicense"] = data.Driving_License + data.Previously_Insured

data
print(data.index.name)

data.index.name = "id"

data.head()
data3 = data.copy()

data3.index = range(100,381209,1) #100'den 900 e kadar yeniden indexi numaralandır

data3.head()
# It was like this

# data= data.set_index("#")

# also you can use 

# data.index = data["#"]
data1 = data.copy()

#data1 = data.set_index(["Type 1","Type 2"]) using more than 1 index it is useful when we want to categorized 
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df.pivot(index="treatment",columns = "gender", values = "response")
df_new = df.set_index(["treatment","gender"])

df_new
df_new.unstack(level=0)
df_new.unstack(level=1)
df_new2 = df_new.swaplevel(0,1)

df_new2
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df.groupby("gender").mean()
df.groupby("treatment").response.min()
df.groupby("treatment")[["age","response"]].min() 
df.info()
def s(x, y = 2):



    c = 2



    for i in range(y):



        c = c + x



    return c

s(2,3)
a = [0,1,2,3,4]



for a[0] in a:



    print(a[0])


