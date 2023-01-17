# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plot
data=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
import matplotlib.pyplot as plt

import seaborn as sns 
data.info()
data.corr()
f,ax = plt.subplots(figsize=(8,3))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=" .1f",ax=ax)

plt.show()
data.head(8) # ilk 8 kişi
data.tail(8) # Son 8 kişi
data.columns
#data.columns=[ i.split()[0]+"_"+i.split()[1]"_"+i.split()[2]"_"+i.split()[3] if (len(i.split()))==4 i.split()[0]+"_"+i.split()[1]"_"+i.split()[2] if (len(i.split()))==3 i.split()[0] +"_"+ i.split()[1] if (len(i.split()))==2 else i for i in data.columns]



    

#data.columns=[i.split()[0]+"_"+i.split()[1]+"_"i.split()[1]+"_"i.split()[1]+"_"  if len(i.split())>1 else i for i in data.columns]



#data.rename(index = {"parental level of education": "parental_level_of_education", 

                    # "math score":"math_score"}, 

                    #             inplace = True) 

data.columns = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 

                'test_preparation_course', 'math_score', 'reading_score', 'writing_score'] 

data.columns
data.reading_score.plot(kind ="line",color="b",label="reading_score",linewidth=2,alpha=.8,grid=False,linestyle=":")

data.math_score.plot(color="y",label="math_score",linewidth=1,alpha=0.9, grid=False,linestyle=":")

plt.legend(loc="upper left")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Plot")

# Scatter Plot

data.plot(kind="scatter",x="writing_score",y="math_score",alpha=.8,color="r")

plt.xlabel("writing_score")

plt.ylabel("math_score")

plt.title("writing_score math_score Scatter Plot ")
data.math_score.plot(kind="hist",bins=30, figsize=(8,8))
dict={"Ahmet": 50, "Mehmet": 85, "Veli": 75}

print(dict.keys())

print(dict.values())
dict["Ahmet"]= 65 # update existing entry

print(dict)

dict["Deniz"]=95 # add new entry

print(dict)

del dict["Mehmet"] # delete entry with Mehmet

print(dict)

print("Veli" in dict)

print(dict)

dict.clear()

print(dict)
data=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
series=data["gender"]

print(type(series))

df= data[["gender"]]

print(type(df))
#Comporations operations

print(5>6)  # bool True or False

print(5!=6)

print(True and False)

print( True or False)
data.columns = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 

                'test_preparation_course', 'math_score', 'reading_score', 'writing_score'] 

data.columns
x = data["math_score"]>95 # 1  filtering vith pandas dataframe

data[x]
# 2 filtering pandas logical_and

data[np.logical_and(data["math_score"]>95,data["writing_score"]>95)]

data[(data["math_score"]>95) & (data["writing_score"]>95)]
i=0

while i<=8:

    print("i is: ",i)

    i+=1

print(" i is equal to 8")
lis2=[1,2,3,4,5,6,7,8,9,10]

for i in lis2:

    print("i is: ",i)

print(" ")



for index,value in enumerate(lis2):

    print(index ," : ", value)

print(" ")

dict={"Ahmet": 50, "Mehmet": 85, "Veli": 75}

for key, value in dict.items():

    print(key," : ", value)

print(" ")



for index, value in data[["reading_score"]][0:2].iterrows():

    print(index," : ", value)
print ("***************2. Part of Homework is begining***************")
r="reading_score"

w="writing_score"

m="mat_score"

def ort (r,w,m):

    z=(r+w+m)/3

    print("ortalama: ",z)

    return ort

"""Üç notun ortalaması"""

print(ort(75,80,85))



x=90

def f():

    x= 85

    return x

print(x)

print(f())
x=90

def f():

    y=x+5

    return y

print(x)

print(f())

import builtins

dir(builtins)
r=6

pi=3.14 

def vol(): 

     

    def ar ():  

    

        def crf(): # circumference

            c=2*pi*r

            return c

        print("Çemberin Çevresi: ",crf(),"cm")

        return crf()/r*2 # area of circle = "pi*r*r" or "r*(2*pi*r)/2"

    print("Dairenin Alanı: ",ar(),"cm2")

    return ar()*4/3*r # volume of sphere = 4/3*pi*r*r*r or 4/3*(pi*r*r)*r





print("Kürenin Hacmi: ",vol(),"cm3")

    

def vol (r,pi=3.14):

    return 4/3*pi*r**3

print(vol(5)) # pi is default

print(vol(5,3)) # pi is given "3"

    
def f(*args):

    for i in args:

        print(i)

f(".a)")

print("")

f(".a)",".b)",".c)",".d)")
def f(**kwargs):

    for key, value in kwargs.items():

        print(key," ",value)

f(student="Deniz", Note=95, Genre="Male")
vol= lambda r: 4/3*3.14*r**3

print(vol(5))                   # pi = 3.14

vol2= lambda r,pi: 4/3*pi*r**3

print(vol2(5,3))                # pi = 3
number_list=[2,4,6,8]

a=map(lambda b: b+b**2,number_list)

print(list(a))
list1=[2,4,6,8]

list2=["a","b","c","d"]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)
u_zip = zip ( *z_list)

u_list1,u_list2 = list(u_zip)

print("List1= ",u_list1)

print("List2= ",u_list2)

data.columns
data.columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch',

       'test preparation course', 'math score', 'reading score',

       'writing score']

data.columns # ı have deleted "_" at the colums words again
# I will put the "_" for space in the columns words again



data.columns=[ i.split()[0]+"_"+i.split()[1]+"_"+i.split()[2]+"_"+i.split()[3] if (len(i.split()))==4 else i.split()[0]+"_"+i.split()[1]+"_"+i.split()[2] if (len(i.split()))==3 else i.split()[0] +"_"+ i.split()[1] if (len(i.split()))==2 else i for i in data.columns]

data.columns
thresoldmat=sum(data.math_score)/len(data.math_score)

data["Mat_level"] = ["High"if i>thresoldmat else "Low" for i in data.math_score]

data.loc[:10,["Mat_level","math_score"]]
print("***************  3. Part of Homework is begining***************")
data.head()
data.tail()
data.columns
data.shape
data.info()
print(data["race/ethnicity"].value_counts(dropna = False))
print(data["parental_level_of_education"].value_counts(dropna = False))
data.describe()
data.boxplot(column="math_score", by="test_preparation_course")
data_nev=data.head()

data_nev
melted=pd.melt(frame=data_nev,id_vars="math_score",value_vars=["race/ethnicity","lunch","reading_score"])

melted
melted.pivot(index = "math_score"  , columns="variable",values="value")
dt1=data.head()

dt2=data.tail()

conc_dts=pd.concat([dt1,dt2],axis=0,ignore_index=True)

dt1
dt2
conc_dts
dt1b=data["math_score"].head(10)

dt2b=data["Mat_level"].head(10)

conc_dtss=pd.concat([dt1b,dt2b],axis=1)

conc_dtss
data.dtypes
data["gender"]=data["gender"].astype("category")

data["math_score"]=data["math_score"].astype("float")

data.dtypes
data["math_score"].head()

data["gender"].head()
data.info()
print("there is no missing data in my work so I can't do nothing")
import pandas as pd
#clas=["8/A","8/B"]

#population=["24","26"]

#list_label=["clas","population"]

#list_col=[clas,population]

#zp=list(zip(list_label,list_col))

#datadic=dict(zp)

#df=pd.DataFrame(datadic)

#df
data  = data.loc[:,["math_score","reading_score","writing_score"]]



data.plot()
data.plot(subplots =True)

plt.show()
data.plot(kind="scatter",x="math_score",y="writing_score")
data.plot(kind="hist", y= "writing_score",bins=50, range=(0,250), normed= True )
fig,axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind="hist", y= "writing_score",bins=50, range=(0,250), normed= True, ax=axes[0])

data.plot(kind="hist", y= "writing_score",bins=50, range=(0,250), normed= True, ax=axes[1], cumulative =True )

plt.savefig('graph.png')

plt
data.describe()
time_list = ["1995-09-19","1991-04-21"]

print(type(time_list))

datatime_object = pd.to_datetime(time_list)

print(type(datatime_object))

import warnings

warnings.filterwarnings("ignore")



data2=data.head()

data_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(data_list)

data2["date"]=datetime_object

# data2=data2.set_index("date")

data2
import warnings

warnings.filterwarnings("ignore")



data2=data.head()

data_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(data_list)

data2["date"]=datetime_object

data2=data2.set_index("date") # For indexing

data2
print(data2.loc["1993-03-15"])

print(data2.loc["1992-02-15":"1993-03-15"]) # Look the first Time
print(data2.loc["1993-03-15"])

print(data2.loc["1992-02-10":"1993-03-15"]) # Look the first Time and now
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

#data = data.set_index("gender")

data.tail(10)
a = [i for i in range(1,1001)]

print(a)
import warnings

warnings.filterwarnings("ignore")



data3=data.head(1000)

data_list1=a

#datetime_object1 = pd.to_datetime(data_list1)

data3["#"]=data_list1

data3=data3.set_index("#")

data3
data["lunch"][2]
data.lunch[2]
data.loc[2,["lunch"]]
data[["lunch","math score"]]
print(type(data["math score"])) # Seri

print(type(data[["math score"]])) # data Frame
data.loc[1:10,"gender":"math score"]
data.loc[10:1:-1,"gender":"math score"]
data.loc[1:10,"math score":]
boolean= data["math score"]>98

data[boolean]
first_filter =data["reading score"]>99

second_filter =data["math score"]>98

data[first_filter&second_filter]
data[data["math score"]<20]
data["reading score"][data["math score"]<20]
def div (n):

    return n/2

data["math score"].apply(div)
data["math score"].apply(lambda n : n/2)
data["total_score"]= data["math score"]+data["reading score"]+data["writing score"]

data.head()
print(data3.index.name)
data3.index.name ="index"

data3.head()
data3.head()

data4 =data3.copy()

data4.index= range(100,1100,1)

data4.head()
data.head()
data5= data.set_index(["gender","race/ethnicity"])

data5.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","M","F"],"response":["15","45","20","9"],"age":["15","4","72","65"]}

df= pd.DataFrame(dic)

df
df.pivot(index="treatment",columns ="gender",values="response")
df1=df.set_index(["treatment","gender"])

df1
df1.unstack(level=0)
df2 =df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean()
df.info()