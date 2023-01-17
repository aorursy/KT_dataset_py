import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(13,13))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
data.head(13)
data.columns
#lineplot

data.Speed.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=0.5,grid=True,linestyle=':')

data.Defense.plot(color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

#plt.show()
#scatterplot

#x-attack,y-defense

data.plot(kind="scatter",x="Attack",y="Defense",alpha=0.5,color="red")

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Attack vs. Defence Scatter Plot")
plt.scatter(data.Attack,data.Defense,alpha=0.3,color="k")
#histogram

#bins=#of bars

data.Speed.plot(kind="hist",bins=50,figsize=(10,10),grid=True)

plt.xlabel("speed")

plt.ylabel("number of pokemons")

#plt.clf() -> prevents to draw the plot
dictionary = {"spain":"madrid","usa":"vegas"}

print(dictionary.keys())

print(dictionary.values())
dictionary["spain"]="barcelona"

print(dictionary)

dictionary["france"]="paris"

print(dictionary)

del dictionary["spain"]

print(dictionary)

print("france"in dictionary)

#dictionary.clear()

#del dictionary

print(dictionary)
#dataframe vs. serial !be careful!

series=data["Defense"]

print(type(series))

data_frame=data[["Defense"]]

print(type(data_frame))
#filtering

x = data["Defense"]>200

data[x]
data[np.logical_and(data["Defense"]>200,data["Attack"]>100)]

#data[(data["Defense"]>200)&(data["Attack"]>100)]
lis=[1,2,3,4,5]

for i in lis:

    print("i is: ",i)
for index,value in enumerate(lis):

    print(index," : ",value)
dictionary = {"spain":"madrid","france":"paris"}

for key,value in dictionary.items():

    print(key," : ",value)
for index,value in data[["Attack"]][0:1].iterrows():

    print(index," : ",value)
def tuble_ex():

    t = (1,2,3)

    return t

a,b,c=tuble_ex()

print(a,b,c)
x=2

def f():

    x=3

    return x

print("global scope: ",x)

print("local scope : ",f())
#nested func

def square():

    def add():

        x=2

        y=3

        z=x+y

        return z

    return add()**2

print(square())
def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)



def f(**kwargs):

    for key,value in kwargs.items():

        print(key," : ",value)

f(countary="spain",capital="madrid",population=123456)
square =lambda x: x**2

print(square(4))

tot = lambda x,y,z: x+y+z

print(tot(1,2,3))
#anonymous function

#map(func,seq) fonksiyonu dizideki elemanlara uygular

number_list=[1,2,3]

y=map(lambda x:x**2,number_list)

print(list(y))
#iterators

name = "ronaldo"

it = iter(name)

print(next(it))

print(next(it))

print(next(it))

print(*it)
#zip

list1=[1,2,3,4]

list2=[5,6,7,8]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)
un_zip=zip(*z_list)

un_list1,un_list2=list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list2))

print(type(list(un_list1)))
num1=[1,2,3]

num2=[i+1 for i in num1]

print(num2)
num1=[5,10,15]

num2=[i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
threshold=sum(data.Speed)/len(data.Speed)

print("threshold: ",threshold)

data["speed_level"]=["high" if i>threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.head()
data[(data["Legendary"]==1)]
data.iloc[162:166:, :]
print(data["Type 1"].value_counts(dropna=False))
data.describe()
data.boxplot(column="Attack",by="Legendary")

plt.show()

data[(data["Legendary"]==1) & (data["Attack"]==50)]
data_new = data.head()

data_new
# id_vars = what we dont wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars="Name",value_vars=["Attack","Defense"])

melted
melted.pivot(index="Name",columns="variable",values="value")
data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
data3=data["Attack"].head()

data4=data["Defense"].head()

conc_data_col=pd.concat([data3,data4],axis=1)

conc_data_col
data.dtypes
data["Type 1"]=data["Type 1"].astype("category")

data["Speed"]=data["Speed"].astype("float")

data.dtypes
data["Type 2"].value_counts(dropna=False)
data["Type 2"].fillna("empty",inplace=True)
assert data["Type 2"].notnull().all()     #returns nothing cuz we dont have nan values
data.head()
country=["Spain","France"]

population=["11","12"]

list_label=["country","population"]

list_col=[country,population]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["capital"]=["madrid","paris"]

df
df["income"]=0 #broadcasting entire column

df
data1=data.loc[:,["Attack","Defense","Speed"]]

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="Attack",y="Defense",alpha=0.5,color="k")

plt.show()
data1.plot(kind = "hist",y = "Defense",bins=50,range=(0,250)) #normed hatası veriyor(rectangle has no normed)
fig, axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),ax=axes[0])

data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),ax=axes[1],cumulative=True)

plt.savefig("graph.png")

plt
data = data.set_index("#")

data.head()
print(type(data["HP"]))

print(type(data[["HP"]]))
data.loc[1:10,"HP":"Defense"]

#data.loc[1:10,["HP","Defense"]]
boolean=data.HP>200

data[boolean]
filter1=data.HP>150

filter2=data.Speed>35

data[filter1&filter2]
data.HP[data.Speed<15]
def div(n):

    return n/2

data.HP.apply(div)  #geçici

#data.HP.apply(lambda n:n/2)
data.HP.head()
data["total_power"]=(data.Attack+data["Sp. Atk"])/(data.Defense+data["Sp. Def"])

data.head()
data.index.name="index_name"

data.head()
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.head()
data1=data.set_index(["Type 1","Type 2"])

data1.head(15)
dic={"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df1=pd.DataFrame(dic)

df1
df1.pivot(index="treatment",columns="gender",values="response")
df2=df1.set_index(["treatment","gender"])

df2
df2.unstack(level=0)
df3=df2.swaplevel(0,1)

df3
pd.melt(df1,id_vars="treatment",value_vars=["age","response"])
df1.groupby("treatment").mean()
df1.groupby("treatment").age.mean()