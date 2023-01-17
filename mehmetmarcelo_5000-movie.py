# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

df1.head()
data=df1.drop(["genres","homepage","id","keywords","spoken_languages","production_companies"],axis=1)

data.head()
data.info()
data.describe()
data.columns
len(data.columns)
data.corr()
f,ax=plt.subplots(figsize=(11,11))

sns.heatmap(data.corr(), annot=True, lw=.7,fmt=".2f",ax=ax)

plt.show()
data.head(10)
data.tail()
data.budget.plot(kind="line",label="budget",color="r",lw=.8,ls="--",alpha=.7,grid=True,figsize=(13,13))

plt.legend(loc="upper right")

plt.xlabel("index",color="black")

plt.ylabel("budget",color="black")

plt.title("Budget")

plt.show()
data.popularity.plot(kind="line",label="popularity",color="b",lw=.8,ls="--",alpha=.7,grid=True,figsize=(13,13))

plt.legend(loc="upper right")

plt.xlabel("index",color="k")

plt.ylabel("popularity",color="k")

plt.title("Popularity")

plt.show()
data.plot(kind="scatter",x="vote_count",y="revenue",alpha=.5,color="g",grid=True,figsize=(13,13))

plt.xlabel("vote_count")

plt.ylabel("revenue")

plt.title("vote_count vs. revenue")

plt.show()
data.vote_average.plot(kind="hist",bins=100,grid=True,figsize=(11,11))

plt.title("vote_average")

plt.show()
data.head()
dictionary={"Avatar":7.2,"Pirates of the Caribbean: At World's End":6.9,"Spectre":6.3,"The Dark Knight Rises":7.6,"John Carter":6.1}

dictionary
data.tail()
dictionary["My Date with Drew"]=6.3

dictionary
#sözlüğün keyleri:

dictionary.keys()
#sözlüğün valueları:

dictionary.values()
#tüm sözlük:

dictionary.items()
print("Avatar" in dictionary)
for key,value in dictionary.items():

    print("Key:",key,",\tValue:",value)

print("")
#sözlüğümüzün içini temizleyerek devam edelim:

dictionary.clear()

dictionary
data.head()
v_a=data["vote_average"]>8

data[v_a]
len(data[v_a])
data[(data["budget"]>100000000) & (data["vote_average"]>8) &(data["popularity"]>120)]
liste1=[185,165,160]

for i in liste1:

    print("i is:",i)

print("")
for index,value in enumerate(liste1):

    print("index:",index,",\tvalue:",value)
def tuple_movie():

    """return defined t tuple"""

    t=tuple(liste1)

    return t

a,b,c=tuple_movie()

print("The Dark Knight:",a,"\nInterstellar",b,"\nInception",c)
vote=8.1

def f():

    vote=8.2

    return vote

print(vote)
print(f())
Inception=8.1

def g():

    The_Dark_Knight=Inception +0.1

    return The_Dark_Knight

print(g())
def h(avatar,The_Dark_Knight=8.2,Interstellar=8.1,Inception=8.1):

    total=avatar+The_Dark_Knight+Interstellar+Inception

    return total

h(7)

#avatar=7 dedik;
h(7,3)

#avatar=7,The_Dark_Knight=3 dedik;
h(7,Inception=6)
def j(*args):

    for i in args:

        print(i)

j(5,7,74,7,6,272,0,-42)
j("\n\n")
def k(**kwargs):

    """print key and value of dictionary"""

    for key,value in kwargs.items():

        print("key:",key,",\tvalue:",value)

k(The_Dark_Knight=8.2,Interstellar="8.1",Inception=8.1)
y=map(lambda x:x**2,liste1)

print(list(y))
film="The Dark Knight"

it=iter(film)

print(next(it))
print(*it)
l1=[185,165,160]

l2=[8.2,8.1,8.1]

z=zip(l1,l2)

print(z)

z_list=list(z)

print(z_list)
unzip=zip(*z_list)

unlist1,unlist2=list(unzip)

print(unlist1)

print(unlist2)

print(type(unlist1))
print(list(unlist1))

print(list(unlist2))
unlist1
print(type(list(unlist1)))
#ama kalıcı olaraak list olmaz,tuple olarak kalır.

print(unlist1)
data.head()
vote_level=sum(data.vote_average)/len(data.vote_average)

data["vote_level"]=["high" if i > vote_level else "on average" if i==vote_level else "low" for i in data.vote_average]

data.loc[:10,["vote_level","vote_average"]]
data.head()
data.vote_level.value_counts(dropna=False)
len(data.vote_level)
print(data.original_language.value_counts(dropna=False))
data[data.original_language.str.contains("tr")]
data.boxplot(column="budget",by="vote_level",figsize=(11,11))

plt.show()
data_new=data.head()

data_new
data_new.append(data[data.original_language.str.contains("tr")],ignore_index=True)

#ignore_index=True diyerek yeni indexini 3521 değil de 5 almasını sağladık.
melted=pd.melt(frame=data_new,id_vars="title",value_vars=["release_date","vote_average","runtime"])

melted
melted2=pd.melt(frame=data_new,id_vars=["title","overview"],value_vars=["release_date","vote_average","runtime"])

melted2
melted3=pd.melt(frame=data_new,id_vars="title",value_vars=["release_date","vote_average"],value_name="details")

melted3
melted
melted.pivot(index="title",columns="variable",values="value")
data1=data.head()

data2=data[data.original_language.str.contains("tr")]

data3=data.tail()

concat_data_row=pd.concat([data1,data2,data3],axis=0)

concat_data_row
data1=data.head()

data2=data[data.original_language.str.contains("tr")]

data3=data.tail()

concat_data_row=pd.concat([data1,data2,data3],axis=0,ignore_index=True)

concat_data_row
film=["Avatar","Spectre","Kurtlar Vadisi Irak"]

vote=[7.2,6.3,4.3]

language=["en","en","tr"]

list_label=["film","vote","language"]

list_col=[film,vote,language]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df1=pd.DataFrame(data_dict)

df1
df1["year"]=["2009","2015","2006"]

df1
df1["income"]=0

df1
data.head()
#1. Plotting All Data

data2=data.loc[:,["popularity","runtime"]]

data2.plot()
#2. subplots

data2.plot(subplots=True)

plt.show()
#3. scatter plot

data2.plot(kind="scatter",x="popularity",y="runtime",grid=True)

#plt.grid(True) # 2 türlü de çalışıyor.

plt.show()
#4. hist plot - hiçbir şey girmeden

data2.plot(kind="hist")

plt.show()
#5. hist plot - klasik bir hist plot

data2.plot(kind="hist",y="runtime",bins=50,range=(0,300),normed=True)
#6. hist plot - cumulative and noncumulative

fig,axes=plt.subplots(nrows=2,ncols=1)

data2.plot(kind="hist", y="runtime", bins=75, range=(0,250), normed=True,ax=axes[0])

data2.plot(kind="hist", y="runtime", bins=75, range=(0,250), normed=True,ax=axes[1],cumulative=True)

plt.grid(True)

plt.savefig("graph.png")

plt.show()
#6. hist plot - cumulative and noncumulative - different dimensions

fig,axes=plt.subplots(nrows=3,ncols=2,figsize=(10,10))

data2.plot(kind="hist", y="runtime", color="b", bins=150, range=(0,250), normed=True,ax=axes[0][0])

data2.plot(kind="hist", y="runtime", color="b", bins=150, range=(0,250), normed=True,ax=axes[0][1],cumulative=True)

data2.plot(kind="hist", y="popularity", color="orange", bins=150, range=(0,150),normed=True,ax=axes[1][0])

data2.plot(kind="hist", y="popularity", color="orange", bins=150, range=(0,150),normed=True,ax=axes[1][1],cumulative=True)

data2.plot(kind="scatter",x="popularity", color="r",y="runtime",grid=True,ax=axes[2][0])

data.budget.plot(kind="line",label="budget", color="r",lw=.8,ls="-",alpha=1,grid=True,ax=axes[2][1])

plt.legend("budget",loc="upper left")

plt.savefig("graph2.png")

plt.show()
data.head()
#1. data column'ını index column'ı yapalım:

data3=data.copy()

data3=data3.set_index("release_date")

data3.head()
print(data3.loc["2012-07-16"])
print(data3.loc["2015-10-26":"2012-07-16"])
data.head()
data6=data.copy()

data6["release_date"]=pd.to_datetime(data6["release_date"])

data6.head()
data6=data6.set_index("release_date")
data6.resample("A").mean()
data6.resample("M").mean()
data6.resample("M").first().interpolate("linear")
data6.resample("M").mean().interpolate("linear")
data.loc[:10,"title":"vote_count"]
data.loc[:10,["title","vote_count"]]
data.loc[10::-1,"title":"vote_count"]
data.loc[:10,"vote_average":]
data.head()
filter1=data.original_language=="fr"

data[filter1].head()
filter2=data.runtime>135

data[filter2].head()
filter3=data.vote_level=="high"

data[filter3].head()
data[filter1 & filter2 & filter3]
data.runtime[data.original_language.str.contains("tr")]
print(data.index.name)
data.index.name="index"

data
data.index=range(1,4804)

data
data7=data.set_index(["original_title","original_language"])

data7.head()
def div(n):

    return 2*n/5

data.vote_average.apply(div)
data.vote_average.apply(lambda n:2*n/5)
data["vote_by_5"] = data.vote_average.apply(lambda n:2*n/5)

data.head()
data["profit"]=data.revenue - data.budget

data.head()
data.profit[1]
data.profit.head()
dic1={"film":["avatar","spectre","john carter","kurtlar vadisi ırak"],"vote":["7.2","6.3","6.1","4.3"],

      "runtime":[162,148,132,122],"release_date":["2009-12-10","2015-10-26","2012-03-07","2006-02-03"]}

data8=pd.DataFrame(dic1)

data8
data8.pivot(index="film",columns="release_date",values="vote")
data8.release_date=pd.to_datetime(data8.release_date)

data8=data8.set_index("release_date")
data8.resample("A").mean().interpolate("linear")
data8.unstack(level=0)
data8.groupby("vote").mean()
data8.groupby("vote").runtime.max()