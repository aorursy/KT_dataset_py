import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dt = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
dt.info()
dt["Type 1"] = dt["Type 1"].astype("category")
dt.dtypes
dt["Type 2"].value_counts(dropna =False)
dt["Type 2"].notnull().all()
dt1=dt

dt1["Type 2"].dropna(inplace=True)
dt["Type 2"].fillna('empty',inplace = True)
dt["Type 2"].notnull().all()
dt.info()
dt.dtypes
dt.shape
dt.columns
dt.head(10)
dt.tail(10)
dthead = dt.head(5)

erit = pd.melt(frame= dthead, id_vars="Name", value_vars=["Attack","Defense","Speed"])

erit
erit.pivot(index="Name", columns= "variable", values="value")
dt1=dt.head(3)

dt2=dt.tail(3)

concated= pd.concat([dt1,dt2], axis=0, ignore_index =True)

concated
dt3=dt["Attack"].head(3)

dt4=dt["Speed"].head(3)

concated1= pd.concat([dt3,dt4], axis=1)

concated1
dt.describe()
dt.boxplot(column="Attack")

plt.show()
dt.corr()
print(dt['Type 2'].value_counts(dropna =False))
dates = ["2005-10-15","2009-05-16","2012-12-12"]

datatime = pd.to_datetime(dates)

mydata = dt.tail(3)

mydata["Date"] = datatime

mydata = mydata.set_index("Date")

mydata
print(mydata.loc["2005-10-15":"2012-12-12"])
mydata.resample("A").mean()
mydata.resample("A").mean().interpolate("linear")
dt = dt.set_index("#")

dt
dt.Speed[33]
dt.loc[7,"Attack"]
dt[["Speed","Attack"]]
dt.loc[3:6,"HP":"Speed"]
def f(x):

    return x/2

dt.Attack.apply(f)
dt["Flex"] = dt.Speed + dt.Defense

dt
print(dt.index.name)

dt.index.name = "IndexNo"

dt.head(3)
sözlük = {"Film" : "madagaskar penguenleri", "Çizgi Film" : "Süngerbob", "Oyun" : "Battlefield 4"}

print(sözlük.keys())

print(sözlük.values())
sözlük["Film"] = "Maskeli Beşler"

print(sözlük)
sözlük["Yemek"] = "Kızarmış Tavuk"

print(sözlük)
del sözlük["Film"]

print(sözlük)
print("Oyun" in sözlük)
sözlük.clear()

print(sözlük)
dt[(dt["Attack"] > 150) & (dt["Defense"] < 100)]
j = dt["Attack"] > 150

b = dt["Defense"] < 100

dt[j & b]
dt.Defense[dt.Attack< 20]
sözlük = {"Film" : "madagaskar penguenleri", "Çizgi Film" : "Süngerbob", "Oyun" : "Battlefield 4"}

for söz, anlam in sözlük.items():

    print(söz, ": " ,anlam)
for index, anlam in dt[["Defense"]][0:11].iterrows():

    print(index,": ",anlam)
def deneme():

    a = (1,2,3)

    return a

x,y,z = deneme()

print(x,y,z)
def f(*args):

    for i in args:

        print(i)

f(1,11,12,2551,63951)
def f(**kwargs):

    for index,anlam in kwargs.items():

        print(index,": ",anlam)

f(Film = "madagaskar penguenleri", ÇizgiFilm = "Süngerbob", Oyun = "Battlefield 4")

    
toplama = lambda x: x+5

print(toplama(10))

print("")

sayılarlatoplama = lambda a,b,c: a+b+c

print(sayılarlatoplama(2,5,6))
deneme = (6,7,8)

karesinial = map(lambda t: t**2, deneme)

print(tuple(karesinial))
list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

z_list = list(z)

print(z_list)
unzip = zip(*z_list)

unlist1,unlist2=unzip

print(unlist1)

print(unlist2)
deneme = (2,3,4)

deneme1 = (i**5 for i in deneme)

print((list(deneme1)))
deneme = (2,3,4)

deneme1 = (i**3 if i == 2 else i-5 if i == 3 else i+5 for i in deneme)

print((list(deneme1)))
ortalama = sum(dt.Attack)/len(dt.Attack)

dt["Attack_Level"] = ["high damage" if i > ortalama else "low damage" for i in dt.Attack]

dt.loc[:20,["Attack_Level","Attack"]]
ürün = ["Süt","Peynir"]

kcal = ["125","102"]

listlabels = ["Madde","Enerji"]

listcolumns = [ürün,kcal]

birleştir = list(zip(listlabels,listcolumns))

print(birleştir)

sözlükle = dict(birleştir)

print(sözlükle)

datam = pd.DataFrame(sözlükle)

datam
datam["Zararlılık"] = ["Yararlı","Yararlı"]

datam
datam["Marka"]= "NaN"

datam
dt2 = dt.copy()

dt2.index = range(100,900,1)

dt2.head()
dt3 = dt.set_index(["Type 1","Type 2","HP"])

dt3.head(10)
dic = {"Mezhep":["Katolik","Protestan","Ortadoks"],"Yaş":["20","10","18"],"Irk":["İngiliz","Alman","Türk"]}

df = pd.DataFrame(dic)

df
df1 = df.set_index(["Yaş","Mezhep"])

df1
df2 = df1.swaplevel(0,1)

df2
df1.unstack(level=0)
df1.unstack(level=1)
dt.groupby("HP").mean()
dt.groupby("HP")[["Defense"]].mean()