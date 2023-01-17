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
data = pd.read_csv("../input/data.csv")

data.columns
data.loc[:10,["Name","Jersey Number"]]
data.head()
data.tail()
data.describe()
f,ax = plt.subplots(figsize=(32,32))

sns.heatmap(data.corr(),annot = True,linewidths=.5,fmt='.2f',ax=ax)

plt.show()
data.Overall.plot(kind="line",color="red",label="Overall",linewidth=1,alpha=1,grid=True)

plt.legend()

plt.xlabel("Number of Players")

plt.ylabel("Overall")

plt.title("comparison")

plt.show()
data.Position.unique()
santrafor = data[data.Position == "ST"]
plt.scatter(santrafor.Overall.head(50),santrafor["Jersey Number"].head(50),color="red",alpha=.5,label="Average Jersey Number")

plt.legend()

plt.xlabel("Overall")

plt.ylabel("Jersey Number")

plt.show
data.Overall.plot(kind="hist",bins=100,figsize=(30,30))

plt.show()
filtre1 = data.Overall > 80

filtre2 = data.GKDiving < 10

TheMen = data[filtre1 & filtre2]

TheMen
npTheMen = data[np.logical_and(filtre1,filtre2)]

npTheMen
TheMen.plot(kind="scatter",x = "Overall",y = "Potential",alpha = 1)

plt.clf()
dictionary = {"messi":"argentinian","neymar":"brazil","ronaldo":"portuguese"}

print(dictionary.keys())

print(dictionary.values())
dictionary["neymar"] = "brazilian"

print(dictionary)

dictionary["kane"] = "english"

print(dictionary)

print("neymar" in dictionary.keys())
del dictionary["messi"]

print(dictionary)
dictionary.clear()

print(dictionary)

# if we write del dictionary,it shows error
gk = data[data.Position == "GK"]
DeGea = data[data.Name == "De Gea"]

print(DeGea.Overall > DeGea.Potential)
i = 10

while i != 15 :

    print(i)

    i = i+1
lis = ["muslera","linnes","fehouli","falcao"]

for index,value in enumerate(lis):

    print(index,value)
dic = {"gk":"muslera","rb":"mariano","rw":"feghouli","st":"falcao"}

for key,value in dic.items():

    print(key,value)
for index,value in data[["Overall"]][0:10].iterrows():

    print(index,value)
def tuples():

    "tuple example"

    t = (1,2,3)

    return t

a,b,c = tuples()

print(a,b,c)
messi = data[data.Name == "L. Messi"]

messi
x = messi.Marking

def f():

    x = messi.Composure

    return x

print(x)

print(f())
y = messi.Overall

def f():

    z = y**2

    return z

print(f())
def addition():

    def square():

        z = messi.Potential*2

        return z

    return square()+messi.Overall

print(addition())
def f(*args):

    for each in args:

        print(each)

print(f(1))   
def f(**kwargs):

    for key,value in kwargs.items():

        print(key,value)

f(name = "falcao",country = "colombia",age = 32)
num_list = [3,6,9]

num_liste = map(lambda y:y*2,num_list)

print(list(num_liste))
lis1 = [3,6,9]

lis2 = [2,5,8]

z = zip(lis1,lis2)

print(list(z))
lis3 = [each*2 if each < 5 else each+10 for each in lis1]

lis3
data.columns
data["talent"] = ["talented" if each > 90 else "not" for each in data.Potential]

data.loc[:20,["talent","Potential"]]
data.info()
print(data["Jersey Number"].value_counts(dropna=False))
data.describe()
data.boxplot(column = "Overall")
mini = 100

for each in data.Overall:

    if(each<mini):

        mini=each

    else:

        continue

each
data1 = data.head(5)

data1
melted_data1 = pd.melt(frame=data1,id_vars="Name",value_vars=["Overall","Potential"])

melted_data1
melted_data1.pivot(index="Name",columns="variable",values="value")
data2 = data.head()

data3 = data.tail()
vertical1 = np.vstack((data2,data3))
horizontal1 = np.hstack((data2,data3))
vertical2 = pd.concat([data2,data3],axis=0)
horizontal2 = pd.concat([data2,data3],axis=1)
data.dtypes
data["Age"] = data["Age"].astype("float")
data.dtypes
data.info()
data["Club"].value_counts(dropna=False)
data["Club"].dropna(inplace=True)
assert  data['Club'].notnull().all()
data["Club"].fillna('empty',inplace = True)
assert  data['Club'].notnull().all()
data.head(7)
name = ["messi","ronaldo","neymar"]

overall = [94,94,92]

list_label = ["name","overall"]

list_columns = [name,overall]

zip2list = list(zip(list_label,list_columns))

data_dic = dict(zip2list)

data1 = pd.DataFrame(data_dic)

data1
data1["potential"] = [94,94,93]

data1
data2 = data.loc[:50,["Overall","Potential"]]

data2.plot()
potent = data[data.Potential > 92]

potent
data2.plot(subplots=True)

plt.show()
data2.plot(kind="scatter",x="Potential",y="Overall")

plt.show()
data2.Overall.plot(kind="hist",bins=15,range=(88,95),normed=True)
fig, axes = plt.subplots(nrows=2,ncols=1)

data2.Overall.plot(kind="hist",bins=15,range=(88,95),normed=True,ax=axes[0])

data2.Overall.plot(kind="hist",bins=15,range=(88,95),normed=True,ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt
data4 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

date2list = pd.to_datetime(date_list)

data4["date"] = date2list

data4 = data4.set_index("date")

data4
data4.loc["1992-01-10":"1993-03-16"]
data4.resample("A").mean()
data4.resample("M").first().interpolate("linear")
data4.resample("M").mean().interpolate("linear")