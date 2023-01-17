# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Concrete_Data_Yeh.csv')
df.head()
df.info()
df.corr()
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()
data=df.copy()
data['cement']=data['cement']/max(data['cement'])
data['csMPa']=data['csMPa']/max(data['csMPa'])
                  
data.head()
g=sns.jointplot(data.cement,data.csMPa,kind="kde",size=7)
plt.savefig('graph.png')
plt.show()
g=sns.jointplot("cement","csMPa",data=data,size=5,color="r")
df.cement.mean()
cols=[i for i in data.columns if i not in "csMPa"]
length=len(cols)
cs = ["b","r","g","c","m","k","lime","c"]
fig = plt.figure(figsize=(13,25))
for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(4,2,j+1)
    ax=sns.distplot(df[i],color=k,rug=True)
    ax.set_facecolor("w")
    plt.legend(loc="best")
    plt.axvline(df[i].mean(),linestyle="dashed",label="mean",color="k")

    plt.title(i,color="navy")
    plt.xlabel("")
plt.figure(figsize=(13,6))
sns.distplot(df["csMPa"],color="b",rug=True)
plt.axvline(df["csMPa"].mean(),
            linestyle="dashed",color="k",
            label="mean",linewidth=2)
plt.legend(loc="best",prop={"size":14})
plt.title("Compressive strength distription")
plt.show()
sns.pairplot(df,markers="h")
plt.show()
data.columns.value_counts().index
sns.lmplot(x="cement",y="water",data=df)
plt.show()
cols=[i for i in df.columns if i not in 'csMPa']
length=len(cols)
plt.figure(figsize=(13,27))
for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(4,2,j+1)
    sns.kdeplot(df[i],
               df["csMPa"],
               cmap="hot",
               shade=True)
    plt.title(i+" & compressive_strength",color="navy")
cols = ['cement', 'slag', 'flyash', 'water', 'superplasticizer',
       'coarseaggregate', 'fineaggregate', 'age', 'csMPa'] 



length = len(cols)

plt.figure(figsize=(12,25))

for i,j in itertools.zip_longest(cols,range(length)):
    
    plt.subplot(3,3,j+1)
    ax = sns.swarmplot( y = df[i],color="blue")
    ax.set_facecolor("w")
    ax.set_ylabel("")
    ax.set_title(i,color="navy")
    plt.subplots_adjust(wspace = .3)
def label(df):
    if df["age"] <= 30:
        return "1 month"
    if df["age"] > 30 and df["age"] <= 60 :
        return "2 months"
    if df["age"] > 60 and df["age"] <= 90 :
        return "3 months"
    if df["age"] > 90 and df["age"] <= 120 :
        return "4 months"
    if df["age"] > 120 and df["age"] <= 150 :
        return "5 months"
    if df["age"] > 150 and df["age"] <= 180 :
        return "6 months"
    if df["age"] > 180 and df["age"] <= 210 :
        return "7 months"
    if df["age"] > 210 and df["age"] <= 240 :
        return "8 months"
    if df["age"] > 240 and df["age"] <= 270 :
        return "9 months"
    if df["age"] > 270 and df["age"] <= 300 :
        return "10 months"
    if df["age"] > 300 and df["age"] <= 330 :
        return "11 months"
    if df["age"] > 330 :
        return "12 months"
df["age_months"] = df.apply(lambda df:label(df) , axis=1)
plt.figure(figsize=(12,5))
order = ['1 month','2 months', '3 months','4 months','6 months','9 months', '12 months']
ax = sns.countplot(df["age_months"],
                   order=order,linewidth=2,
                   edgecolor = "k"*len(order),
                   palette=["w"])
ax.set_facecolor("royalblue")
plt.title("age distribution in months")
plt.grid(True,alpha=.3)
plt.show()
age_mon=df.groupby("age_months")["csMPa"].describe().reset_index()
age_mon
age_mon=df.groupby("age_months")["csMPa"].describe().reset_index()

order  = ['1 month','2 months', '3 months','4 months','6 months','9 months', '12 months']
cols   = [ 'mean', 'std' , 'min' , 'max']
length=len(cols)
cs     = ["b","orange","white","r"] 

fig=plt.figure(figsize=(13,15))
for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(4,1,j+1)
    ax=sns.pointplot("age_months",i,data=age_mon,
                    order=order,
                    markers="H",
                    linestyles="dotted",color=k)
    plt.subplots_adjust(hspace=.5)
    ax.set_facecolor("k")
    plt.title(i+"-compressive strength by months",color="navy")
fig=plt.figure(figsize=(13,8))
ax=fig.add_subplot(111)
plt.scatter(df["water"],df["cement"],
           c=df["csMPa"],s=df["csMPa"]*3,
           linewidth=1,edgecolor="k",cmap="viridis")
ax.set_facecolor("w")
ax.set_xlabel("water")
ax.set_ylabel("cement")
lab=plt.colorbar()
plt.title("scatter plot between cement and water")
plt.grid(True,alpha=.3)
plt.show()
df.head()
fig=plt.figure(figsize=(13,8))
ax=fig.add_subplot(111)
plt.scatter(df["coarseaggregate"],df["fineaggregate"],c=df["csMPa"],s=df["csMPa"]*3,
           linewidth=1,edgecolor="k",cmap="viridis_r")
ax.set_facecolor("w")
ax.set_xlabel("coarseaggregate")
ax.set_ylabel("fineaggregate")
lab = plt.colorbar()
lab.set_label("compressive_strength")
plt.title("scatter plot between fine_agg and coarse_agg")
plt.grid(True,alpha=.3)
plt.show()