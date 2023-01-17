import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
e=pd.read_csv("../input/python-seaborn-datas/50ulke.csv")

e.pop("2023")

e.pop("Rankings")

e.pop("2. Rankings")

sns.distplot(e["Growth"])
n=pd.read_csv("../input/python-seaborn-datas/dunyam.csv")

n.dropna(how="any",inplace=True)

sns.set(style="dark",palette="muted",font_scale=2)

sns.distplot(n["Birthrate"],bins=20,kde=False,color="y")

plt.tight_layout()
sns.set(style="darkgrid",palette="muted",font_scale=1.5)

sns.distplot(n["Annual rate of change"],hist=False,rug=True,color="r")

plt.tight_layout()
sns.set(style="white",palette="Blues",font_scale=1.5)

sns.distplot(n["Average age"],hist=False,color="g",kde_kws={"shade":True})

plt.tight_layout()
sns.set(style="whitegrid",palette="Blues",font_scale=1.5)

sns.distplot(n["Population"],color="m")

plt.tight_layout()
sns.set(style="darkgrid",font_scale=1.5)



f, axes = plt.subplots(2,2,figsize=(15,10))



sns.distplot(n["Birthrate"],bins=20,kde=False,color="y",ax=axes[0,0])



sns.distplot(n["Annual rate of change"],hist=False,rug=True,color="r",ax=axes[0,1])



sns.distplot(n["Average age"],hist=False,color="g",kde_kws={"shade":True},ax=axes[1,0])



sns.distplot(n["Population"],color="m",ax=axes[1,1])



plt.tight_layout()
sns.jointplot(n["Birthrate"],n["Average age"],data=n)
sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n)
sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n,kind="hex",color="r")
sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n,kind="reg",xlim=(-2,6),ylim=(0,10),color="r",size=10)
sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n,kind="kde",xlim=(-2,6),ylim=(0,10),color="r",size=8)
sns.kdeplot(e["Per person"])
sns.kdeplot(e["Growth"])
sns.kdeplot(e["Per person"],e["Growth"],shade=True,cmap="Reds")
sns.kdeplot(e["Per person"],e["Growth"],shade=True,cmap="Blues")
sns.pairplot(n,palette="#95a5a6")
sns.pairplot(e,hue="Continent",palette="inferno")
sns.rugplot(e["Growth"],color="y",height=0.2)

sns.kdeplot(e["Growth"],color="r")
sns.boxplot(x="Continent",y="Per person",data=e,width=0.5)

plt.tight_layout()
sns.boxplot(x="Continent",y="Growth",data=e,palette="Set3")
ev=pd.read_csv("../input/python-seaborn-datas/marriage.csv")
sns.boxplot(x="Month",y="Revenue",data=ev,hue="Marriage",palette="PRGn")
sns.set(style="whitegrid")

sns.violinplot(x="Month",y="Revenue",data=ev,hue="Marriage",palette="PRGn",split=True,inner="points")
sns.set(style="whitegrid")

sns.violinplot(x="Month",y="Revenue",data=ev,hue="Marriage",palette="PRGn")
sns.set(style="whitegrid")

sns.violinplot(x="Continent",y="Growth",data=e,palette="Set3",split=True)
sns.barplot(x="Continent",y="Per person",data=e,palette="BuGn_d")
sns.barplot(x="Continent",y="Per person",data=e,palette="RdBu_r")
sns.barplot(x="Continent",y="Per person",data=e,palette="Set1")

sns.despine(left=True,bottom=True)
sns.countplot(x="Continent",data=e)
sns.stripplot(x="Continent",y="Growth",data=e,color="red",jitter=True)
sns.violinplot(x="Continent",y="Growth",data=e)

sns.swarmplot(x="Continent",y="Growth",data=e,color="red")
sns.factorplot(x="Continent",y="Growth",data=e,kind="bar")
sns.factorplot(x="Continent",y="Growth",data=e,kind="violin")
sns.factorplot(x="Per person",y="Growth",data=e,kind="point")
t=pd.read_csv("../input/python-seaborn-datas/titanic.csv")
sns.factorplot(x="Pclass",y="Survived",hue="Sex",size=6,data=t,kind="bar",palette="muted")
sns.factorplot(x="Pclass",y="Survived",hue="Sex",size=6,data=t,kind="violin",palette="muted")
enf=pd.read_csv("../input/python-seaborn-datas/tufe.csv")

enfl=enf.pivot_table(index="Month",columns="Year",values="inflation")
sns.heatmap(enfl,annot=True,linecolor="black",lw=0.5)
sns.clustermap(enfl,figsize=(6,6))
sns.lmplot(x="Per person",y="Growth",data=e)
sns.lmplot(x="Growth",y="Per person",data=e,hue="Continent")
sns.lmplot(x="Growth",y="Per person",data=e,col="Continent")
m=sns.PairGrid(n)

m.map_diag(sns.distplot)

m.map_upper(plt.scatter)

m.map_lower(sns.kdeplot)
k=sns.PairGrid(e)

k.map_diag(sns.distplot)

k.map_upper(plt.scatter)

k.map_lower(sns.kdeplot)
s=pd.read_csv("../input/python-seaborn-datas/marriage.csv")
ss=sns.FacetGrid(data=s,col="Month",row="Marriage")

ss.map(sns.distplot,"Revenue")
ss=sns.FacetGrid(data=s,col="Month",row="Marriage")

ss.map(plt.hist,"Revenue")