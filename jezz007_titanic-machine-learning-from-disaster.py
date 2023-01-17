import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#importing seaborn and matplotlib for plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ds=pd.read_csv("../input/train.csv")

ds.head()
data=ds.iloc[:,[0,1,2,4,5,6,7,-1]]
data.head()
p=data.groupby(["Survived","Pclass"])["PassengerId"].count().rename("Count").reset_index()

p["ClassPercentage"]=p["Count"]/p.groupby("Pclass")["Count"].transform("sum")

f=plt.figure()
ax=f.add_subplot(111)
ax.set(xlabel="Pclass",ylabel="Percentage",xlim=(0,4),ylim=(0,1),title="Class vs Survival")
ax.set_xticks([1,2,3])
ax.set_xticklabels(["Class 1","Class 2","Class 3"])
ax.bar([.8,1.8,2.8],p[p.Survived==1]["ClassPercentage"],color=["g"],label="Survived",width=.4)
ax.bar([1.2,2.2,3.2],p[p.Survived==0]["ClassPercentage"],color=["r"],label="Not Survived",width=.4)
plt.legend()
plt.show()
p=data.groupby(["Survived","Sex"])["PassengerId"].count().rename("Count").reset_index()
p["%bySex"]=p.Count*100/(p.groupby("Sex")["Count"].transform("sum"))
p
f,ax=plt.subplots(1,2,figsize=(8,6))
ax[0].set(title="Female")
ax[0].pie(p[p.Sex=="female"]["%bySex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
ax[1].set(title="Male")
ax[1].pie(p[p.Sex=="male"]["%bySex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
plt.legend()
plt.show()
p=data.groupby(["Survived","Pclass","Sex"])["PassengerId"].count().rename("Count").reset_index()
p["%byClass_Sex"]=p["Count"]/p.groupby(["Pclass","Sex"])["Count"].transform("sum")
p
with plt.style.context("ggplot"):
    f,ax=plt.subplots(3,2,figsize=(8,18))

    ax[0,0].set(title="Female")
    ax[0,0].pie(p[(p.Sex=="female")&(p.Pclass==1)]["%byClass_Sex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
    ax[0,1].set(title="Male")
    ax[0,1].pie(p[(p.Sex=="male")&(p.Pclass==1)]["%byClass_Sex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
    ax[0,0].set_ylabel("Class 1")

    ax[1,0].set(title="Female")
    ax[1,0].pie(p[(p.Sex=="female")&(p.Pclass==2)]["%byClass_Sex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
    ax[1,1].set(title="Male")
    ax[1,1].pie(p[(p.Sex=="male")&(p.Pclass==2)]["%byClass_Sex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
    ax[1,0].set_ylabel("Class 2")

    ax[2,0].set(title="Female")
    ax[2,0].pie(p[(p.Sex=="female")&(p.Pclass==3)]["%byClass_Sex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
    ax[2,1].set(title="Male")
    ax[2,1].pie(p[(p.Sex=="male")&(p.Pclass==3)]["%byClass_Sex"],colors=["r","g"],labels=p["Survived"].unique(),autopct="%1.1f%%",)
    ax[2,0].set_ylabel("Class 3")

    plt.legend()
    plt.suptitle("Survival Ratio vs Gender among different classes")
    plt.show()
p=data.groupby(["Survived","Embarked"])["PassengerId"].count().rename("Count").reset_index()
p["Emb%"]=p["Count"]/p.groupby("Embarked")["Count"].transform("sum")
p
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.bar([.8,1.8,2.8],list(p.loc[p["Survived"]==0]["Count"]),width=.4,label="died",color="r")
ax.bar([1.2,2.2,3.2],list(p.loc[p["Survived"]==1]["Count"]),width=.4,label="Survived",color="g")
ax.set_xticks([1,2,3])
ax.set_xticklabels(["Cherbourg","Queenstown","Southampton"])
plt.legend()
plt.show()

p=data.groupby(["Embarked","Pclass","Survived"]).count()
p["ClassEmb%"]=p["PassengerId"]*100/p.groupby(["Embarked","Pclass"])["PassengerId"].transform("sum")
p["Emb%"]=p["PassengerId"]*100/p.groupby(["Embarked"])["PassengerId"].transform("sum")
p.rename(columns={"PassengerId":"Count"},inplace=True)
p
with plt.style.context("ggplot"):
    f=plt.figure(figsize=(18,12))
    g=gridspec.GridSpec(3,3)
    plt.suptitle("Survival Count vs Pclass among diff Embarkment")
  
    #Row 1
    ax0=plt.subplot(g[:3,:-2])
    ax0.set(title="Cherbourg")
    ax0.set_xticks([1,2,3])
    ax0.set_xticklabels(["Class 1","Class 2","Class 3"])
    ax0.set_yticks(np.arange(0,300,40))
    ax0.set_ylabel("Count")
    ax0.set_xlabel("Pclass")
    ax0.bar([.8,1.8,2.8],p.xs(["C",0],level=[0,2])["Count"],width=.4,label="died",color="r")
    ax0.bar([1.2,2.2,3.2],p.xs(["C",1],level=[0,2])["Count"],width=.4,label="Survived",color="g")
    ax0.legend()
    
    ax01=plt.subplot(g[0,-1])
    ax01.set_title("Class 1")
    ax01.pie(p.xs(["C",1],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    ax02=plt.subplot(g[1,-1])
    ax02.set_title("Class 2")
    ax02.pie(p.xs(["C",2],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    ax03=plt.subplot(g[2,-1])
    ax03.set_title("Class 3")
    ax03.pie(p.xs(["C",3],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    plt.tight_layout()
    plt.show()
    
    
    
   
   
#Row 2
with plt.style.context("ggplot"):
    f1=plt.figure(figsize=(18,12))
    g=gridspec.GridSpec(3,3)
    plt.suptitle("Survival Count vs Pclass among diff Embarkment")
   
    ax1=plt.subplot(g[:3,:-2])
    ax1.set(title="Queenstown")
    ax1.set_xticks([1,2,3])
    ax1.set_xticklabels(["Class 1","Class 2","Class 3"])
    ax1.set_yticks(np.arange(0,300,40))
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Pclass")
    ax1.bar([.8,1.8,2.8],p.xs(["Q",0],level=[0,2])["Count"],width=.4,label="died",color="r")
    ax1.bar([1.2,2.2,3.2],p.xs(["Q",1],level=[0,2])["Count"],width=.4,label="Survived",color="g")
    ax1.legend()
    
    ax11=plt.subplot(g[0,-1])
    ax11.set_title("Class 1")
    ax11.pie(p.xs(["Q",1],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    ax12=plt.subplot(g[1,-1])
    ax12.set_title("Class 2")
    ax12.pie(p.xs(["Q",2],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    ax13=plt.subplot(g[2,-1])
    ax13.set_title("Class 3")
    ax13.pie(p.xs(["Q",3],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    plt.tight_layout()
    plt.show()
 #Row 3
with plt.style.context("ggplot"):    
    f2=plt.figure(figsize=(18,12))
    g=gridspec.GridSpec(3,3)
    plt.suptitle("Survival Count vs Pclass among diff Embarkment")
    
    
    ax2=plt.subplot(g[:3,:-2])
    ax2.set(title="Southampton")
    ax2.set_xticks([1,2,3])
    ax2.set_xticklabels(["Class 1","Class 2","Class 3"])
    ax2.set_yticks(np.arange(0,300,40))
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Pclass")
    ax2.bar([.8,1.8,2.8],p.xs(["S",0],level=[0,2])["Count"],width=.4,label="died",color="r")
    ax2.bar([1.2,2.2,3.2],p.xs(["S",1],level=[0,2])["Count"],width=.4,label="Survived",color="g")

    ax2.legend()
    
    
    ax21=plt.subplot(g[0,-1])
    ax21.set_title("Class 1")
    ax21.pie(p.xs(["S",1],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    ax22=plt.subplot(g[1,-1])
    ax22.set_title("Class 2")
    ax22.pie(p.xs(["S",2],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    ax23=plt.subplot(g[2,-1])
    ax23.set_title("Class 3")
    ax23.pie(p.xs(["S",3],level=[0,1])["ClassEmb%"],colors=["r","g"],labels=["Died","Survived"],autopct="%1.1f%%")
    plt.tight_layout()
    plt.show()   
data["Age"].fillna(-50).plot.hist(bins=np.arange(0,90,5),figsize=(8,6),density=True)

data.loc[data["Survived"]==1]["Age"].reset_index()["Age"].fillna(-50).plot.hist(histtype="bar",density=True,color="g", bins=np.arange(0,90,5),figsize=(8,6))
data.loc[data["Survived"]==0]["Age"].reset_index()["Age"].fillna(-50).plot.hist(histtype="bar",density=True,color="r", bins=np.arange(0,90,5),figsize=(8,6))
