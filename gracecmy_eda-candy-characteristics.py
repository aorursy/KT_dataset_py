import numpy as np

import pandas as pd

pd.set_option("display.max_columns",100)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

params={"axes.titlesize":16,

        "axes.titleweight":"bold",

        "axes.titlelocation":"center"}

plt.rcParams.update(params)
df=pd.read_csv("../input/fivethirtyeight-candy-power-ranking-dataset/candy-data.csv")

df.head()
df.info()
df.describe().T
df.isnull().sum()
plt.figure(figsize=(8,19))

sns.barplot(x=df["winpercent"],y=df["competitorname"],order=df[["competitorname","winpercent"]].sort_values(by="winpercent",ascending=False).iloc[:,0],palette="PuRd_r")

plt.xlabel("Win percentage")

plt.ylabel("")

plt.title("Candy Ranking")
plt.figure(figsize=(10,10))

mask=np.zeros_like(df.corr(),dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(data=df.corr(),annot=True,square=True,mask=mask,cmap="GnBu",linewidths=1,linecolor="white")

plt.title("Candy characteristics correlation")
def grapher(colm,titled,labe,colr):

    fig,axes=plt.subplots(1,2,figsize=(10,4))

    df[colm].value_counts().plot(kind="pie",autopct="%1.1f%%",startangle=90,colors=colr,labels=labe,explode=[0,0.05],ax=axes[0])

    sns.swarmplot(x=df[colm],y=df["winpercent"],palette=colr,ax=axes[1])

    axes[0].set_xlabel("")

    axes[1].set_xlabel("")

    axes[0].set_ylabel("")

    axes[1].set_ylabel("Win percentage")

    axes[1].set_xticklabels(labe)

    plt.suptitle(titled,fontweight="bold",fontsize=16,y=1.05)

    plt.tight_layout()

    plt.show()
grapher("chocolate","Chocolate",["Does not contain \nchocolate","Contains \nchocolate"],["wheat","saddlebrown"])
grapher("fruity","Fruity",["Is not fruity","Is fruity"],["antiquewhite","darkorange"])
grapher("caramel","Caramel",["Does not contain caramel","Contains caramel"],["peachpuff","goldenrod"])
grapher("peanutyalmondy","Peanuts/Peanut Butter/Almonds",["Does not contain \npeanuts/peanut butter \n/almonds","Contains peanuts/ \npeanut butter/almonds"],["burlywood","sienna"])
grapher("nougat","Nougat",["Does not contain nougat","Contains nougat"],["mistyrose","navajowhite"])
grapher("crispedricewafer","Crisped Rice/Wafers/a Cookie Component",["Does not contain \ncrisped rice/wafers \n/a cookie component","Contains crisped rice \n/wafers/a cookie \ncomponent"],["khaki","tan"])
grapher("hard","Texture",["Soft","Hard"],["lightskyblue","lightgreen"])
grapher("bar","Form",["Not a Bar","Bar"],["palevioletred","crimson"])
grapher("pluribus","Sold As",["Individual","One of Many"],["mediumpurple","hotpink"])
def grapher2(coln,colur,titler,labelx,binsize):

    fig=plt.figure(figsize=(6,6))

    ax1=plt.subplot2grid((4,1),(0,0))

    ax2=plt.subplot2grid((4,1),(1,0),rowspan=3)

    sns.distplot(a=df[coln],color=colur,bins=binsize,ax=ax1)

    sns.regplot(x=df[coln],y=df["winpercent"],color=colur,ax=ax2)

    plt.suptitle(titler,fontweight="bold",fontsize=16,y=0.93)

    ax1.set_xlabel("")

    ax1.set_xticklabels([])

    ax1.set_yticklabels([])

    ax2.set_ylim([0,100])

    ax2.set_xlabel(labelx)

    ax2.set_ylabel("Win Percentage")

    plt.show()
grapher2("sugarpercent","lightskyblue","Sugar Percentile","Sugar Percentile",10)
grapher2("pricepercent","mediumaquamarine","Price Percentile","Price Percentile",10)