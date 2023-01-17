import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv("../input/pokemon/Pokemon.csv")

data.head()
data.isnull().sum()
data.shape

data.drop("#",axis=1,inplace=True);
data.info()
data["Type 2"].describe()
categorical=["Name","Type 1","Type 2"]

numerical=data.columns ^ categorical

numerical
data[numerical].describe()
data[categorical].describe().head(4)
cols=list(numerical)

cols.remove("Legendary")



sns.pairplot(data[cols])

fig,ax=plt.subplots(1,1,figsize=(19,6))

df=data.sort_values(by='Type 1', ascending=False)



sns.countplot(data["Type 1"])

ax.set_xticks(range(len(np.unique(data["Type 1"].values))))

ax.set_xticklabels(np.unique(data["Type 1"].values), rotation=45, ha='right')

ax.set_title('Type_1')



fig,ax=plt.subplots(1,1,figsize=(19,6))

sns.countplot(data["Type 2"].dropna())

ax.set_xticks(range(len(np.unique(data["Type 2"].dropna().values))))

ax.set_xticklabels(np.unique(data["Type 2"].dropna().values), rotation=45, ha='right')

ax.set_title('Type_2')



fig=plt.figure(figsize=(10,6))

sns.distplot(data["Attack"],label="Attack")

sns.distplot(data["Sp. Atk"],label="Sp. Atk")

plt.legend()
fig=plt.figure(figsize=(10,6))

sns.distplot(data["Defense"],label="Defense")

sns.distplot(data["Sp. Def"],label="Sp. Def")

plt.legend()
fig,ax=plt.subplots(2,2,figsize=(15,9))

sns.barplot(x=data["Generation"],y=data["Speed"],ax=ax[0][0])

ax[0][0].set_xlabel("Generation")

ax[0][0].set_ylabel("Speed")



sns.barplot(x=data["Generation"],y=data["HP"],ax=ax[1][0])

ax[1][0].set_xlabel("Generation")

ax[1][0].set_ylabel("HP")









df=data.copy()

df=df.groupby(["Generation"]).sum()



labels =range(1,7)

colors = sns.color_palette() 

explode =(0.13,0,0.13,0,0.13,0) 



ax[0][1].pie(100.0*df["Speed"]/sum(df["Speed"]), labels=labels,explode=explode, colors=colors, startangle=90, autopct='%.1f%%', shadow = True); 

ax[1][1].pie(100.0*df["HP"]/sum(df["HP"]), labels=labels,explode=explode, colors=colors, startangle=90, autopct='%.1f%%', shadow = True); 

















df=data.copy()

df=df.groupby(["Generation"]).sum()

fig,ax=plt.subplots(1,2,figsize=(16,7))



labels =range(1,7)

colors = sns.color_palette() 

explode =(0.13,0,0.13,0,0.13,0) 

# fig, ax1 = plt.subplots(figsize = (19,6)) 

patches, texts, autotexts=ax[0].pie(100.0*df["Total"]/sum(df["Total"]), labels=labels,explode=explode, colors=colors, startangle=90, autopct='%.1f%%', shadow = True) 

[ i.set_fontsize(21) for i in texts]



ax[0].set_title("TOTAL POWER OF GENERATIONS")

explode =(0,0,0.13,0,0.13,0) 



patches, texts, autotexts=ax[1].pie(100.0*df["Legendary"]/sum(df["Legendary"]), labels=labels,explode=explode, colors=colors, startangle=90, autopct='%.1f%%', shadow = True) 

ax[1].set_title("Legendary OF GENERATIONS")

[ i.set_fontsize(21) for i in texts]



plt.tight_layout() 

plt.show()

# df
fig,ax=plt.subplots(1,1,figsize=(7,4))



cols=list(numerical)

cols.remove("Generation")

cols.remove("Legendary")

cols.remove("Total")

sns.boxplot(data=data[cols]);

plt.tight_layout()

# cols
fig,ax=plt.subplots(3,2,figsize=(19,12))

index=1

for i in range(3):

  for j in range(2):

    df=data.loc[data["Generation"]==index]

    sns.boxplot(data=df[cols],ax=ax[i][j])

    ax[i][j].set_title("Generation "+str(index))

    ax[i][j].set_ylim([0,300])

    index+=1

plt.tight_layout()
df=data.groupby(["Generation"])

temp=df["Total"].max()

best=[]

index=1

for i in [1,3,5]:

  best.append(data.loc[(data["Total"]==temp[i]) & (data["Generation"]==i)].head(1))

  

top3_total=pd.concat([data.iloc[best[0].index[0]],data.iloc[best[1].index[0]],data.iloc[best[2].index[0]]],axis=1).transpose()

top3_total
