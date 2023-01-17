import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)
cf.go_offline()

%matplotlib inline

df=pd.read_csv("../input/Pokemon.csv")
df.head(2)
df=df.replace({'Generation': {1: "G1",2: "G2",3:"G3",4:"G4",5:"G5",6:"G6"}})
df.head(10)
df.info()
df["Type 1"].value_counts()
df["Type 1"].unique()
df["Type 2"].unique()
df_type1=pd.value_counts(df["Type 1"][df["Type 2"].isnull()]).to_frame().reset_index()
df_type1.columns = ['Pokemon_Type','Count']

df_type2=pd.value_counts(df["Type 1"][df["Type 2"].notnull()]).to_frame().reset_index()
df_type2.columns = ['Pokemon_Type','Count']
df_type1
df_type2
labels = list(df_type1["Pokemon_Type"])
sizes = list(df_type1["Count"])

labels1 = list(df_type2["Pokemon_Type"])
sizes1 = list(df_type2["Count"])
explode = (0.1, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1)

fig1, ax1 = plt.subplots(2,1,figsize=(15,15))
ax1[0].set_title("Type 1 Pokemons")
ax1[0].pie(sizes,explode=explode, labels=labels, autopct='%1.1f%%',radius=0.5,
        shadow=True, startangle=90)
ax1[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1[1].set_title("Type 2 Pokemons")
ax1[1].pie(sizes1,explode=explode, labels=labels1, autopct='%1.1f%%',radius=0.5,
        shadow=True, startangle=90)
ax1[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
Type_1_df2=df[df["Type 2"].isnull()]
Type_2_df2=df[df["Type 2"].notnull()]
Type_1_df2.head()
f,ax=plt.subplots(2,1,figsize=(14,12))
ax[0].set_title("Type 1 Pokemons")
ax[1].set_title("Type 2 Pokemons")
plt.xticks(rotation=45)
sns.set_context(context="notebook")
sns.boxplot(x="Type 1", y="Total", data=Type_1_df2,ax=ax[0])
sns.boxplot(x="Type 2", y="Total", data=Type_2_df2,ax=ax[1])

plt.show()
y=["G1","G2","G3","G4","G5","G6"]
Type_2_df_GT=pd.DataFrame({"Generation":["G1","G2","G3","G4","G5","G6"],
                    "Count":[df["Name"][(df["Generation"]==i) & (df["Type 2"].notnull())].count() for i in y],
                    "HP":[df["HP"][(df["Generation"]==i) & (df["Type 2"].notnull())].max() for i in y],
                    "Attack":[df["Attack"][(df["Generation"]==i) & (df["Type 2"].notnull())].max() for i in y],
                    "Defense":[df["Defense"][(df["Generation"]==i) & (df["Type 2"].notnull())].max() for i in y],
                    "Sp. Atk":[df["Sp. Atk"][(df["Generation"]==i) & (df["Type 2"].notnull())].max() for i in y],
                    "Sp. Def":[df["Sp. Def"][(df["Generation"]==i) & (df["Type 2"].notnull())].max() for i in y],
                    "Speed":[df["Speed"][(df["Generation"]==i) & (df["Type 2"].notnull())].max() for i in y]})
Type_1_df_GT=pd.DataFrame({"Generation":["G1","G2","G3","G4","G5","G6"],
                     "Count":[df["Name"][(df["Generation"]==i) & (df["Type 2"].isnull())].count() for i in y],
                     "HP":[df["HP"][(df["Generation"]==i) & (df["Type 2"].isnull())].max() for i in y],
                     "Attack":[df["Attack"][(df["Generation"]==i) & (df["Type 2"].isnull())].max() for i in y],
                     "Defense":[df["Defense"][(df["Generation"]==i) & (df["Type 2"].isnull())].max() for i in y],
                     "Sp. Atk":[df["Sp. Atk"][(df["Generation"]==i) & (df["Type 2"].isnull())].max() for i in y],
                     "Sp. Def":[df["Sp. Def"][(df["Generation"]==i) & (df["Type 2"].isnull())].max() for i in y],
                     "Speed":[df["Speed"][(df["Generation"]==i) & (df["Type 2"].isnull())].max() for i in y]})
Type_1_df_GT.iplot(kind="bar",x="Generation",title="Type 1 Pokemon",xTitle="Generation")
Type_2_df_GT.iplot(kind="bar",x="Generation",title="Type 2 Pokemon",xTitle="Generation")
TYPE2_POK=Type_2_df2[["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed","Generation"]].corr()
TYPE1_POK=Type_1_df2[["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed","Generation"]].corr()
f, ax = plt.subplots(2,1,figsize=(10,14))
ax[0].set_title("Type 1 Pokemons")
ax[1].set_title("Type 2 Pokemons")
sns.heatmap(TYPE1_POK,annot=True,cmap="Blues_r",ax=ax[0])
sns.heatmap(TYPE2_POK,annot=True,cmap="Blues_r",ax=ax[1])
f.suptitle("Correlation coefficient between the Type 1 and Type 2")
plt.plot()
Type_2_df2[["Defense","Speed"]].iplot(kind="scatter",x="Speed",y="Defense",mode="markers",xTitle="Speed",yTitle="Defense")
fig,ax=plt.subplots(2,1,figsize=(10,12))
ax[0].set_title("Type 1 Pokemons")
ax[1].set_title("Type 2 Pokemons")
sns.countplot(x="Generation",data=Type_1_df2,palette="rainbow",ax=ax[0])
sns.countplot(x="Generation",data=Type_2_df2,palette="rainbow",ax=ax[1])
plt.plot()
x=['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric',
       'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice',
       'Dragon', 'Dark', 'Steel', 'Flying']
df_stack=pd.DataFrame({"Pokemons":df["Type 1"].unique(),
               "G1":[len(df[(df["Generation"]=="G1") & (df["Type 1"]==i)]) for i in x],
                "G2":[len(df[(df["Generation"]=="G2") & (df["Type 1"]==i)]) for i in x],
                "G3":[len(df[(df["Generation"]=="G3") & (df["Type 1"]==i)]) for i in x],
                "G4":[len(df[(df["Generation"]=="G4") & (df["Type 1"]==i)]) for i in x],
                "G5":[len(df[(df["Generation"]=="G5") & (df["Type 1"]==i)]) for i in x],
                "G6":[len(df[(df["Generation"]=="G6") & (df["Type 1"]==i)]) for i in x]
               })
df_stack.iplot(kind="bar",x="Pokemons",barmode="stack",title="Pokemons by Type")
Type_1_df2[["HP","Attack","Defense"]].iplot(kind="surface")
Type_1_df2[["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]].iplot(kind="box",title="Type 1 Pokemons")
Type_2_df2[["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]].iplot(kind="box",title="Type 2 Pokemons")
# q=Type 1_df2["Name"][(Type 1_df2["Generation"]=="G3") & Type 1_df2["Legendary"]==True].count()
y=["G1","G2","G3","G4","G5","G6"]
df_Legen_T1=pd.DataFrame({"Generation":y,
   "Legendary_Pokemons":[Type_1_df2["Name"][(Type_1_df2["Generation"]==i) & (Type_1_df2["Legendary"]==True)].count() for i in y],
   "NON_Legendary_Pokemon":[Type_1_df2["Name"][(Type_1_df2["Generation"]==i) & (Type_1_df2["Legendary"]==False)].count() for i in y]})
df_Legen_T2=pd.DataFrame({"Generation":y,
   "Legendary_Pokemons":[Type_2_df2["Name"][(Type_2_df2["Generation"]==i) & (Type_2_df2["Legendary"]==True)].count() for i in y],
   "NON_Legendary_Pokemon":[Type_2_df2["Name"][(Type_2_df2["Generation"]==i) & (Type_2_df2["Legendary"]==False)].count() for i in y]})
df_Legen_T1.iplot(kind="bar",x="Generation",title="Type 1 Pokemons")
df_Legen_T2.iplot(kind="bar",x="Generation",title="Type 2 Pokemons")
Type_1_df2[["HP","Attack"]].iplot(kind="scatter",x="HP",y="Attack",mode="markers",xTitle="HP",yTitle="Attack")
Type_2_df2[["HP","Attack"]].iplot(kind="scatter",x="HP",y="Attack",mode="markers",xTitle="HP",yTitle="Attack")
sns.pairplot(Type_1_df2[["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]])
sns.pairplot(Type_2_df2[["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]])