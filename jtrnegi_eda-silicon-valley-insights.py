#WE WILL USE BASIC TOOLS OF PYTHON ANALYSIS LIKE PANDAS FOR DATA ANALYSIS, AND (SEABORN AND MATPLOTLIB) FOR DATA VISUALISATION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/Reveal_EEO1_for_2016.csv")
df.head(5)
totals = df[df['job_category']=='Totals']
totals['count'] = totals['count'].astype(dtype='int32')
race = totals.groupby(["race"])
g=race.mean()
gg=g.drop("Overall_totals",axis=0,inplace=True)
gg=g.drop("year",axis=1,inplace=True)
g["race"]=g.index
gg=g.drop("race",axis=1)
gg["races"]=g.index

fig,ax=plt.subplots(figsize=(20,13))
sns.barplot(x="races",y="count",data=gg,ax=ax)

a=totals.groupby(["company","gender"])["count"].sum().unstack()
fig,ax= plt.subplots(figsize=(20,13))
a.plot.bar(ax=ax)

