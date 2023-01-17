#IMPORT BASIC LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
matches=pd.read_csv("../input/matches.csv")
matches2017=matches[matches["season"]==2017]

plt.subplots(figsize=(10,10))
matches["player_of_match"].value_counts().head(7).plot.bar()
plt.subplots(figsize=(10,6))
matches['toss_winner'].value_counts().plot.bar(width=0.8)
plt.show()
plt.subplots()
sns.countplot(x="season",hue="toss_decision",data=matches)
plt.show()
b=list(matches2017["toss_winner"].unique())
fig,axes1=plt.subplots(figsize=(20,13))
sns.countplot(x="toss_winner",data=matches2017,hue="winner" ,ax=axes1)

axes1.set_xticklabels(b,rotation=90)
axes1.set_xlabel("")
df=matches[matches['toss_winner']==matches['winner']]
slices=[len(df),(577-len(df))]
labels=['yes','no']
plt.pie(slices,labels=labels,startangle=90,autopct='%1.1f%%',colors=['r','g'])
plt.show()