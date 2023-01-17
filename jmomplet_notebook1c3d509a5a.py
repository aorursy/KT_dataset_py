%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



data_path = '../input/Video_Games_Sales_as_at_22_Dec_2016.csv'
df = pd.read_csv(data_path, header=0)

df.head()



dfsub = df[df.Rating.notnull()]
corr = dfsub[['Global_Sales','EU_Sales','NA_Sales','JP_Sales','Critic_Score','User_Score']].corr()

f, ax = plt.subplots(figsize=(7,7))

sns.heatmap(corr,linewidths=0.25,vmax=1.0,square=True,cmap="PuBuGn",linecolor='k',annot=True)
dfsub.groupby(dfsub['Platform']).count()
gen = dfsub[(dfsub['Platform'].isin(['Wii','WiiU']))]

gen.head()
yearlySales = gen.groupby(['Year_of_Release','Platform']).Global_Sales.sum()

yearlySales.unstack().plot(kind='bar',stacked=True,colormap='Blues',grid=False)
dfsub = df[df.Rating.notnull()]

dfsub.head()
dfsub.groupby(df['Rating']).count()
dfsub[dfsub['Rating'].isin(['AO','EC','K-A','RP'])]
mask = np.logical_not(dfsub['Rating'].isin(['AO','EC','K-A','RP']))



dfsub = dfsub[mask]



dfsub['Rating'].unique()
df_group = dfsub[['NA_Sales','EU_Sales','JP_Sales','Global_Sales','Other_Sales']].groupby(dfsub['Rating']).sum()

df_group
f, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

sns.countplot(x="Rating",data=dfsub,palette="Greens_d",ax=ax1)



ax1.set_ylabel("Number of Games")

sns.barplot(x="Rating",y="Global_Sales",data=dfsub,estimator=sum,ax=ax2,ci=None)

ax2.set_ylabel("Total Global Sales (millions)")
f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex='col')

sns.barplot(x="Rating",y="Global_Sales",data=dfsub,ax=ax1)

ax1.set_ylabel("Global Sales")

sns.barplot(x="Rating",y="NA_Sales",data=dfsub,ax=ax2)

ax2.set_ylabel("NA Sales")

sns.barplot(x="Rating",y="EU_Sales",data=dfsub,ax=ax3)

ax3.set_ylabel("EU Sales")

sns.barplot(x="Rating",y="JP_Sales",data=dfsub,ax=ax4)

ax4.set_ylabel("JP Sales")
fig, axs = plt.subplots(1,4)

ratings = ['E','E10+','T','M']



for rating, plot in zip(ratings,range(4)):

    scores = dfsub['Global_Sales'][dfsub['Rating'] == rating]

    ax = scores.plot.hist(bins=2000,ax=axs[plot],figsize=(10,2.5),title="%s Sales" % rating)

    ax.set_xscale('log')
E_Sales = dfsub['Global_Sales'][dfsub['Rating']=='E']

E10_Sales = dfsub['Global_Sales'][dfsub['Rating']=='E10+']

M_Sales = dfsub['Global_Sales'][dfsub['Rating']=='M']

T_Sales = dfsub['Global_Sales'][dfsub['Rating']=='T']
sns.boxplot(data=dfsub,x="Rating",y="Critic_Score")
sns.boxplot(data=dfsub,x="Platform",y="Critic_Score")
dfsub.query("Critic_Score < 20").Name
dfsub = df[df.Year_of_Release.notnull()]

sns.barplot(x="Year_of_Release", y ="Critic_Score", data=dfsub)

plt.xticks(rotation=75)