import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

from scipy import stats

import seaborn as sns



master=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",index_col="datetime",parse_dates=True)

labels=["NE","SE","SW","NW"]

master["dir_o_l"]=pd.cut(master.dir_o[master.dir_o!=-1], len(labels),labels=labels)

master["dir_4K_l"]=pd.cut(master.dir_4K[master.dir_o!=-1], len(labels),labels=labels)

master["dir_36K_l"]=pd.cut(master.dir_36K[master.dir_o!=-1], len(labels),labels=labels)

table_4K=pd.crosstab(master.dir_o_l, master.dir_4K_l, margins=True,)

table_36K=pd.crosstab(master.dir_o_l, master.dir_36K_l, margins=True,)

table_two_model=pd.crosstab(master.dir_4K_l, master.dir_36K_l, margins=True,)

fig, axs = plt.subplots(3,figsize = (15,15))









sns.heatmap(table_4K,annot=True,cmap="YlGnBu",ax=axs[0],fmt='.0f')

sns.heatmap(table_36K,annot=True,cmap="YlGnBu",ax=axs[1],fmt='.0f')

sns.heatmap(table_two_model,annot=True,cmap="YlGnBu",ax=axs[2],fmt='.0f')







fig, axs = plt.subplots(3,figsize = (15,15))



master["dir_o_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=axs[0])

master["dir_4K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=axs[1])

master["dir_36K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=axs[2])
g = sns.jointplot("dir_4K", "dir_o", data=master[master.dir_o!=-1].sample(500), kind="reg",

                  xlim=(0, 360), ylim=(0, 360),color="b",).annotate(stats.pearsonr)





((master["dir_4K"]-master["dir_o"])[master.dir_o!=-1]).describe()

g = sns.jointplot("dir_36K", "dir_o", data=master[master.dir_o!=-1].sample(500), kind="kde",

                  xlim=(0, 360), ylim=(0, 360),color="b",).annotate(stats.pearsonr)
((master["dir_36K"]-master["dir_o"])[master.dir_o!=-1]).describe()



g = sns.jointplot("dir_36K", "dir_4K", data=master.sample(500), kind="scatter",

                  xlim=(0, 360), ylim=(0, 360),color="b",).annotate(stats.pearsonr)
(master["dir_36K"]-master["dir_4K"]).describe()

sns.boxplot(master['mod_o'])
g = sns.jointplot("mod_36K", "mod_o", data=master.sample(500), kind="reg",

                  color="b",).annotate(stats.pearsonr)
(master["mod_36K"]-master["mod_o"]).describe()

(master["mod_36K"]-master["mod_o"]).plot(kind="box",grid=True,notch=True)
g = sns.jointplot("mod_4K", "mod_o", data=master.sample(500), kind="reg",

                  color="b",).annotate(stats.pearsonr)
(master["mod_4K"]-master["mod_o"]).describe()

(master["mod_4K"]-master["mod_o"]).plot(kind="box",grid=True,notch=True)
g = sns.jointplot("mod_4K", "mod_36K", data=master.sample(500), kind="reg",

                  color="b",).annotate(stats.pearsonr)
(master["mod_4K"]-master["mod_36K"]).describe()
(master["mod_36K"]-master["mod_4K"]).plot(kind="box",grid=True,notch=True)
g = sns.jointplot("wind_gust_4K", "wind_gust_o", data=master[master.wind_gust_o!=-1].sample(500), kind="reg",

                  color="b",).annotate(stats.pearsonr)
((master["wind_gust_4K"]-master["wind_gust_o"])[master.wind_gust_o!=-1]).describe()
((master["wind_gust_4K"]-master["wind_gust_o"])[master.wind_gust_o!=-1]).plot(kind="box",grid=True,notch=True)
g = sns.jointplot("wind_gust_36K", "wind_gust_o", data=master[master.wind_gust_o!=-1].sample(500), kind="reg",

                  color="b",).annotate(stats.pearsonr)
((master["wind_gust_36K"]-master["wind_gust_o"])[master.wind_gust_o!=-1]).describe()
((master["wind_gust_36K"]-master["wind_gust_o"])[master.wind_gust_o!=-1]).plot(kind="box",grid=True,notch=True)
g = sns.jointplot("wind_gust_4K", "wind_gust_36K", data=master[master.wind_gust_o!=-1].sample(500), kind="reg",

                  color="b",).annotate(stats.pearsonr)
(master["wind_gust_36K"]-master["wind_gust_4K"]).describe()
((master["wind_gust_4K"]-master["wind_gust_36K"])[master.wind_gust_o!=-1]).plot(kind="box",grid=True,notch=True)
ax = sns.jointplot(x="temp_4K", y="temp_o", data=master.sample(1000),kind="reg").annotate(stats.pearsonr)
sns.boxplot(master['temp_o'])
ax = sns.jointplot(x="temp_36K", y="temp_o", data=master,kind="reg").annotate(stats.pearsonr)
ax = sns.jointplot(x="rh_36K", y="rh_o", data=master.sample(1000),kind="reg").annotate(stats.pearsonr)

sns.boxplot(master['rh_o'])
ax = sns.jointplot(x="rh_4K", y="rh_o", data=master.sample(1000),kind="reg").annotate(stats.pearsonr)
ax = sns.jointplot(x="visibility_4K", y="visibility_o",kind="kde", data=master)
sns.boxplot(master['visibility_o'])
ax = sns.scatterplot(x="visibility_36K", y="visibility_4K", data=master)