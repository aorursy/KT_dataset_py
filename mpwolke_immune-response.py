# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/ai4all-project/results/deconvolution/CIBERSORTx_Results_Krasnow_facs_droplet.csv', encoding='ISO-8859-2')

df.head()



df_grp = df.groupby(["viral_load","czb_id"])[["B cell","Basal","Basophil/Mast", "Ciliated", "Dendritic", "Goblet", "Ionocyte", "Monocytes/macrophages", "Neutrophil", "T cell"]].sum().reset_index()

df_grp.head()
df_grp = df_grp.rename(columns={"B cell":"Bcell","T cell":"Tcell", "Monocytes/macrophages": "macrophages"})
plt.figure(figsize=(15, 5))

plt.title('czb_id')

df_grp.czb_id.value_counts().plot.bar();
df_grp_plot = df_grp.tail(80)
df_grp_r = df_grp.groupby("czb_id")[["Bcell","Tcell","macrophages", "Dendritic", "Ciliated", "Neutrophil"]].sum().reset_index()
df_grp_r.head()
df_grp_rl20 = df_grp_r.tail(20)
fig = px.bar(df_grp_rl20[['czb_id', 'Bcell']].sort_values('Bcell', ascending=False), 

             y="Bcell", x="czb_id", color='czb_id', 

             log_y=True, template='ggplot2', title='B Cells vs CZB ID')

fig.show()
df_grp_rl20 = df_grp_rl20.sort_values(by=['Bcell'],ascending = False)
plt.figure(figsize=(40,15))

plt.bar(df_grp_rl20.czb_id, df_grp_rl20.Bcell,label="Bcell")

plt.bar(df_grp_rl20.czb_id, df_grp_rl20.Tcell,label="Tcell")

plt.bar(df_grp_rl20.czb_id, df_grp_rl20.macrophages,label="macrophages")

plt.xlabel('viral_load')

plt.ylabel("Count")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)



plt.legend(frameon=True, fontsize=12)

plt.title('Immune Response',fontsize=30)

plt.show()



f, ax = plt.subplots(figsize=(40,15))

ax=sns.scatterplot(x="czb_id", y="Bcell", data=df_grp_rl20,

             color="black",label = "Bcell")

ax=sns.scatterplot(x="czb_id", y="Tcell", data=df_grp_rl20,

             color="red",label = "Tcell")

ax=sns.scatterplot(x="czb_id", y="macrophages", data=df_grp_rl20,

             color="blue",label = "macrophages")

plt.plot(df_grp_rl20.czb_id,df_grp_rl20.Bcell,zorder=1,color="black")

plt.plot(df_grp_rl20.czb_id,df_grp_rl20.Tcell,zorder=1,color="red")

plt.plot(df_grp_rl20.czb_id,df_grp_rl20.macrophages,zorder=1,color="blue")

plt.xticks(fontsize=13)

plt.yticks(fontsize=15)

plt.legend(frameon=True, fontsize=12)
df_grp_d = df_grp.groupby("viral_load")[["Bcell","Tcell","macrophages"]].sum().reset_index()
df_grp_dl20 = df_grp_d.tail(20)
df_grp_d['Bcell_new'] = df_grp_d['Bcell']-df_grp_d['Bcell'].shift(1)

df_grp_d['Tcell_new'] = df_grp_d['Tcell']-df_grp_d['Tcell'].shift(1)

df_grp_d['macrophages_new'] = df_grp_d['macrophages']-df_grp_d['macrophages'].shift(1)
new = df_grp_d

new = new.tail(14)
f, ax = plt.subplots(figsize=(23,10))

ax=sns.scatterplot(x="viral_load", y="Bcell", data=df_grp_dl20,

             color="black",label = "B cells")

ax=sns.scatterplot(x="viral_load", y="Tcell", data=df_grp_dl20,

             color="red",label = "T cells")

ax=sns.scatterplot(x="viral_load", y="macrophages", data=df_grp_dl20,

             color="blue",label = "Macrophages")

plt.plot(df_grp_dl20.viral_load,df_grp_dl20.Bcell,zorder=1,color="black")

plt.plot(df_grp_dl20.viral_load,df_grp_dl20.Tcell,zorder=1,color="red")

plt.plot(df_grp_dl20.viral_load,df_grp_dl20.macrophages,zorder=1,color="blue")