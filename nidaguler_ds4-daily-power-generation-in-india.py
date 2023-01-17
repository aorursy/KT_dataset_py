import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/daily-power-generation-in-india-20172020/file.csv")
df.head()
df.columns=["date","region","thermal_gen_act","thermal_gen_est","nuc_gen_act","nuc_gen_est","hydro_gen_act","hydro_gen_est"]
df.head()
df.tail()
df.isna().sum()
df=df.dropna()
df.tail()
df.isna().sum()
df.info()
df["thermal_gen_act"]=df["thermal_gen_act"].apply(lambda x:str(x).replace(',','')if ',' in str(x) else str(x))

df["thermal_gen_act"]=df["thermal_gen_act"].apply(lambda x: float(x))
df["thermal_gen_est"]=df["thermal_gen_est"].apply(lambda x:str(x).replace(",","")if ',' in str(x) else str(x))

df["thermal_gen_est"]=df["thermal_gen_est"].apply(lambda x: float(x))
df.info()
df.head()
df.reset_index(inplace=True, drop=True)

df.head()
df.tail()
df.describe()
df.corr()
df.region.unique()
labels = df.region.value_counts().index

colors = ['r','g','blue']

explode = [0,0,0]

sizes = df.region.value_counts().values



# visual 

plt.figure(0,figsize = (6,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Daily Power Gen According to Region',color = 'blue',fontsize = 15)

plt.show()
ax=df.thermal_gen_act.plot.kde()

ax=df.thermal_gen_est.plot.kde()

ax.legend()

plt.show()
ax=df.nuc_gen_act.plot.kde()

ax=df.nuc_gen_est.plot.kde()

ax.legend()

plt.show()
ax=df.hydro_gen_act.plot.kde()

ax=df.hydro_gen_est.plot.kde()

ax.legend()

plt.show()
sns.jointplot(x="thermal_gen_act", y="thermal_gen_est", data=df);
sns.jointplot(x="nuc_gen_act", y="nuc_gen_est", data=df);
sns.jointplot(x="hydro_gen_act", y="hydro_gen_est", data=df);
sns.jointplot(x=df.thermal_gen_act, y=df.thermal_gen_est, data=df, kind="kde");
sns.jointplot(x=df.nuc_gen_act, y=df.nuc_gen_est, data=df, kind="kde");
sns.jointplot(x=df.hydro_gen_act, y=df.hydro_gen_est, data=df, kind="kde");
f,ax1 = plt.subplots(figsize =(10,6))

sns.pointplot(x='region',y='thermal_gen_act',data=df,color='lime',alpha=0.8)

sns.pointplot(x='region',y='thermal_gen_est',data=df,color='palegreen',alpha=0.8)

sns.pointplot(x='region',y='nuc_gen_act',data=df,color='red',alpha=0.8)

sns.pointplot(x='region',y='nuc_gen_est',data=df,color='firebrick',alpha=0.8)

sns.pointplot(x='region',y='hydro_gen_act',data=df,color='blue',alpha=0.8)

sns.pointplot(x='region',y='hydro_gen_est',data=df,color='skyblue',alpha=0.8)

plt.text(-0.6,-300,'region-thermal_gen_act',color='lime',fontsize = 15,style = 'italic')

plt.text(-0.6,-400,'region-thermal_gen_est',color='palegreen',fontsize = 15,style = 'italic')

plt.text(0.5,-300,'region-nuclear_gen_act',color='red',fontsize = 15,style = 'italic')

plt.text(0.5,-400,'region-nuclear_gen_est',color='firebrick',fontsize = 15,style = 'italic')

plt.text(1.6,-300,'region-hydro_gen_act',color='blue',fontsize = 15,style = 'italic')

plt.text(1.6,-400,'region-hydro_gen_est',color='skyblue',fontsize = 15,style = 'italic')

plt.xlabel('region',fontsize = 15,color='blue')

plt.ylabel('values',fontsize = 15,color='blue')

plt.title('thermal  -  nuclear - hydro',fontsize = 20,color='blue')

plt.grid()

plt.show()