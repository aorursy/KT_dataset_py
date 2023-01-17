# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
data_2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")

data_2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv")

data_2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv")

data_2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv")

data_2019=pd.read_csv("/kaggle/input/world-happiness/2019.csv")   
data_2015.info()
data_2016.info()
data_2017.info()
data_2018.info()
data_2019.info()
data_2018.isnull().sum()
data_2019.head()
data_2019[data_2019["Country or region"]=="Turkey"]
data_2018[data_2018["Country or region"]=="Turkey"]
data_2017[data_2017["Country"]=="Turkey"]
data_2016[data_2016["Country"]=="Turkey"]
data_2015[data_2015["Country"]=="Turkey"]
df_TR=data_2019.copy()
df_TR=df_TR[df_TR["Country or region"]=="Turkey"]
df_TR
df_TR= pd.concat([data_2019[data_2019["Country or region"]=="Turkey"],data_2018[data_2018["Country or region"]=="Turkey"]])
df_TR
df_TR["Year"]=["2019","2018"]
df_TR.columns=["Rank","Country","Score","GDP","Social_Support","Healthy_Life_Ex","Freedom","Generosity","Per_of_Corruption","Year"]
df_TR
data_2017[data_2017["Country"]=="Turkey"]
data_2017.drop(columns=["Whisker.high","Whisker.low"],inplace=True)
data_2016.drop(columns=["Lower Confidence Interval","Upper Confidence Interval","Region"],inplace=True)
data_2015.drop(columns=["Standard Error","Region"],inplace=True)
data_2015[data_2015["Country"]=="Turkey"]
data_2016[data_2016["Country"]=="Turkey"]
data_2017[data_2017["Country"]=="Turkey"]
data_2017_gen=data_2017["Generosity"]
data_2017_dys=data_2017["Dystopia.Residual"]
data_2017.drop(columns=["Generosity","Dystopia.Residual"],inplace=True)
data_2017_new=pd.concat([data_2017,data_2017_gen,data_2017_dys],axis=1)
data_2017_new[data_2017_new["Country"]=="Turkey"]
colums=list(data_2017_new.columns)
colums
data_2016.columns=colums
data_2015.columns=colums
data_2015.columns
data_2015.head(1)
df_TR2= pd.concat([data_2017_new[data_2017_new["Country"]=="Turkey"],data_2016[data_2016["Country"]=="Turkey"],data_2015[data_2015["Country"]=="Turkey"]])
df_TR2
df_TR2["Year"]=["2017","2016","2015"]
df_TR2
df_TR2.index=df_TR2.Year
df_TR2
df_TR.index=df_TR.Year
df_TR.drop(columns="Year",inplace=True)
df_TR2.drop(columns="Year",inplace=True)
df_TR2.columns=["Country","Rank","Score","GDP","Family","Healthy_Life_Ex","Freedom","Per_of_Corruption","Generosity","Dystopia"]
df_TR2 = df_TR2.reindex(columns=['Rank',"Country","Score","GDP","Family","Healthy_Life_Ex","Freedom","Generosity","Per_of_Corruption","Dystopia"])
df_TR2
df_TR2.rename(columns={"Family":"Social_Support"},inplace=True)
df_TR2
df_TR
df_TR["Dystopia"]=df_TR.Score-df_TR.GDP-df_TR.Social_Support-df_TR.Healthy_Life_Ex-df_TR.Freedom-df_TR.Generosity-df_TR.Per_of_Corruption
df_TR
df=pd.concat([df_TR,df_TR2])
df
df_c=df.iloc[:, 3:]
df_c
df.iloc[:,2:].corr(method='spearman')
df_c.plot.barh(stacked=True, figsize=(8,6));
labels =df_c.columns

explode = [0,0,0,0,0,0,0]

sizes = df_c.iloc[0]



plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')

plt.title("Türkiye'nin 2019 yılı mutluluğa etki eden faktörlerin oranları",color = 'Red',fontsize = 15)

plt.show()
labels =df_c.columns

explode = [0,0,0,0,0,0,0]

sizes = df_c.mean()



plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')

plt.title("Türkiye'nin 2015-2019 yılları arasında mutluluğa etki eden faktörlerin ortalamaların oranları",color = 'red',fontsize = 15)

plt.show()
f,ax1 = plt.subplots(figsize =(10,6))

sns.pointplot(x=df.index,y='Rank',data=df,color='red',alpha=0.8)

plt.xticks(rotation=45)

plt.xlabel('Year',fontsize = 15,color='blue')

plt.ylabel('Rank',fontsize = 15,color='blue')

plt.ylim(60,90)

plt.title('Turkiye Mutluluk Sıralaması(2015-2019)',fontsize = 20,color='red')

plt.grid()

plt.show()


f,ax = plt.subplots(figsize = (9,5))

labels =df_c.columns

colorss = ['cyan','lime','pink',"blue","red","green","yellow"]

sns.pointplot(x=df.index,y='GDP',data=df,color="cyan")

sns.pointplot(x=df.index,y='Social_Support',data=df,color="lime")

sns.pointplot(x=df.index,y="Healthy_Life_Ex",data=df,color="pink")

sns.pointplot(x=df.index,y='Freedom',data=df,label="Freedom",color="blue")

sns.pointplot(x=df.index,y='Generosity',data=df,label="Genereosity",color="red")

sns.pointplot(x=df.index,y='Per_of_Corruption',data=df,label="Corruption",color="green")

sns.pointplot(x=df.index,y='Dystopia',data=df,label="Dystopia",color="yellow")



plt.xticks(rotation=45)

plt.xlabel('Year',fontsize = 15,color='blue')

plt.ylabel('Value',fontsize = 15,color='blue')

plt.title('Turkiye Mutluluk Sıralamasına Etki Eden Faktörlerin Yıllara Göre Değişimi(2015-2019)',fontsize = 20,color='red')

ax.legend(labels=labels,loc='lower right',frameon = True)

#ax.set(xlabel='Percentage of Region', ylabel='Region',title = "Factors affecting happiness score")

plt.show()
data_2019.head()
data_2019[data_2019.Score==data_2019.Score.max()]
data_2019[data_2019.Score==data_2019.Score.min()]
data_2019.isnull().sum()
data_2019.iloc[:,2:].corr(method='spearman')
sns.heatmap(data_2019.iloc[:,2:].corr(method='spearman'), cmap="viridis", annot=True)
plt.figure(figsize=(12,10))

sns.barplot(x=data_2019[:10]["Country or region"], y=data_2019.Score)

plt.xticks(rotation= 90)

plt.xlabel('Country')

plt.ylabel('Country Happiness Score')

plt.title('Happiness Score')

plt.show()
df_2019=data_2019.copy()

df_2019.index=df_2019["Country or region"]
df_2019.head()
df_2019.iloc[1,3:].values.sum()
df_2019.iloc[0]["Score"]
df_2019["Dystopia"]=df_2019.Score-df_2019.iloc[:,3]-df_2019.iloc[:,4]-df_2019.iloc[:,5]-df_2019.iloc[:,6]-df_2019.iloc[:,7]-df_2019.iloc[:,8]
df_2019.head(2)
df_2019.iloc[:5,3:].plot.barh(figsize=(12,10))
df_2019.iloc[:10,3:].plot.barh(stacked=True, figsize=(14,8))
sns.heatmap(df_2019.iloc[:,2:].corr(method='spearman'), cmap="viridis", annot=True)