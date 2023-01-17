# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/turkish-employee-dataset/calisan_data.csv")
df.head()
mail_turu=[]

for i in df["mail"]:

    ilk=i.find("@")

    son=i.find(".com")

    i=i[ilk+1:son]

    mail_turu.append(i)

    

dogum_yili=[]

for i in df["DogumTarihi"]:

    dogum_yili.append(i[-4:])
df=df.rename(columns={"mail":"Mail"})

df["MailTuru"]=np.array(mail_turu)

df.insert(6,"DogumYili",np.array(dogum_yili))

df["DogumYili"]=df["DogumYili"].astype("int64")
df.head()
df.info()
sns.set_style("whitegrid")
plt.figure(figsize=(20,7))

sns.kdeplot(df["DogumYili"],shade=True)
sns.FacetGrid(df,hue="Sehir",height=5,aspect=3.3).map(sns.kdeplot,"DogumYili",shade=True).add_legend()
sns.FacetGrid(df,hue="Departman",height=5,aspect=3.3).map(sns.kdeplot,"DogumYili",shade=True).add_legend()
sns.FacetGrid(df,height=5,col="Sehir").map(sns.kdeplot,"DogumYili",shade=True).add_legend()
sns.FacetGrid(df,hue="Departman",height=5,col="Sehir").map(sns.kdeplot,"DogumYili",shade=True).add_legend()
sns.FacetGrid(df,hue="Sehir",height=5,aspect=3.3).map(sns.kdeplot,"Maas",shade=True).add_legend()
sns.FacetGrid(df,hue="Departman",height=5,aspect=3.3).map(sns.kdeplot,"Maas",shade=True).add_legend()
sns.FacetGrid(df,height=5,col="Sehir").map(sns.kdeplot,"DogumYili",shade=True).add_legend()
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,7))

df["TelefonTuru"].value_counts().plot.pie(autopct="%1.1f%%",shadow=True,ax=ax[0])

ax[0].set_title("Pasta Grafiği",fontsize=20)

sns.countplot(df["TelefonTuru"],order=df["TelefonTuru"].value_counts().index,ax=ax[1])

ax[1].set_title("Sütun Grafiği",fontsize=20)

plt.show()
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,7))

df["Departman"].value_counts().plot.pie(autopct="%1.1f%%",shadow=True,ax=ax[0],colors=["#66B3FF","#FFCC99","#82E0AA","#F1948A","#BB8FCE"])

ax[0].set_title("Pasta Grafiği",fontsize=20)

sns.countplot(df["Departman"],order=df["Departman"].value_counts().index,ax=ax[1],palette="pastel")

ax[1].set_title("Sütun Grafiği",fontsize=20)

plt.xticks(rotation=30)

plt.show()
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,7))

df["Sehir"].value_counts().plot.pie(autopct="%1.1f%%",shadow=True,ax=ax[0],colors=["#F1C40F","#21618C","#1D8348","#E74C3C"])

ax[0].set_title("Pasta Grafiği",fontsize=20)

sns.countplot(df["Sehir"],order=df["Sehir"].value_counts().index,ax=ax[1],palette=["#F1C40F","#21618C","#1D8348","#E74C3C"])

ax[1].set_title("Sütun Grafiği",fontsize=20)

plt.xticks(rotation=30)

plt.show()
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,7))

df["MailTuru"].value_counts().plot.pie(autopct="%1.1f%%",shadow=True,ax=ax[0],colors=["#FF00FF","#00FFFF","#00FF00","#FF0000","#C0C0C0"])

ax[0].set_title("Pasta Grafiği",fontsize=20)

sns.countplot(df["MailTuru"],order=df["MailTuru"].value_counts().index,ax=ax[1],palette=["#FF00FF","#00FFFF","#00FF00","#FF0000","#C0C0C0"])

ax[1].set_title("Sütun Grafiği",fontsize=20)

plt.xticks(rotation=30)

plt.show()
plt.figure(figsize=(20,9))

sns.countplot(df["DogumYili"],order=df["DogumYili"].value_counts().index)

plt.title("Sütun Grafiği",fontsize=16)

plt.show()
plt.figure(figsize=(20,7))

plt.title("Departmanın Maaşa Göre Durumu",fontsize=16)

sns.barplot(x="Departman",y="Maas",data=df)
plt.figure(figsize=(20,7))

plt.title("Departmanın Doğum Yılına Göre Dağılımı",fontsize=16)

sns.barplot(x="Departman",y="DogumYili",data=df,ci=None,estimator=np.sum)
plt.figure(figsize=(20,7))

plt.title("Doğum Yılının Maaşa Göre Durumu",fontsize=16)

sns.barplot(x="DogumYili",y="Maas",data=df,ci=None)
plt.figure(figsize=(20,7))

plt.title("Doğum Yılının Maaş ve Şehire Göre Durumu",fontsize=16)

sns.barplot(x="DogumYili",y="Maas",hue="Sehir",data=df,ci=None)
sns.catplot(x="DogumYili",y="Maas",data=df,aspect=3.3)
sns.catplot(x="DogumYili",y="Maas",hue="Sehir",data=df,aspect=3.3)
sns.catplot(x="DogumYili",y="Maas",hue="Departman",data=df,aspect=3.3)
sns.catplot(x="MailTuru",y="Maas",col="Sehir",data=df)
sns.catplot(x="MailTuru",y="Maas",hue="Departman",col="Sehir",data=df)
sns.catplot(x="DogumYili",y="Maas",data=df,kind="point",aspect=3.3)
sns.catplot(x="DogumYili",y="Maas",data=df,hue="Sehir",kind="point",aspect=3.3)
sns.catplot(x="DogumYili",y="Maas",data=df,hue="Departman",kind="point",aspect=3.3)
sns.catplot(x="MailTuru",y="Maas",col="Sehir",data=df,kind="point")
sns.catplot(x="MailTuru",y="Maas",col="Sehir",hue="Departman",data=df,kind="point")
plt.figure(figsize=(20,7))

sns.boxplot(x="DogumYili",y="Maas",data=df)
plt.figure(figsize=(20,7))

sns.boxplot(x="DogumYili",y="Maas",hue="Sehir",data=df,palette="pastel")
plt.figure(figsize=(20,7))

sns.boxplot(x="DogumYili",y="Maas",hue="Departman",data=df,palette="hls")