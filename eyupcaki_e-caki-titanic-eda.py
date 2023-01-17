# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

## görselleştirme kütüphamesi ekledik 
import matplotlib.pyplot as plt 
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")  ##uyarıları gösterme diyoruz. kod çalışıyor iken hata zannedbiliyoruz.  

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
plt.style.available  ## kullanılabilir plot sitillieri 
a=[1,2,3,4]
plt.plot(a)
plt.show()

train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId=test_df["PassengerId"]   ##passenger ıd burada çakladık ileride kullanacağız


train_df.columns
train_df.head()
train_df.describe()

## sayısal verilere ait değerleri görüyoruz. Hızlıca bakılıp değerler üretmeye çalıştık. 

train_df.info()

# veri türlerine bakıyoruz. 

def bar_plot(variable):
    """
    inputlar sex
    output bar plot ve value count 
    """
    # get özellikler 
    var=train_df[variable]
   #count number of categorical variable 
    varValue=var.value_counts()  ## cinsiyetten kaç tane var gibi artık variable fonsiyonunda ne gelirse 
    
    # görselleştirme 
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frekansı")
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    
category1=["Survived", "Pclass", "Sex","SibSp","Parch","Embarked"]
for c in category1:
    bar_plot(c)
category2=["Cabin", "Name","Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable], bins=50)  #bins daha iyi görmemizi sağlar yoksa barlar birine giriyor default bir değeri var ona göre çalışır 
    plt.xlabel(variable)
    plt.ylabel("Frekansı")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
    
    
numericVar=["Fare","Age","PassengerId"]

for n in numericVar:
    plot_hist(n)
# Plcass ve Survived  yolcu sınıfları ve hayatta kalması 



train_df[["Pclass", "Survived"]].groupby (["Pclass"], as_index=False).mean().sort_values(by="Survived",ascending=False)

##train_df[["Pclass", "Survived"]]  Pclass ve Survived al sadece 
##Pclass hayatta kalmaya etkisi pclass gorupby yaptık . ve ortalamasını göster dedik. sortvalues görede survived azalan  sıralama 


#Sonuç 


#Pclass	Survived
#0	1	0.629630
#1	2	0.472826
#2	3	0.242363
# Cinsiyet ve hayatta kalma oranı 

train_df[["Sex", "Survived"]].groupby (["Sex"], as_index=False).mean().sort_values(by="Survived",ascending=False)

## Yanında bir kişi olan =0.53 Yanında kimse olmayan 0.34 
train_df[["SibSp", "Survived"]].groupby (["SibSp"], as_index=False).mean().sort_values(by="Survived",ascending=False)
## Parch ile Survived ilişkisi
## yanımızda çocuğumuz yada eşimiz varsa 


train_df[["Parch", "Survived"]].groupby (["Parch"], as_index=False).mean().sort_values(by="Survived",ascending=False)
def detect_outliers(df, features):
    outlier_indices=[]
    
    
    for c in features:
        # 1 quartile 
        Q1=np.percentile(df[c],25)
        
        #3. quartile
        Q3=np.percentile(df[c],75)
        
        #IQR
        IQR=Q3-Q1
        
        # Outlier step
        outlier_step=IQR*1.5
        
        #detect outlier and their indeces
        outlier_list_col=df[(df[c]< Q1 - outlier_step) | (df[c] > Q3 + outlier_step )].index
        
        #store indeces
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(i for i, v in outlier_indices.items()if v> 2) #1 tane outlier varsa kalacak 2 den fazla ise atacak 
    
    return multiple_outliers
    
a=["a","a","a","b","b","c"]
Counter(a)
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
  ## drop outliers

    
train_df=train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

train_df_len=len(train_df)
train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)

train_df.head()
train_df.columns[train_df.isnull().any()]   ## null değerler
train_df.isnull().sum()   ##Survived 418 test datasının sorusu zaten diğerleri boş burdadaki bazı dataları dolduracak kadar bilgimiz yok 
## ama bazılarını doldurabiliriz. 


train_df[train_df["Embarked"].isnull()]   ## nereden gemiye bindikleri belli değil aşağıdaki listeye baktık kabin nosu olur gibi 

train_df.boxplot(column="Fare", by="Embarked")
plt.show()

## C den binmiş olma ihtimali yüksek 

train_df["Embarked"]=train_df["Embarked"].fillna("C") ## C ile doldurduk 
train_df[train_df["Embarked"].isnull()]   ## null var mı diyede baktık 
train_df[train_df["Fare"].isnull()]   ## null var mı diyede baktık 
train_df[train_df["Pclass"]==3]  ## Pclass 3 olan değerlere baktık 

np.mean(train_df[train_df["Pclass"]==3]["Fare"])  ## Pclass 3 olan  ödemiş oldukları paraın ortalaması  
train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3] ["Fare"])) 
train_df[train_df["Fare"].isnull()] # baktık boş birşey kalmadı 
list1=["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(),annot=True, fmt=".2f")   # annot false yazı olmuyor 
plt.show()
g=sns.factorplot(x="SibSp", y="Survived", data=train_df, kind="bar", size=6)
g.set_ylabels("Survived Probability")
plt.show()


## 
train_df["SibSp"]

g=sns.factorplot(x="Parch",y="Survived", kind="bar", data=train_df, size=6)
g.set_ylabels("Survived Probability")
plt.show()   ## aile ve çocuk sayısı
##siyah oklar standart sapma demek . 0.6 ortalama ama değer ise 1.0 ile 0.2 arasında 3 Parch baktık 
g=sns.factorplot(x="Pclass", y="Survived",data=train_df, kind="bar", size=6)
g.set_ylabels("Survived Probability")
plt.show()

g=sns.FacetGrid(train_df,col="Survived")
g.map(sns.distplot, "Age", bins=25)
plt.show()

g=sns.FacetGrid(train_df, col="Survived", row="Pclass")
g.map(plt.hist, "Age", bins=25)
g.add_legend()
plt.show()
g=sns.FacetGrid(train_df,row="Embarked", size=2)
g.map(sns.pointplot, "Pclass", "Survived", "Sex")
g.add_legend()
plt.show()
g=sns.FacetGrid(train_df,row="Embarked", col="Survived", size=2.5)
g.map(sns.barplot,"Sex", "Fare")
g.add_legend()
plt.show()
train_df[train_df["Age"].isnull()]   ##256 tane yaş yok 

sns.factorplot(x="Sex", y="Age", data=train_df, kind="box")
plt.show()

##Cinsiyet yaş için bir veri sunmadı erkek ve kadın yaşları aynı kullanamadık 

sns.factorplot(x="Sex", y="Age", hue="Pclass",  data=train_df, kind="box")
plt.show()

## en yaşlılar 1. sınıfta en geç olanlarda 3. sınıfta olduğunu gördük 
sns.factorplot(x="Parch", y="Age",   data=train_df, kind="box")
sns.factorplot(x="SibSp", y="Age",   data=train_df, kind="box")
plt.show()
train_df["Sex"]=[1 if i=="male" else 0 for i in train_df["Sex"]]   ##male ve female yazarken erkeklere 1 bayanlara 0 yazdırdık yoksa grafikte görünmüyordu
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot=True)
plt.show()

## yaş ile cinsiyet arasında bağlantı yok colerasyon yok yani 
## SibSp, Parch, Pclass 

## Yaşları non olan dataların index bulalım

index_nan_age=list(train_df["Age"][train_df["Age"].isnull()].index)  ## non olanlar listendi 


for i in index_nan_age:   ##sibsp parch pclass özelliklerine bakarak yaşları belirleyeceğiz.
    age_pred=train_df["Age"][((train_df["SibSp"]==train_df.iloc[i]["SibSp"]) & (train_df["Parch"]==train_df.iloc[i]["Parch"]) & (train_df["Pclass"]==train_df.iloc[i]["Pclass"]))].median()
    age_med=train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i]=age_pred
    else:
        train_df["Age"].iloc[i]=age_med
    
    
age_pred
train_df[train_df["Age"].isnull()]   ## Tün nul olanları temizledik 