# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_train=pd.read_csv("/kaggle/input/titanic/train.csv")
data_gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
data_test=pd.read_csv("/kaggle/input/titanic/test.csv")
data_train
data_gender
data_test
pd.options.display.max_rows=None
pd.options.display.max_columns=None
display(data_train)
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
pd.get_option("display.max_rows")
pd.set_option("display.max_rows", 20)
display(data_test)
data_train
data_train, data_test, data_gender
data_train.columns
len(data_train.columns), len(data_test.columns),len(data_gender.columns)
data_test.info()
data_gender.info()
data_test = pd.merge(data_gender,data_test, on="PassengerId")
data_test
data_test.info()
data_train.columns==data_test.columns
data=pd.concat([data_train,data_test], ignore_index=True)
data
veri1=data.copy()
veri1
data.info()
# sadece int ve float olan değişkenleri diakkate alır
data.describe()
# transpoz alınarak okuma kolaylığı sağlanabilir
data.describe().T
# yaşların ortalamaası
data["Age"].mean()
data["Age"].median()
data["Age"].max()
perc=[.10,.20,.40,.60,.80,.90]
include1=["object","float", "int"]
include=["float", "int"]
desc=data.describe(percentiles=perc, include=include)
desc.T
#data setimizin korelasyonu
korelasyon=data.corr()
korelasyon
# korelasyon tablosundaki 1'den küçük olan max sayıyı bulma 
korelasyon[korelasyon<1].abs().max()
a=korelasyon.abs()<1
b=korelasyon.abs()>0.5
korelasyon.abs()[a&b]


korelasyon.abs()
a=korelasyon.abs()<1
korelasyon.abs()[a]
data
data[data.Cabin.isna()]
data.info()
#cabin bilgisi dolu olanları listeleme
data[data.Cabin.isna()==False]
#cabin sütununda tekrar eden verilerin sayısı
data.Cabin.value_counts()
# isim sutununda tekrar eden verilere ulaşma
data["Name"].value_counts()
data[(data["Name"]=="Kelly, Mr. James") | (data["Name"]=="Connolly, Miss. Kate") ]
data["Ticket"].value_counts()
# aynı bilete sahip kişilere ulaşma
data[data["Ticket"]=="CA. 2343"]
# yapılmış sınıflandırmaları görme (unique)
data.Survived.unique()
data.Survived.unique(),data.Pclass.unique(),data.Sex.unique(),data.SibSp.unique(), data.Parch.unique(), data.Embarked.unique()
# titanicdeki yalnız seyahat eden kişilerin listesi
data[(data.SibSp==0) & (data.Parch==0)]
data["Survived"].mean()
# ortalama
data[(data["SibSp"]>=1)|(data["Parch"]>=1)]["Survived"].mean()
data[(data.SibSp==0) & (data.Parch==0)]["Survived"].mean()
data.isna().sum()
data
# kolon isimlerini değiştirme
veri=data.copy()
veri.rename(columns={"PassengerId":"YolcuId",
                     "Survived":"Yasam",
                    "PClass":"Sınıf",
                    "Name":"İsim",
                    "Sex":"Cinsiyet",
                    "Age":"Yas",
                    "SibSp":"Kardes_es_sayısı",
                    "Parch":"Cocuk_eb_sayısı",
                    "Ticket":"Bilet",
                    "Fare":"Ücret",
                     "Cabin":"Kabin",
                     "Embarked":"Liman"}, inplace=True)
veri
# içerik düzenleme
veri["Yasam"].replace(0,"yasamıyor", inplace=True)
veri["Yasam"].replace(1,"yasıyor", inplace=True)
veri
veri["Liman"].replace(["S","C","Q"],["Southampton","Cherbourg","Queenstown"], inplace=True)
veri
data["Cinsiyet"].replace({"male":"Erkek","female":"Kadın"}, inplace=True)
data
data["Cinsiyet"].replace({"erkek":"Erkek","kadın":"Kadın"}, inplace=True)
data
# nan değerlerin doldurulması
data["Kabin"].fillna(data["Yas"].mean(),inplace=True)

data
data["Kabin"].replace(data["Yas"].mean(),"Belirsiz", inplace=True)

data
data.Cinsiyet.groupby(data.Cinsiyet).count()
data.groupby(data.Cinsiyet).count()
data.groupby("Pclass")["Yasam"].describe()
data.groupby("Pclass").aggregate({"Ücret":"mean", "Yas":"mean"})
data.groupby("Pclass").aggregate({"Ücret":max, "Yas":"mean"})
veri.columns
veri
veri["Survived"].replace("yasamıyor",0, inplace=True)
veri["Survived"].replace("yasıyor",1, inplace=True)
veri
# cinsiyetler ve sınıflara göre yaşam değişkenin ortalamaları
veri.pivot_table("Yasam",index="Cinsiyet", columns="Pclass")
veri.groupby("Cinsiyet").aggregate(["min","count",np.mean,max])
veri.groupby("Cinsiyet")["Yas","Ücret"].aggregate(["min","count",np.mean,max])
# cinsiyete ve bilet sınıfına göre 0-25 ile 25-80 yas aralığındaki yaşam ortalaması
yas_degisken=pd.cut(veri["Yas"],[0,25,80])
veri.pivot_table("Yasam", ["Cinsiyet", yas_degisken], "Pclass")

# veriyi ort göre iki gruba bölme
ucret=pd.qcut(veri["Ücret"],2)
ucret
veri1.pivot_table("Survived", ("Sex",ucret),(yas_degisken, "Pclass"))
veri1.pivot_table(index="Sex", columns="Pclass", aggfunc={"Survived":"mean","Fare":max})
# cinsiyet ve bilet sınıflarına göre yaşam oranları ve toplam duruma bakış
veri1.pivot_table("Survived", index="Sex", columns="Pclass", margins=True)