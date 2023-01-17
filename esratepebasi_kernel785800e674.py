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
#her bir veri setini okuttuk

data_train=pd.read_csv("/kaggle/input/titanic/train.csv")

data_gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

data_test=pd.read_csv("/kaggle/input/titanic/test.csv")
data_train.tail()
#sib sp ust nesilden tanidigi

#parch alt nesilden kuzen kardes kac tane

#fare bilet ucreti

#embarked bindigi liman
data_test.head()
data_gender.head()
data_test2=pd.merge(data_test,data_gender,on="PassengerId")
data_test2.head()
len(data_test2.columns)
len(data_train.columns)
data_test2.columns==data_train.columns
data_test2.columns
data_train.columns
#kolonlari yer degistirdik train 1 test ile ayni sirada yaptik.sutunlari toptan alip yer degistiriyor

data_train2=data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
data_train2.tail()
data_test2.columns==data_train2.columns
data=pd.concat([data_train2,data_test2],ignore_index=True)
data.head()
data_train.PassengerId.count()
data_test.PassengerId.count()
#ilk iki data verisi toplami

data_train.PassengerId.count()+data_test.PassengerId.count()
#son data verisi sayiisi

data.PassengerId.count()

#kayip olmamis
#artik data isimli veri seti hepsini toplamis durumda
data
data.info()
data.describe().T
data.corr()
data.Pclass.unique()
data.Cabin.unique()
data.Cabin.unique().size

#.count dataframe ozelligi.bu arrray oldugu icin hata veriyor
len(data.Cabin.unique())
data.Name.unique().size
#1309 kisi vardi.2 tane ayni isim kullanilmis
data["Name"].value_counts()
data[data["Name"]=="Connolly, Miss. Kate"]
#bu iki kisi farkli kisiler oldugunu anladik

data[data["Name"]=="Kelly, Mr. James"]
#bu iki kisi farkli kisiler.o zaman drop yapmaya gerek yok
data["Ticket"].value_counts()
data[data["Ticket"]=="CA. 2343"]
data.isna().sum()
data.head()
#kolon isimlerini turkcelestirelim.

data.rename(columns={"PassengerId": "YolcuID", 

                     "Pclass": "Sinif",

                     "Name": "Ad_Soyad",

                     "Sex": "Cinsiyet",

                     "Age" : "Yas",

                     "SibSp":"Aile_1",

                     "Parch" : "Aile_2",

                     "Ticket" : "BiletID",

                     "Fare" : "Fiyati",

                     "KoltukNO" : "kamara",

                     "Embarked" : "Liman",

                     "Survived" : "Yasam"

                    }, inplace=True) 
data
#icerikleri replace edelim.

data["Yasam"].replace(0,"oldu", inplace=True)

data["Yasam"].replace(1,"yasiyor", inplace=True)

data["Liman"].replace("S","Southampton", inplace=True)

data["Liman"].replace("C","Cherbourg", inplace=True)

data["Liman"].replace("Q","Queenstown", inplace=True)

data["Cinsiyet"].replace("male","Erkek", inplace=True)

data["Cinsiyet"].replace("female","Kadin", inplace=True)
data
# data.fillna("Belirsiz",inplace=True)
data
data.Cinsiyet.groupby(data.Cinsiyet).count()
data.Cinsiyet.value_counts()
data.Yasam.groupby(data.Yasam).count()
data[(data["Sinif"]==1)&(data["Cinsiyet"]=="Kadin")]
data.drop(["Aile_1","Aile_2"],axis=1,inplace=True)
data
#kadin ve erkekleri ayri ayri tutalim.

erkekler = data[data["Cinsiyet"]=="Erkek"]

kadinlar = data[data["Cinsiyet"]=="Kadin"]

cocuklar = data[data["Yas"]<=18]

kadinlar
data.groupby("Sinif")["Yas","Fiyati"].mean()
veri=data.copy()
veri.dropna(inplace=True)  #bu islem mantikli degil
veri
veri["Fiyati"]=veri.Fiyati.astype("int64")
veri
veri.apply(np.max)  #numpy daki max islemini dataframe de kullanabilmek icin