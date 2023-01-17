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
#bir data 3'e bolunmus

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
data_train.head()
data_test.head()
data_gender.head()
data_test2 = pd.merge(data_test, data_gender , on = 'PassengerId')

data_test2.head()
len(data_train.columns)
data_test2.columns ==data_train.columns
data_train.columns
data_test2.columns
#sutun sayilari isimleri ve terlerinin kontrolu yapilip duzeltmeler yapalim.

#train i test ile ayni sirada yaptik

data_train2 = data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
data_train2.head()
data_train2.tail()
#esitledigimiz datalari alt alta concat ettik

data = pd.concat([data_train2,data_test2], ignore_index=True)

data.head()
data_train.PassengerId.count()
data_test.PassengerId.count()
data_train.PassengerId.count()+data_test.PassengerId.count()
data.PassengerId.count()
data.PassengerId.count()==data_train.PassengerId.count()+data_test.PassengerId.count()
data.info()
#sadece int ve float alir

data.describe().T
#korelasyon

data.corr()
data.Cabin.unique()
#array yontemleri icin size ile yapmaliyiz. Count calismaz

data.Cabin.unique().size
data.Name.unique().size #toplam ozgun sayiyi bulabiliriz
data["Name"].value_counts() #Hangilerinden kacar tane var (uniq'lerden kacar tane var diye gosteriyor)
data["Cabin"].value_counts()
data[data["Name"]=="Connolly, Miss. Kate"]
data[data["Name"]=="Kelly, Mr. James"]
data["Ticket"].value_counts()
data[data["Ticket"]=="CA. 2343"]
#sutunlardali bos/non degerleri gosteriri

data.isna()
#kac tane bos/non oldugunu bulmak icin

data.isna().sum()
#turkcelestirebiliriz, nan'lari degistirebiliriz...

data.head()
#kolon isimlerini turkcelestirelim. #datayi basta kopyasinin alinmasi tavsiye edilir.

data.rename(columns={"PassengerId": "YolcuID", 

                     "Pclass": "Sinif",

                     "Name": "Ad_Soyad",

                     "Sex": "Cinsiyet",

                     "Age" : "Yas",

                     "SibSp":"Aile_1",

                     "Parch" : "Aile_2",

                     "Ticket" : "BiletID",

                     "Fare" : "Fiyati",

                     "KoltukNO" : "Oda",

                     "Embarked" : "Liman",

                     "Survived" : "Yasam"

                    }, inplace=True)
data.head()
#icerikleri replace edelim.

data["Yasam"].replace(0,"oldu", inplace=True)

data["Yasam"].replace(1,"yasiyor", inplace=True)

data["Liman"].replace("S","Southampton", inplace=True)

data["Liman"].replace("C","Cherbourg", inplace=True)

data["Liman"].replace("Q","Queenstown", inplace=True)

data["Cinsiyet"].replace("male","Erkek", inplace=True)

data["Cinsiyet"].replace("female","Kadin", inplace=True)
data.head()
data["Oda"].fillna("Belirsiz", inplace=True)
data.head()
#data.fillna("Belirsiz", inplace=True) #hepsindekiler bu sekilde degistirilmis olur.
#Cinsiyete gore gruplayip sayali, saymak gibi bi islem yapmaliyiz....

data.Cinsiyet.groupby(data.Cinsiyet).count()
data.Cinsiyet.value_counts() #bu sekilde de yapilabilir...
data.Liman.groupby(data.Liman).count()
#coklu suzme. sinifi=1 cinsiyeti= kadin ve yasi >=20 olanlar...

data[(data['Sinif']==1) & (data['Cinsiyet']== 'Kadin') & (data['Yas']>=20)]
#bazi sutunlari drop etme

data.drop(["Aile_1","Aile_2"],axis=0, inplace=True)

data.head()
#kadin ve erkekleri ayri ayri tutalim.

erkekler = data[data["Cinsiyet"]=="Erkek"]

kadinlar = data[data["Cinsiyet"]=="Kadin"]

cocuklar = data[data["Yas"]<=18]

kadinlar.head()
kadinlar.count()
#ctrl yan sılash tum kodlari # li alir

#siniflarin yas ve fiyat ortalamasini ayni anda aldi

data.groupby("Sinif")["Yas", "Fiyati"].mean()
#data["Fiyati"] = data.Fiyati.astype('int64') # integer'a ceviriyor.
ver1=data.copy()
ver1.dropna(inplace=True) #verideki nan degerlerini atacak
ver1.info()
ver1["Fiyati"]=ver1.Fiyati.astype('int64')

ver1.head() # integer yapmis olduk
ver1.apply(np.max) #uygula demek o sutunlarda istedigimiz fonksiyonlardaki islemi yapiyor