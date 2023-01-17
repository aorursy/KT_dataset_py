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
#bir data 3 e bolunmus. 

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")    

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

data_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
data_train.head()
data_test.head()
data_gender.head()
# daginik olan test verisini birlestirelim

data_test2 = pd.merge(data_test, data_gender, on = 'PassengerId') 

data_test2.head()
len(data_test2.columns)
len(data_train.columns)
data_test2.columns==data_train.columns
data_train.columns
data_test2.columns
#kolonlari yer degistirdik. train i test ile ayni sirada yaptik.



data_train2 = data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]

data_train2.tail()
data_test2.columns == data_train2.columns
data_test2.head()
data_train2.head()
#esitledigimiz datalari alt alta concat ettik

data = pd.concat([data_train2,data_test2], ignore_index=True)

data.head(3)
data_train.PassengerId.count()

data_test.PassengerId.count()
data_train.PassengerId.count() + data_test.PassengerId.count()
data.PassengerId.count()
data
data.info()
data.describe().T #sadece int ve float alir
#korelasyon



data.corr()
#cabin de kac cesit unique deger var.    

data.Cabin.unique()
data.Name.unique().size    #ile toplam sayiya bakabilirsiniz.
# Name sutununda tekrar eden degerleri sorgular

data["Name"].value_counts()
data[data["Name"]=="Kelly, Mr. James"]
data[data["Name"]=="Connolly, Miss. Kate"]
# Ticket sutununda tekrar eden degerleri sorgular



data["Ticket"].value_counts()
data[data["Ticket"]=="CA. 2343"]



#not bu 11 kisiye bakip yorum yapmaya calisin.
# sutunlardaki bos/NaN degerleri toplar.

data.isna().sum()
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

                     "Cabin" : "KoltukNO",

                     "Embarked" : "Liman",

                     "Survived" : "Yasam"

                    }, inplace=True)   
data.columns
data.head()
#icerikleri replace edelim.



data["Yasam"].replace(0,"oldu", inplace=True)

data["Yasam"].replace(1,"yasiyor", inplace=True)

data["Liman"].replace("S","Southampton", inplace=True)

data["Liman"].replace("C","Cherbourg", inplace=True)

data["Liman"].replace("Q","Queenstown", inplace=True)

data["Cinsiyet"].replace("male","Erkek", inplace=True)

data["Cinsiyet"].replace("female","Kadin", inplace=True)



data.head(10)
# NaN Degerleri dolduralim.   > fillna



data["KoltukNO"].fillna("Belirsiz", inplace=True)

data.head()
#cinsiyete gore gruplayip, sayalim



data.Cinsiyet.groupby(data.Cinsiyet).count()
# Yasam durumuna gore gruplayip, sayalim



data.Yasam.groupby(data.Yasam).count()
#sinifi ==1 ve cinsiyeti == kadin olan degerler. coklu suzme



data[(data['Sinif'] == 1) & (data['Cinsiyet'] == 'Kadin')]
#bazi sutunlari drop edelim



data.drop(["Aile_1","Aile_2"],axis=1, inplace=True)
data.head()
#kadin ve erkekleri ayri ayri tutalim.

erkekler = data[data["Cinsiyet"]=="Erkek"]

kadinlar = data[data["Cinsiyet"]=="Kadin"]

cocuklar = data[data["Yas"]<=18]



kadinlar.head()
kadinlar.Yas.mean()
erkekler.Yas.mean()
# kac cocuk var? 

cocuklar.YolcuID.count()
cocuklar.Yas.mean()
sinif_1 = data[data["Sinif"] == 1]

sinif_2 = data[data["Sinif"] == 2]

sinif_3 = data[data["Sinif"] == 3]
# verideki siniflarin yas ve fiyat ortalamalarini alacak.

data.groupby("Sinif")["Yas","Fiyati"].mean()
#farkli islemler yapalim. Onun icin hazir veriyi copy edelim

veri = data.copy()

veri.head()
#verideki NaN degerleri ve o satiri atacak

veri.dropna(inplace=True)
veri.count()
#yasi integer yaptik ve tekrar yas sutunu ile degisim yaptik.

veri["Fiyati"] = veri.Fiyati.astype('int64')
veri.head()
#her kolondaki en yuksek veriyi getirir.

veri.apply(np.max)
# Yasi 80 olan kisi

veri[veri["Yas"] >= 80]
veri["Yas"]
a = veri.iloc[:,[2,3]]
a.T
# pivot almak, satirlari sutun yapmak



veri.pivot(index ='Ad_Soyad', columns ='YolcuID')
