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

data_train.head(5)
data_test=pd.read_csv("/kaggle/input/titanic/test.csv")

data_test.head(3)
data_gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

data_gender
data_test2=pd.merge(data_test,data_gender,on="PassengerId")

data_test2
len(data_test2.columns)

len(data_train.columns)
data_train.columns==data_test2.columns
data_train.columns
data_test2.columns
data_test2.columns
#sutunlari data test2 deki gibi yap

data_train2=data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]

                        

data_train2
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
data.Cabin.unique()
data.Name.unique()
data.Name.unique().size#array dondugu icin size kullandik
data["Name"].value_counts()#ayni isimden  kactane var
#suzme islemi yap

data[data["Name"]=="Connolly, Miss. Kate"]
data[data["Name"]=="Kelly, Mr. James"]
data["Ticket"].value_counts()#ayni bilet kactane var
data[data["Ticket"]=="CA. 2343"]
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

                     "Cabin" : "Kamara",

                     "Embarked" : "Liman",

                     "Survived" : "Yasam"

                    }, inplace=True) 

data.head(10)
#icerikleri replace edelim.

data["Yasam"].replace(0,"oldu", inplace=True)

data["Yasam"].replace(1,"yasiyor", inplace=True)

data["Liman"].replace("S","Southampton", inplace=True)

data["Liman"].replace("C","Cherbourg", inplace=True)

data["Liman"].replace("Q","Queenstown", inplace=True)

data["Cinsiyet"].replace("male","Erkek", inplace=True)

data["Cinsiyet"].replace("female","Kadin", inplace=True)

data
data["Kamara"].fillna("Belirsiz",inplace=True)

data.head()
data.Cinsiyet.groupby(data.Cinsiyet).count()
data.Cinsiyet.value_counts()
data.Yasam.groupby(data.Yasam).count()
data[(data["Sinif"]==1)&(data["Cinsiyet"]=="Kadin")]
# data.drop(["Aile_1","Aile_2"],axis=1,inplace=True)

#kadin ve erkekleri ayri ayri tutalim.

erkekler = data[data["Cinsiyet"]=="Erkek"]

kadinlar = data[data["Cinsiyet"]=="Kadin"]



kadinlar.head()

kadinlar.count()
data.groupby("Sinif")["Yas","Fiyati"].mean()
data["Fiyati"]=data.Fiyati.astype("int64")#verileri tipin i degistirdi
veri=data.copy()

veri
veri.dropna()

veri.apply(np.max)
