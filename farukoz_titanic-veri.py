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
#bir data 3'e bölünmüş

data_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

data_test = pd.read_csv('//kaggle/input/titanic/test.csv')

data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_train.head()
data_test.head()
data_gender.head()
data_test2 = pd.merge(data_test, data_gender, on = 'PassengerId')

data_test2.head()
len(data_test2.columns)
len(data_train.columns)
data_test2.columns==data_train.columns
data_test2.columns
data_train.columns
# clonları yer değiştiriliyor

data_train2 = data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
data_train2.tail()
# tüm verileri birliştiriyoruz

data = pd.concat([data_train2,data_test2], ignore_index=True)

data.head()
data_train.PassengerId.count()
data_test.PassengerId.count()
data_train.PassengerId.count()+data_test.PassengerId.count()
data.PassengerId.count()
data
data.info()
data[data['Survived']==1]['Survived'].count()
data.describe().T
data.Cabin.unique()
data.Cabin.unique().size
data.Name.unique().size
# Değerlerden kaçar tane olduğunu gösterir

data['Cabin'].value_counts()
data['Name'].value_counts()
data[data['Name']=='Kelly, Mr. James']
data['Ticket'].value_counts()
data[data['Ticket']=='CA. 2343']
# stunlardaki bos verilerin toplamını verir

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

                     "KoltukNO" : "Kamara",

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
data['Cabin'].fillna('Belirsiz', inplace=True)

data.head()
# cinsiyete göre gruplayıp sayalım

data.Cinsiyet.groupby(data.Cinsiyet).count()
data.Cinsiyet.value_counts()
data.Yasam.groupby(data.Cinsiyet).count()
data[(data['Sinif']==1)&(data['Cinsiyet']=='Kadin')&(data['Yas']>=20)]
data.drop(['Aile_1','Aile_2'],axis=1, inplace=True)

data
data.groupby('Sinif')['Yas','Fiyati'].mean()
veri=data.copy()

veri
veri.dropna(inplace=True)

veri
veri['Fiyati']=veri.Fiyati.astype('int64')

veri.head()
#stunlarda özel tanımlı fonksiyonları uygulayabilme olanağı verir

veri.apply(np.max)