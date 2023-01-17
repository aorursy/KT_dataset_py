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
data_train= pd.read_csv("/kaggle/input/titanic/train.csv")

data_test= pd.read_csv("/kaggle/input/titanic/test.csv")

data_gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
data_train.head()
data_test.head()
data_gender.head()
data_test2= pd.merge(data_test, data_gender,on="PassengerId")



data_test2.head()
len(data_test2.columns)
len(data_train.columns)
data_test2.columns==data_train.columns
data_train.columns
data_test2.columns
data_train2=data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
data_train2.tail()
data_test2.columns==data_train2.columns
data=pd.concat([data_train2,data_test2], ignore_index=True)

data.head()
data_train.PassengerId.count()+data_test.PassengerId.count()
data.PassengerId.count()
data.head()
data.info()
data.describe().T
data.corr()
data.Cabin.unique().size
data.Name.unique().size
data["Name"].value_counts()
data[data['Name']=='Connolly, Miss. Kate']
data[data['Name']=='Kelly, Mr. James']
data["Ticket"].value_counts()
data[data['Ticket']=='CA. 2343']
data.isna().sum()
data.head()
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
data["Yasam"].replace(0,"oldu", inplace=True)

data["Yasam"].replace(1,"yasiyor", inplace=True)

data["Liman"].replace("S","Southampton", inplace=True)

data["Liman"].replace("C","Cherbourg", inplace=True)

data["Liman"].replace("Q","Queenstown", inplace=True)

data["Cinsiyet"].replace("male","Erkek", inplace=True)

data["Cinsiyet"].replace("female","Kadin", inplace=True)
data.head()
# data.fillna('Belirsiz',inplace=True)

# data.head()
data.Cinsiyet.groupby(data.Cinsiyet).size()

data.Cinsiyet.value_counts()
data.Yasam.groupby(data.Cinsiyet).size()
data[(data['Sinif']==1)&(data['Cinsiyet']=='Kadin')]
# data.drop(['Aile_1,Aile_2'],axis=1,inplace=True)
erkekler=data[data['Cinsiyet']=='Erkek']

kadinlar=data[data['Cinsiyet']=='Kadin']



kadinlar.head()

erkekler.head()
data.groupby('Sinif')['Yas','Fiyati'].mean()
veri=data.copy()

veri.count()
veri.dropna(inplace=True)

veri.count()
veri['Fiyati']=veri.Fiyati.astype('int64')

veri.head()
veri.apply(np.max)