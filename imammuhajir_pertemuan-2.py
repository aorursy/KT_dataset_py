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
import pandas as pd 

import numpy as np 



from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")
# ini codingan untuk membaca data 

train = pd.read_csv("/kaggle/input/titanic/train.csv") 

test = pd.read_csv("/kaggle/input/titanic/test.csv")
# memunculkan data train

train
# memunculkan data test

test
# ini untuk memunculkan data hanya 5 row data di atas

train.head()
# ini untuk memuncilkan data hanya 5 row di bawah 

train.tail()
# ini code untuk melihat, berapa ukuran data (row, coloumns)

train.shape
# menunjukan, jumlah data, dan type data 

train.info()
# ini untuk data angka (numberic)

# untuk melihat persebarannya secara statistik

train.describe()
# ini mengimport package untuk visualisasi data 

import matplotlib.pyplot as plt

import seaborn as sns
# ini unutuk membuat bar plot, 

sns.barplot(x='Sex', y='Survived', data=train)
# ini untuk ngambil salah satu colomn, 

train["Sex"]

# format data merupakan (series), 
# mengambil 1 Columns

train[['Sex']]

# format data merupakan "DataFrame"
# ini untuk mengambil data lebih dari 1 

train[['Sex', 'Cabin']]
# ini buat ngambil row nya dari data 

train.iloc[2:9]

# ini buat ganti data kategorik jadi data numberik 

# ini memisalkan male = 1, female = 2

sex_map = {"male" : 1,"female" :0 }

train["Sex"] = train['Sex'].map(sex_map)

# ini buat ganti data kategorik jadi data numberik 

sex_map = {"male" : 1,"female" :0 }

test["Sex"] = test['Sex'].map(sex_map)

# cek data, apakah male dan female sudah berubah?

test.head()
# ini untuk menghitung jumlah setiap nilai colomnnya

# dapat juga di gunakan untuk mengetahui, berapa jumlah type dari setiap columns

train['Embarked'].value_counts()
train.head()
# ini untuk mendrop data colomns lebih dari 1

train = train.drop(['Ticket', "Cabin", "Embarked"], axis = 1)

test = test.drop(['Ticket', 'Cabin', "Embarked"], axis = 1)
# ini merupakan data yang sudah numberik semua

train
# kita cek apakah data tersebut sudah bertype int atau float, 

# bila sudah float dan int maka sudah siap untuk di masukan ke dalam model



train.info()



# tetapi masih ada data2 yang null/kosong belum terisi
# ini cara mengisi colomns yang kosong dengan rata2 nya

train['Age'] = train['Age'].fillna(train["Age"].mean())
train.info()
train_data = train.drop("Survived", axis =1 )

target = train['Survived']



train_data.shape, target.shape
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
model = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(model, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

# ini merupakan rata2 accuracy 

round(np.mean(score)*100, 2)
test