# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
veriler = pd.read_csv('/kaggle/input/people/people.csv')
#boy = veriler['boy']#series

boy = veriler[['boy']]#dataFrame

#test

print(boy)

boykilo = veriler[['boy','kilo']]

print(boykilo)


from sklearn.impute  import SimpleImputer



imputer= SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=0, copy=True)   



Yas = veriler.iloc[:,1:4].values

#print(Yas)

imputer = imputer.fit(Yas[:,1:4])

Yas[:,1:4] = imputer.transform(Yas[:,1:4])

print(Yas)
ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories='auto')

ulke=ohe.fit_transform(ulke).toarray()

print(ulke)
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )

print(sonuc)



sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])

print(sonuc2)



cinsiyet = veriler.iloc[:,-1].values

print(cinsiyet)



sonuc3 = pd.DataFrame(data = cinsiyet , index=range(22), columns=['cinsiyet'])

print(sonuc3)



s=pd.concat([sonuc,sonuc2],axis=1)

print(s)



s2= pd.concat([s,sonuc3],axis=1)

print(s2)
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

print(x_train)

print(x_test)

print(y_train)

print(y_test)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)



print(X_train)

print("    **********")

print(X_test)
