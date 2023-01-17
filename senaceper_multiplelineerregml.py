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
#MULTİPLE LİNEER REGRESYON



import numpy as np #matematiksel işlemler

import pandas as pd #verileri okumak için

import matplotlib.pyplot as plt #verileri çizdirmek için

from sklearn.linear_model import LinearRegression #regresyon yapmak için kullanılacak kütüphane

from sklearn.model_selection import train_test_split #test ve train olarak ayırmak için kullanılan kütüphane

from sklearn.metrics import mean_squared_error, r2_score #MSE, 

data= pd.read_csv("../input/student/StudentsPerformance.csv",sep=",") #data okundu



print(data)



x=data.iloc[:,[5,6]].values # datada Math Score alanı ile Reading Score alanları x e atıldı. Yani iki özellik kullanıldı

y=data[["writing score"]]#  bu iki özelliğin writing score ile arasındaki ilişki bulunacak.



#veri seti eğitim ve test olarak ayrıldı. %70lik kısım eğitimde %30luk kısım test kısmında.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)



multiple_linear_reg = LinearRegression()#Lineer Regresyon uygulanması için fonksiyon tanımlandı.

multiple_linear_reg.fit(x_train,y_train)#Makine öğrenmesi gerçekleştirildi.

y_pred=multiple_linear_reg.predict(x_test)#x değişkeninin test değerleri tahmin edilerek y tahmin değerleri hesaplandı.



print('Eğim değeri:', multiple_linear_reg.coef_)

print('Katsayılar [b1,b2]: ', multiple_linear_reg.intercept_)

print("Hata oranı: %.2f" % mean_squared_error(y_test, y_pred))

accuracy = multiple_linear_reg.score(x_test, y_test)#accuracy hesaplandı

print("Accuracy : ", accuracy)