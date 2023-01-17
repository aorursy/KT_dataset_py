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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

#Kullanılacak kütüphaneler tanımlandı.





df=pd.read_csv("../input/student/StudentsPerformance.csv",sep=",")#Kullanılacak veri seti okundu df(dataframe)e atıldı.



#math score ile writing score arasındaki ilişki için regresyon uygulandı

x=df[['math score']]#math score alanı x değişkenine

y=df[['writing score']]#Writing score alanı y değişkenine atıldı.

x=pd.DataFrame.as_matrix(x) # data matrise dönüştürüldü

y=pd.DataFrame.as_matrix(y)



#veri seti eğitim ve test olarak ayrıldı. %70lik kısım eğitimde %30luk kısım test kısmında.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)





linear_reg = LinearRegression()#Lineer Regresyon uygulanması için fonksiyon tanımlandı.

linear_reg.fit(x_train, y_train)#Makine öğrenmesi gerçekleştirildi.

y_pred = linear_reg.predict(x_test)#x değişkeninin test değerleri tahmin edilerek y tahmin değerleri hesaplandı.



#y=b0+b1*x

print('Kat sayı değeri:', linear_reg.coef_)#b0 değeri

print('Eğim : ',linear_reg.intercept_)#b1 değeri

print("Hata oranı: %.2f" % mean_squared_error(y_test, y_pred))#MSE Hesaplaması

print('Varyans skoru : %.2f' % r2_score(y_test, y_pred))#Varyans

plt.scatter(x_test, y_test,  c='b')#Grafikte veriler gösterildi.

plt.plot(x_test, y_pred, linewidth=2, c='r')#Doğru çizildi.

plt.xlabel("Math Score")

plt.ylabel("Writing Score")

plt.show()