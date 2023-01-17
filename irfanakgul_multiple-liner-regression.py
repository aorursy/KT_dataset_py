# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.metrics import mean_squared_error, r2_score



import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

import seaborn as sns

from warnings import filterwarnings

filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/advertising/Advertising.csv", usecols=[1,2,3,4])

df
df.head()
df.corr()
# plot the heatmap

corr = df.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
x = df.drop("sales", axis=1)  # bagimli degisken olan sales(y) atip, tum bagimsiz degiskenleri x'e atadik.

y = df["sales"]   #bagimli degiskenimiz
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)
lm = LinearRegression()

model = lm.fit(x_train,y_train)
model.intercept_   #sabit katsayi > b=0 > x=0 dayken y nin aldigi deger.
model.coef_   

#modelimizin tum bagimsiz degiskenlerinin katsayisi yani her bir bagimsiz degiskeni bir birim arttirdigimizda 

#..satislarda ne kadarlik bir artis saglanir.
yeni_veri = [[30], [10],[40]]   # bu uc sayilik artis satislarin nasil etkiler fakat bunun dataframe cevirilmesi gerek.

yeni_veri = pd.DataFrame(yeni_veri).T  # .T tabloyu cevirir.

model.predict(yeni_veri) 



# tv, radio ve gazete reklamlarinda [30], [10],[40] birimleri kadar yapilan artislarin, ..

#...satislara ne kadar etki ettigini tahmin edecek
rmse = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))

# hata_payi = egitim(train) setimizdeki gercek y ile tahmin edilen bgmsiz 

#...degiskenlerinin hatakarelerinin ortalamalarinin karekokunu aldik.

rmse
rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

# hata_payi = test setimizdeki gercek y ile tahmin edilen bgmsiz 

#...degiskenlerinin hatakarelerinin ortalamalarinin karekokunu aldik.

rmse



# NOT > Egitim hatasi ile test hatasi arasindaki fark cok degilse modelleme iyidir yorumu yapilabilir. 
model.score(x_train, y_train) 
# Valide (dogrulanmis) edilmis r2  degero; 

cross_val_score(model, x_train,  y_train, cv= 10, scoring="r2").mean()



#Modelimizden "10" tane birbirinden farkli "r2" degeri uretir. ve bunun ortalamasini aldik(mean). 

#iste bizim asil dogrulanmis r2 degerimiz budur,
#gec

#np.sqrt(-cross_val_score(model, x_train,  y_train, cv= 10, scoring="neg_mean_squared_error").mean())



#Modelimizden "10" tane birbirinden farkli "neg_mean_squared_error(egitim hatasi)" degeri uretir. 

#ve bunun ortalamasini aldik(mean). 

#iste bizim asil dogrulanmis r2 degerimiz budur,

# Not uretilen deger negatif olabilir. basina eksi(-) koyarak pozitif yaptik



#tUM BU BIZIM TRAIN VERIMIZDIR. KIYAS YAPMAK ICIN TEST VERIMIZINDE AYNI ISLEMLERI UYGULAYIP ARASINDA KI FARKA BAKMAK LAZIM, 

#...EGER FARK YAKINSA MODELIMIZ DOGRUDUR,IYIDIR, GUZELDIR 



# GERCEK HATAMIZ ISTE BU SEKILDE ALINIR,,, 

# egitim tahminlerimiz : 

y_head = model.predict(x_test)

y_head[0:5]
y_test_1 =np.array(range(0,len(y_test)))

y_test_1
# r2 degerimiziz : 

r2_degeri = r2_score(y_test, y_head)

print("Test r2 hatamiz = ",r2_degeri) 



plt.plot(y_test_1,y_test,color="red")

plt.plot(y_test_1,y_head,color="blue")

plt.show()