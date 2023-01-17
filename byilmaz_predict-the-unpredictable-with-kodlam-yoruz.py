# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

#Birçok Jupyter defter kullanıcısı, defterin başında %matplotlib inline komutunu vermeyi tercih eder. Bu sayede ürettiğiniz matplotlib grafikleri defterin içine gömülür ve tam bir belge oluşturmanızı sağlar.

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input")) #input dosya yolunun altındaki dosyaları listeliyor ve ekrana yazdırıyor.



# Any results you write to the current directory are saved as output.
pd.options.display.precision = 20 # virgülden sonra kaç basamak olucağının hassalık ayarı

train = pd.read_csv("../input/train.csv", nrows=10000000,

                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}) 

#pandas ile csv dosyası okunuyor.

#1 Dosya yolu

#2 Kaç satır okunacak

#3 Veri tipleri belirleniyor





train.head(10) #ilk 10 datayı göster
train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)

#Columns doğrusundaki isimleri anlamlı isimlerle değiştiriyoruz.

train.head() # ilk 5 satırı (default olarak 5) yazdırıyoruz.
plt.rcParams["figure.figsize"] = (40,10) #asagidaki grafiğin ekrandaki boyutunu belirliyoruz.
import matplotlib.pyplot as plt

plt.plot(train.quaketime.values[:20], train.signal.values[:20], 'ro') #Grafiği times ve signals listelerine göre çizdiriyoruz. Ilk 20 deger

#r kırmızı demek. o ise yuvarlak demek. ro diyerek grafikte kırmızı noktalamalar yapıyoruz.



plt.axis([train.quaketime.values[0]-0.0000000001, train.quaketime.values[20] + 0.0000000001, -10, 10]) 

#axisleri belirliyoruz. 

#X ekseni times dizisinin ilk verisinin 0.0000000001 eksiğinden times dizisinin son verisinin 0.0000000001 fazlası aralığında

#Y ekseni -10 ve 10 aralığında

plt.show() # ekrana çizdiriyoruz.
print(train.describe()) #DF(DATAFRAME)'imizin belirli istatistiklerine bakıyoruz.
print(train['signal'].unique())  # signal kolonundaki verilerin sadece benzersiz olanlarını yazdırıyoruz

print(type(train['signal'].unique()))  # Signal kolonundaki benzersiz verilerin tipine bakiyoruz. Sonrasinda ona gore islem yapmak icin.

print(np.sort(train['signal'].unique())) # Numpy array oldugunu bildigimiz datayi siraliyoruz. Dizinin taban ve tavan degerlerini gormek icin
meanDF = train.groupby(['signal'], as_index=False).mean() #sinyale göre gruplayıp ortalamasını alıyoruz. # Indexliyoruz

meanDF.head() # ekrana yazdırıyoruz.(default 5)
meanDF.describe() # ortalamalara gore istatisik bilgileri
meanDF.loc[meanDF['quaketime'] == 5.73671644094828270255].head() # Maximum ortalama degere sahip signal
plt.rcParams["figure.figsize"] = (40,15) #grafiğin ekrandaki boyutunu belirliyoruz.
plt.plot(meanDF.signal.values[:200], meanDF.quaketime.values[:200], 'ro') #Grafiği times ve signals listelerine göre çizdiriyoruz. 

#r kırmızı demek. o ise yuvarlak demek. ro diyerek grafikte kırmızı noktalamalar yapıyoruz.

#axisleri belirliyoruz. 

plt.xlabel("Signals")

plt.ylabel("Times")

plt.title("Ortalamaya gore degerler")

plt.show() # ekrana çizdiriyoruz.

plt.rcParams["figure.figsize"] = (200,40) #grafiğin ekrandaki boyutunu belirliyoruz.
plt.plot(meanDF.signal.values[:200], meanDF.quaketime.values[:200], 'r') #Grafiği times ve signals listelerine göre çizdiriyoruz.  r kirmizi demek

#axisleri isimlerimi belirliyoruz. 

plt.xlabel("Signals")

plt.ylabel("Times")

plt.title("Ortalamaya gore degerler")

plt.show() # ekrana çizdiriyoruz.
sampleMeanDF = meanDF.sample(200 , random_state=1) # verilerin icinden rastgele 200 tanesini aliyoruz # Random state =1 her seferinde ayni degerleri versin diye

sampleMeanDF.sort_values(by=['signal'], inplace=True) # signale gore kucukten buyuge siraliyoruz

sampleMeanDF.head(10) # ilk 10 degeri ekrana bastiriyoruz,
plt.rcParams["figure.figsize"] = (200,40) #grafiğin ekrandaki boyutunu belirliyoruz.
plt.plot(sampleMeanDF.signal.values, sampleMeanDF.quaketime.values, 'ro') #Grafiği times ve signals listelerine göre çizdiriyoruz. 

#r kırmızı demek. o ise yuvarlak demek. ro diyerek grafikte kırmızı noktalamalar yapıyoruz.

#axisleri belirliyoruz. 

plt.xlabel("Signals")

plt.ylabel("Times")

plt.title("Orneklenmis ortalamaya gore degerler")

plt.show() # ekrana çizdiriyoruz.

plt.rcParams["figure.figsize"] = (40,10) #grafiğin ekrandaki boyutunu belirliyoruz.
plt.plot(sampleMeanDF.signal.values, sampleMeanDF.quaketime.values, 'r') #Grafiği times ve signals listelerine göre çizdiriyoruz. 

#r kırmızı demek. o ise yuvarlak demek. ro diyerek grafikte kırmızı noktalamalar yapıyoruz.

#axisleri belirliyoruz. 

plt.xlabel("Signals")

plt.ylabel("Times")

plt.title("Orneklenmis ortalamaya gore degerler")

plt.show() # ekrana çizdiriyoruz.
sampleMeanDF.describe() # sample ornegin istatistik bilgisi
halfSampleDF = train.sample(frac=0.5, random_state=4) # verilerin icinden rastgele yarsini aliyoruz # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html

halfSampleDF.sort_values(by=['signal'], inplace=True) # signale gore kucukten buyuge siraliyoruz

halfSampleDF.head(10) # ilk 10 degeri ekrana bastiriyoruz,
plt.rcParams["figure.figsize"] = (200,40) #grafiğin ekrandaki boyutunu belirliyoruz.
plt.plot(halfSampleDF.signal.values, halfSampleDF.quaketime.values, 'ro') #Grafiği times ve signals listelerine göre çizdiriyoruz. 

#r kırmızı demek. o ise yuvarlak demek. ro diyerek grafikte kırmızı noktalamalar yapıyoruz.

#axisleri belirliyoruz. 

plt.xlabel("Signals")

plt.ylabel("Times")

plt.title("Orneklenmis ortalamaya gore degerler")

plt.show() # ekrana çizdiriyoruz.

plt.rcParams["figure.figsize"] = (40,10) #grafiğin ekrandaki boyutunu belirliyoruz.
plt.plot(halfSampleDF.signal.values, halfSampleDF.quaketime.values, 'r') #Grafiği times ve signals listelerine göre çizdiriyoruz. 

#r kırmızı demek. o ise yuvarlak demek. ro diyerek grafikte kırmızı noktalamalar yapıyoruz.

#axisleri belirliyoruz. 

plt.xlabel("Signals")

plt.ylabel("Times")

plt.title("Orneklenmis ortalamaya gore degerler")

plt.show() # ekrana çizdiriyoruz.
from sklearn.preprocessing import PolynomialFeatures # ortalama degerlere gore Polynomial Regression

from sklearn.linear_model import LinearRegression   # https://github.com/krishnaik06/Polynomial-Linear-Regression/blob/master/polynomial_regression.py

X = meanDF.signal.values.reshape(2065,1)

y = meanDF.quaketime.values.reshape(2065,1)

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)
plt.rcParams["figure.figsize"] = (200,40) #grafiğin ekrandaki boyutunu belirliyoruz.
plt.scatter(X, y, color = 'red')

plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('(Polynomial Regression)')

plt.xlabel('signal')

plt.ylabel('times')

plt.show()
x_test = np.array([4,8]).reshape(2,1)

lin_reg_2.predict(poly_reg.fit_transform(x_test))
sortedTrain = train.sort_values(by=['signal'],) # signale gore kucukten buyuge siraliyoruz

sortedTrain.head()
sortedTrain.reset_index()

sortedTrain.head()
X = sortedTrain.signal.values.reshape(10000000 ,1)     # Tum degerlere gore Polynomial Regression

y = sortedTrain.quaketime.values.reshape(10000000 ,1)  # https://github.com/krishnaik06/Polynomial-Linear-Regression/blob/master/polynomial_regression.py

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)
x_test = np.array([4,8]).reshape(2,1)

lin_reg_2.predict(poly_reg.fit_transform(x_test))