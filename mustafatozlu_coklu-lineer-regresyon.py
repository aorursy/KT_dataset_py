# Gerekli Tüm kütüphaneler import edildi.

import pandas as pd #csv formattaki veriyi almak için kullanacaz

import numpy as np # matematiksel işlemlerde hızlı hesaplamak için kullanacaz 

import matplotlib.pyplot as plt # grafik çizdirmek için kullanacaz

from sklearn.model_selection import train_test_split #Verilerimizi eğitim ve test için ayırmaya kullancaz

from sklearn import metrics # kütüphane yardımı ile MSE değerini hesaplayacaz

from sklearn.linear_model import LinearRegression #Lineer regresyon yapmak için kullanacaz katsayı hesabı için

%matplotlib inline

plt.rcParams['figure.figsize'] = (8.0,6.0) # çıkan grafiğin boyutunu belirliyoruz

dataset = pd.read_csv('../input/petrol_consumption.csv') #Datasetin okunması

dataset.head() #alınan veriler gösterilmesi için
dataset.describe() #Dataset dosyası hakkında istatistiksel olarak bilgi veriyor
#csv değerlerini ayırıyoruz

X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways',

       'Population_Driver_licence(%)']] #katsayıları bulunması gereken değişkenler

y = dataset['Petrol_Consumption'] #Çıkması gereken gerçek değer
#Eğitim verisi ve test datası olarak bölme işlemi yapıyoruz.

#Verimizin 5te 1i test verisi olarak ayarladık.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
#Eğitim verileri ile modeli oluşturmak için lineer regresyondan fit fonksiyonunu uyguluyoruz  

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#Katsayıları bir önceki fit fonksiyonu ile hesaplanmıştı bunları listeledik.

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Katsayilar'])

coeff_df
#Modelimizi ekrana yazdırdık

print('Modelimiz => Y=', regressor.coef_[0],'*Petrol_Tax +',regressor.coef_[1],'*Average_income +',regressor.coef_[2],'*Paved_Highways + ',regressor.coef_[3],'*population_driver_licence(%)',)
#Oluşturduğumuz modeli ile Test verisinde tahminde bulunuyoruz

y_pred = regressor.predict(X_test)
#Oluşturduğumuz tahminler aşağıda tablo şeklinde yazdırılıyor

#Kıyaslamak için gerçek ve tahmini değeri görebiliyoruz 

df = pd.DataFrame({'Gercek': y_test, 'Tahmin': y_pred})

df
#sklearn kütüphanesi yardımıyla MSE MAE RMSE değerleri hesaplandı. 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))