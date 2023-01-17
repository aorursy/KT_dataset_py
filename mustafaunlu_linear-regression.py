import numpy as np # lineer cebir işlemleri için

import pandas as pd # verinin organizasyonu için

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

#Grafik çizdirme kütüphanesi

import matplotlib.pyplot as plt



import os #Sistem 

import warnings #uyarılar

print(os.listdir("../input/linearregressiondataset"))

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/linearregressiondataset/linear-regression-dataset.csv",sep = ";")

#dataset tanımlama
plt.scatter(df.deneyim,df.maas)

plt.xlabel("deneyim")

plt.ylabel("maas")

plt.show()

#deneyim ve maas grafiksel gösterim
# linear regression model olusturma

linear_reg = LinearRegression()



x = df.deneyim.values.reshape(-1,1)

y = df.maas.values.reshape(-1,1)

"""Şimdi veri çerçevesini eğitim ve test setlerine ayıralım (%20):"""

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=9)

linear_reg.fit(X_train,y_train)
b0 = linear_reg.predict(np.array(0).reshape(-1,1))

print("b0: ",b0)

#fit edilmis modeli kullanarak 0 deneyim maasını tahmin ettirme
b0_ = linear_reg.intercept_

print("b0_: ",b0_)   # y eksenini kestigi nokta intercept



b1 = linear_reg.coef_

print("b1: ",b1)   # egim slope
####

# maas = 1809 + 1151*deneyim formulu ile tanimlayabiliriz...

maas_yeni = 1809.81189224 + 1151.87310697*11

print(maas_yeni)

print(linear_reg.predict(np.array(11).reshape(-1,1)))

#goruldugu gibi iki sonucta ayni cikmaktadir..

####





# visualize line

plt.scatter(x, y, color = "m", 

			marker = "o", s = 30)





array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim

y_head = linear_reg.predict(array)  # maas



plt.plot(array, y_head,color = "red")

plt.xlabel('x') 

plt.ylabel('y')

plt.show()

print("deneyim 100 iken maas: "+str(int(linear_reg.predict(np.array(100).reshape(-1,1))[0][0]))+" TL")

print("deneyim 10 iken maas: "+str(int(linear_reg.predict(np.array(10).reshape(-1,1))[0][0]))+" TL")

print("deneyim 1 iken maas: "+str(int(linear_reg.predict(np.array(1).reshape(-1,1))[0][0]))+" TL")
pred = linear_reg.predict(X_test)

test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

test_set_r2 = r2_score(y_test, pred)



print("MSE: "+str(test_set_rmse))

print("R-2: "+str(test_set_r2))
import numpy as np 

import matplotlib.pyplot as plt 



def estimate_coef(x, y): 

	# gözlem / puan sayısı

	n = np.size(x) 



	# x ve y vektörünün ortalaması 

	m_x, m_y = np.mean(x), np.mean(y) 



	# x ile ilgili çapraz sapma ve sapmanın hesaplanması 

	SS_xy = np.sum(y*x) - n*m_y*m_x 

	SS_xx = np.sum(x*x) - n*m_x*m_x 



	# regresyon katsayılarının hesaplanması 

	b_1 = SS_xy / SS_xx 

	b_0 = m_y - b_1*m_x 



	return(b_0, b_1) 



def plot_regression_line(x, y, b): 

    # SADECE GORSEL AMACLI

	# gerçek noktaları dağılım grafiği olarak çizmek

	plt.scatter(x, y, color = "m", 

			marker = "o", s = 30) 



	# öngörülen yanıt vektörü 

	y_pred = b[0] + b[1]*x 



	# regresyon çizgisinin çizilmesi 

	plt.plot(x, y_pred, color = "g") 



	# etiket koymak

	plt.xlabel('x') 

	plt.ylabel('y') 



	# gösterme işlevi 

	plt.show() 



def main(): 

    #rastgele tutarlı x ve y oluşturma

	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 

	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 



	# tahmin katsayıları 

	b = estimate_coef(x, y) 

	print("tahmin katsayıları:\nb_0 = {} \\nb_1 = {}".format(b[0], b[1])) 



	# x ve y ile regresyon cizgisi cizme 

	plot_regression_line(x, y, b) 



if __name__ == "__main__": 

	main() 

import matplotlib.pyplot as plt 

from sklearn import datasets, linear_model 



# boston veri kümesini yükleyin

boston = datasets.load_boston(return_X_y=False) 



# özellik matrisi (X) ve yanıt vektörünü (y) tanımlama 

X = boston.data 

y = boston.target 



# X ve y'yi eğitim ve test setlerine bölmek

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 

													random_state=1) 



# doğrusal regresyon nesnesi olusturma

reg = linear_model.LinearRegression() 



# eğitim setlerini kullanarak modeli eğitme

reg.fit(X_train, y_train) 



# katsayılar

print('Coefficients: \n', reg.coef_) 



# varyans skoru: 1, mükemmel bir tahmin anlamına gelir 

print('Variance score: {}'.format(reg.score(X_test, y_test))) 



#gorsel

plt.style.use('fivethirtyeight') 



## Eğitim verilerinde artık hataların çizilmesi

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 

			color = "green", s = 10, label = 'Train data') 



## Test verilerinde artık hataların çizilmesi

plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 

			color = "blue", s = 10, label = 'Test data') 



# sıfır hata icin cizgi çizme

plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 



plt.legend(loc = 'upper right') 



## plot başlığı 

plt.title("kalıntı hatalar") 



#goster

plt.show() 
