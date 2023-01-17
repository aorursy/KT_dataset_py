# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import data

data=pd.read_csv("../input/maaslar.csv")
data.head()
#maas ve eğitim seviyesi arasında ilişki kurmak istiyorum

x=data.iloc[:,1:2] #Eğitim seviyesi kolonunu al //slicing

y=data.iloc[:,2:] #Eğitim seviyesi kolonunu al
#linear regression

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x,y) # x' ten y yi öğren

print(type(x))



plt.scatter(x,y) # Eğitim seviyesi ev maaşı 2 boyutlu uzaya dağıt

plt.plot(x,lr.predict(x),color="red") #Her bir x e karşılık gelen tahminler göster
#polynomial regression //multiple regression olarak düşünülebilir.

from sklearn.preprocessing import PolynomialFeatures

poly_reg= PolynomialFeatures(degree=2) #x^0, x^1, x^2 yi alacaz ve sistemi bu x lerin çarpanlarını öğretecez amaç bu. 

x_poly= poly_reg.fit_transform(x)

print(x_poly)



lin_reg2= LinearRegression()

lin_reg2.fit(x_poly,y)

plt.scatter(x,y)

plt.plot(x,lin_reg2.predict(x_poly),color="red")

plt.show()
#4. dereceden denemek istersek daha iyi sonuç alabiliriz. Ama bu veri kümesine özel bir durum olabilir.

#polynomial regression //multiple regression olarak düşünülebilir.

from sklearn.preprocessing import PolynomialFeatures

poly_reg= PolynomialFeatures(degree=4) #x^0, x^1, x^2 yi alacaz ve sistemi bu x lerin çarpanlarını öğretecez amaç bu. 

x_poly= poly_reg.fit_transform(x) #Eğitim süreci

print(x_poly)



lin_reg2= LinearRegression()

lin_reg2.fit(x_poly,y)



#visualization

plt.scatter(x,y)

plt.plot(x,lin_reg2.predict(x_poly),color="red")

plt.show()
#tahminler

print(lr.predict([[11]]))

print(lr.predict([[6]]))



#ploynomial predict için öncelikle değei polynoöial dünyaya transform etmek gerekiyor

print(lin_reg2.predict(poly_reg.fit_transform([[6]])))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
#SVR Support Vector Machine

data=pd.read_csv("../input/maaslar.csv")

data.head(10)
#öncelikle ölçekleme işlemi gerçekleştiriyoruz

from sklearn.preprocessing import StandardScaler

sc1= StandardScaler()

# x ve y daha önce slicing edilmiş kolonlar maaş ve eğitim seviyesi

x_olcekli= sc1.fit_transform(x)

sc2= StandardScaler()

y_olcekli= sc2.fit_transform(y)



from sklearn.svm import SVR



svr_reg= SVR(kernel="rbf")

svr_reg.fit(x_olcekli,y_olcekli)



plt.scatter(x_olcekli,y_olcekli, color="red")

plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")

plt.show() #show ile bu plt yi çizdir ve bitir diyoruz. Demezsek bundan sonrakileri aynı plot üzerine çizer.

print(svr_reg.predict([[11]]))

print(svr_reg.predict([[6.6]]))

#Decision Tree

data= pd.read_csv("../input/veriler.csv")
#Decisin Tree için ölçeklemeye gerek yok

from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)

r_dt.fit(x,y) #Var olan 10 değeri öğrendi



plt.scatter(x,y)

plt.plot(x,r_dt.predict(x))

plt.show()
#Tahmin #decision tree bir aralık olarak tahmin yapar yani belli aralıktaki değerler için aynı değerleri döndürür.

print(r_dt.predict([[11]])) #10 dan sonraki tüm değerleri 50000 olark tahmin etti.

print(r_dt.predict([[6.6]])) # 6.5' dan sonra 7' nin sınıfı olan 10000 olarak tahmin etti
#Ensemle Learning: Kollektif Öğrenme

#Birden fazla sınıflandırma ya da tahmin algoritması aynı anda kullanılarak hata oranı düşürülebilir.

#Rassal Orman ağaçları biden fazla decision tree kullandığı için ensemble learning' dir.

#Amaç veri kümesini birden fazla küçük parçaya bölüp her parçadan farklı bir karar ağacı oluşturmak sonrasında ise sonuçları birleştirmek

from sklearn.ensemble import RandomForestRegressor

rfr= RandomForestRegressor(n_estimators=10, random_state=0) #iki parametre vermemiz gerekir. 

#Biri random_state diğeri ise kaç tane decision tree çizileceğini söyleyen n_estimator

rfr.fit(x,y)



#tahmin

print(rfr.predict([[6.6]])) #ortak bir değer döndürür tahmin aşamasında



#visualization

plt.scatter(x,y,color="red")

plt.plot(x,rfr.predict(x),color="blue")

plt.show()

#R2 değeri. 1' e yaklaştıkça model iyi, 0' a yaklaştıkça model kötü!
from sklearn.metrics import r2_score

print("Random Forest R2 değeri: ", r2_score(y, rfr.predict(x)))

print("Decision Tree R2 değeri: ", r2_score(y, r_dt.predict(x)))

#Decision Tree var olan verilere göre ezbe yaptığı için sonuç mükemmel ama tahminler noktasında sıkıntı. 

#Dolayısıyla direk r2 score a bakıp yorum yapmak bizi hataya sevkedebilir.

print("SVR R2 değeri: ", r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print("Polynomial Regression R2 değeri: ", r2_score(y, lin_reg2.predict(poly_reg.fit_transform(x))))

print("Linear Regression R2 değeri: ", r2_score(y, lr.predict(x)))

o2_data= pd.read_csv("../input/maaslar_yeni.csv")

o2_data.head()
x=o2_data.iloc[:,2:3].values

y=o2_data.iloc[:,-1:].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33)



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)

Y_test=sc.fit_transform(y_test)
#Linear regression 

from sklearn.linear_model import LinearRegression

lr= LinearRegression()

lr.fit(X_train,y_train)



import statsmodels.regression.linear_model as sm

r_ols=sm.OLS(lr.predict(x),x)

r=r_ols.fit()

print(r.summary())



tahmin=lr.predict(X_test)

print("Tahmin:\n",tahmin)

y_test
lr_data=pd.read_csv("../input/veriler.csv")
lr_data
x= lr_data.iloc[5:,1:4].values #bağımsız değişkenler. 5' ten sonraki verileri aldık çünkü ilk 5 veri outliers, modeli bozuyor!

y= lr_data.iloc[5:,4:].values #bağımlı değişken

print(x)

print(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33, random_state=0)



from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train=sc.fit_transform(x_train) # x_train' den öğren ve transform et

X_test= sc.transform(x_test) # x_test te ise öğrenmiş olduğun yöntemi kullan



from sklearn.linear_model import LogisticRegression

logr= LogisticRegression(random_state=0)

logr.fit(X_train,y_train)



y_pred= logr.predict(X_test)

print(y_pred)

print(y_test)



from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=5, metric= "minkowski")

knn.fit(X_train,y_train)

y_pred= knn.predict(X_test)

cm= confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.svm import SVC

svc= SVC(kernel="rbf")

svc.fit(X_train,y_train)



y_pred= svc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print("SVC")

print(cm)

print("print accuracy of SVC algo: ",svc.score(X_test,y_test))

from sklearn.naive_bayes import GaussianNB

gnb= GaussianNB()

gnb.fit(X_train,y_train)



y_pred= gnb.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print("GNB")

print(cm)

print("print accuracy of naive bayes algo: ",gnb.score(X_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dtc= DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train,y_train)



y_pred=dtc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print("GNB")

print(cm)

print("print accuracy of decision tree algo: ",dtc.score(X_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(n_estimators=10, criterion="entropy")



rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print("Random Forest Tree")

print(cm)

print("print accuracy of Random Forest Tree algo: ",rfc.score(X_test,y_test))