# Temel Kütüphaneleri çalışma ortamına ekleyerek başlayalım



import numpy as np



import pandas as pd



from pandas import Series, DataFrame



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/Train_UWu5bXk.csv')

test = pd.read_csv('../input/Test_u94Q5KV.csv')
# linear regression algoritmasını sklearn kütüphanesinden yüklüyoruz.



from sklearn.linear_model import LinearRegression



lreg = LinearRegression()



#Veriyi parçalıyoruz, çünkü çapraz doğrulama yapacağız



X = train.loc[:,['Outlet_Establishment_Year','Item_MRP']]



x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales)
#Modelimizi eğitebiliriz.



lreg.fit(x_train,y_train)



pred = lreg.predict(x_cv)



#Hataların karesel ortalamasına bakıyoruz

mse = np.mean((pred - y_cv)**2)
mse
# Katsayılara bir göz atalım(Yani denklemdeki thetalarımız)

coeff = DataFrame(x_train.columns)



coeff['Coefficient Estimate'] = Series(lreg.coef_)



coeff
lreg.score(x_cv,y_cv)

#Not: Sizin yapacağınız çalışmada Rkare değeri farklı çıkabilir.
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace = True)

train['Outlet_Establishment_Year'].fillna((train['Outlet_Establishment_Year'].mean()), inplace = True)
train.head()
#Şimdi Modelimizi tekrar kontrol edelim artık eksik değer olan bir sütun kalmadığından bu işlemei yapabiliriz.

X = train.loc[:,['Outlet_Establishment_Year', 'Item_MRP', 'Item_Weight']]



#Elimizdeki veriyi tekrar parçalıyoruz.

x_train, x_cv, y_train, y_cv = train_test_split(X, train.Item_Outlet_Sales)

#Şimdi de Modeli Eğitelim

lreg.fit(x_train,y_train)
#Modeli eğittiğimize göre göre Şimdi tahminlerde bulunalım

pred = lreg.predict(x_cv)
#Şimdi de Ortalama Kare Hatası'na(mse)'ye bakalım'

mse = np.mean((pred - y_cv)**2)



mse
# Şimdi Katsayıları kontrol edelim bakalım nasıl bir korelasyon ilişkileri var.

coeff = DataFrame(x_train.columns)



coeff['Coefficient Estimate'] = Series(lreg.coef_)



coeff
#Son duruma göre R-Square değerimize bakalım ve tahmin yeteneğimizin ne kadar olduğunu görelim.



lreg.score(x_cv, y_cv)
# Regresyon modeli için veri ön işleme adımları

# Eksik değerleri öncelikle ele alıyoruz ve eksik alanları dolduruyoruz.

train['Item_Visibility'] = train['Item_Visibility'].replace(0,np.mean(train['Item_Visibility']))

train['Outlet_Establishment_Year'] =2013 - train['Outlet_Establishment_Year']



train['Outlet_Size'].fillna('Small', inplace = True)



#Yeni bir değişken oluşturarak kategorik değişkenleri sürekli değişkenlere çeviriyoruz(one -hot-encoding).

mylist = list(train.select_dtypes(include =['object']).columns)



dummies = pd.get_dummies(train[mylist], prefix = mylist)



train.drop(mylist, axis = 1, inplace = True)



X = pd.concat([train, dummies], axis = 1)
from sklearn.linear_model import LinearRegression

# importing linear regression

lreg = LinearRegression()



# for cross validation



from sklearn.model_selection import train_test_split



X = train.drop('Item_Outlet_Sales',1)



x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales, test_size =0.3)
lreg.fit(x_train, y_train)
# Linear modelimizi eğitelim



lreg.fit(x_train,y_train)



# calculating mse



mse = np.mean((x_cv - y_cv)**2)



mse



# evaluation using r-square



lreg.score(x_cv, y_cv)
predictors = x_train.columns



coef = Series(lreg.coef_,predictors).sort_values()



coef.plot(kind='bar', title='Modal Coefficients')
from sklearn.linear_model import Ridge



##training the model



ridgeReg = Ridge(alpha=0.05, normalize=True)



ridgeReg.fit(x_train,y_train)



pred = ridgeReg.predict(x_cv)



##calculating mse



mse = np.mean((pred - y_cv)**2)
mse
from sklearn.linear_model import Ridge



##training the model



ridgeReg = Ridge(alpha=0.5, normalize=True)



ridgeReg.fit(x_train,y_train)



pred = ridgeReg.predict(x_cv)



##calculating mse



mse = np.mean((pred - y_cv)**2)

mse
from sklearn.linear_model import Ridge



##training the model



ridgeReg = Ridge(alpha=5, normalize=True)



ridgeReg.fit(x_train,y_train)



pred = ridgeReg.predict(x_cv)



##calculating mse



mse = np.mean((pred - y_cv)**2)

mse
from sklearn.linear_model import Lasso



lassoReg = Lasso(alpha=0.3, normalize=True)



lassoReg.fit(x_train,y_train)



pred = lassoReg.predict(x_cv)



#calculating mse



mse = np.mean((pred - y_cv)**2)



mse



lassoReg.score(x_cv,y_cv)
from sklearn.linear_model import ElasticNet



ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)



ENreg.fit(x_train,y_train)



pred_cv = ENreg.predict(x_cv)



#calculating mse



mse = np.mean((pred_cv - y_cv)**2)





ENreg.score(x_cv,y_cv) #R-Squar
mse