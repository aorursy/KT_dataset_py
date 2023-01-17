#iKullanacağımız Kütüphaneler.

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

%matplotlib inline
#Çalışmada kullanacağımız veri setini ekliyoruz.

egitim = pd.read_csv("../input/train.csv")
egitim.head()
egitim.shape
egitim.info()
plt.subplots(figsize=(12, 9))

sns.distplot(egitim['SalePrice'], fit = stats.norm)



(mu, sigma) = stats.norm.fit(egitim['SalePrice'])



#Şimdi oluşturduğumuz dağılımı çizdirelim.

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma = $ {: .2f})'.format(mu, sigma)], loc = 'best')

plt.ylabel('Frekans')



#Olasılık durumunuda çizdirelim

fig = plt.figure()

stats.probplot(egitim['SalePrice'], plot = plt)

plt.show()
#Evet güzel yakışıklı kütüphanelerimizden  by numpy kütüphanesi bize logaritmik dönüşümde yardımcı olacak

egitim['SalePrice'] = np.log1p(egitim['SalePrice'])



#Şimdi normal dağılımı tekrar kontrol edelim



plt.subplots(figsize =(12, 9))

sns.distplot(egitim['SalePrice'], fit = stats.norm)



#

(mu, sigma) = stats.norm.fit(egitim['SalePrice'])



#Şimdi dağılımı görselleştirelim

plt.legend(['Normal Dağılım. ($\mu=$ {:.2f} and $\sigma = $ {:.2f} )' .format(mu, sigma)], loc = 'best')

plt.ylabel('Log Aldıktan sonraki Frekans')



#Olasılık durumunu görselleştirelim

fig = plt.figure()

stats.probplot(egitim['SalePrice'], plot = plt)

plt.show()
#Bakalım ne kadar eksik veri mevcut eğitim verisetimizde

egitim.columns[egitim.isnull().any()]
#Gelin bu eksik verilerin bir grafiğini çizelim

plt.figure(figsize =(12, 6)) # bu bizim grafik çerçevemizi belirliyor

sns.heatmap(egitim.isnull()) # seaborn kütüphanesinden değerli arkadaşımız heatmak görseli bize yardımcı oluyor 

plt.show()
#Şimdi gelin her bir sütunda bulunan  bu eksik değerlerin ne kadar olduklarına bakalım.

Isnull = egitim.isnull().sum() / len(egitim) * 100

Isnull = Isnull[Isnull > 0]

Isnull.sort_values(inplace = True, ascending = False)

Isnull
#Önce verileri dönüştürelim

Isnull = Isnull.to_frame()
#Şimdi Her bir sütundaki değerlerin sayısını alalım

Isnull.columns = ['count']
Isnull.index.names = ['İsimler']
Isnull['Name'] = Isnull.index
#Artık görselleştirelim değil mi 

plt.figure(figsize = (15, 9))

sns.set(style = 'whitegrid')

sns.barplot(x = 'Name', y = 'count', data = Isnull)

plt.xticks(rotation = 90)

plt.show()
#Verilerimizin içerisinden sadece sayısal verisetini ayıralım yukarıdaka görmüştük, 81 tane sütundan 38 tanesi sayısal veri içeriyordu

egitim_corr = egitim.select_dtypes(include = [np.number])
egitim_corr.shape
#Gelin Id sütunu veri setimizden silelim çünkü bu bize korelasyon ilişkisi hakkında bir bilgi vermeyecek 

del egitim_corr['Id']
#Evet Artık korelasyon grafiğimizi çizebiliriz.

corr = train_corr.corr()

plt.subplots(figsize = (20, 11))

sns.heatmap(corr, annot = True)
eniyi_50_deger = corr.index[abs(corr['SalePrice'] > 0.5)]

plt.subplots(figsize = (12, 8))

eniyi_iliski = egitim[eniyi_50_deger].corr()

sns.heatmap(eniyi_iliski, annot = True)

plt.show()
#OverallQual özelliğindeki benzersiz değerlere bakalım

egitim.OverallQual.unique()
#Gelin bunları bir çubuk grafiğinde görelim

sns.barplot(egitim.OverallQual, egitim.SalePrice)
#Birde bu durumları kutu grafiğinde inceleyelim

plt.figure(figsize=(18,10))

sns.boxplot(x = egitim.OverallQual, y = egitim.SalePrice)
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.set(style = 'ticks')

sns.pairplot(egitim[col], size = 3, kind = 'reg')
print('Hedef değişken(SalePrice) ile en iyi ilişkisi olan değişkeni bulalım')

corr = egitim.corr()

corr.sort_values(['SalePrice'], ascending = False, inplace = True)



corr.SalePrice
#Mesela PoolQC değeri yaklaşık %99 boş veriden oluşuyor  bunu None olarak dolduralım

egitim['PoolQC'] = egitim['PoolQC'].fillna('None')
#Özellikler içinde % 50 civarında eksik değer içierenleri None ile dolduralım

egitim['MiscFeature'] = egitim['MiscFeature'].fillna('None')

egitim['Alley'] = egitim['Alley'].fillna('None')

egitim['Fence'] = egitim['Fence'].fillna('None')

egitim['FireplaceQu'] = egitim['FireplaceQu'].fillna('None')
# Mahalleye göre gruplandıralım ve tüm değerleri medyan LotFrontage ile eksik değeri doldurun

egitim['LotFrontage'] = egitim.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
#GarageType, GarageFinish, GarageQual ve GarageCond bunlarıda none ile değiştirelim

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    egitim[col] = egitim[col].fillna('None')
#GarageYrBlt, GarageArea ve GarageCars bunlarıda sıfır ile değiştirelim

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    egitim[col] = egitim[col].fillna(int(0))

#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual None ile değiştirelim

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):

    egitim[col] = egitim[col].fillna('None')

#MasVnrArea'yı da sıfır ile değiştirelim 

egitim['MasVnrArea'] = egitim['MasVnrArea'].fillna(int(0))
#MasVnrType  sütununu da None ile değiştirelim

egitim['MasVnrType'] = egitim['MasVnrType'].fillna('None')
#Mode Değerlerini ekleyelim

egitim['Electrical'] = egitim['Electrical'].fillna(egitim['Electrical']).mode()[0]
#Utilities e de ihtiyacımız yok onuda atalım veri setimizden

egitim = egitim.drop(['Utilities'], axis =1)

#Şimdi gelin bakalım veri setimizin ahvali nasıl, null değer içeriyormu bir kontrol edelim

plt.figure(figsize =(10, 5))

sns.heatmap(egitim.isnull())
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',

        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 

        'SaleType', 'SaleCondition', 'Electrical', 'Heating')
from sklearn.preprocessing import LabelEncoder

for c in cols:

    lbl = LabelEncoder()

    lbl.fit(list(egitim[c].values))

    egitim[c] = lbl.transform(list(egitim[c].values))
#Evet Artık verilerimizi tahminde kullanmak için hazırlıkları tamamlayabiliriz

#Hedef değişkenimiz olan SalePrice değişkenini yeni bir değişkene yani y'ye atayarak başlayalım

y = egitim['SalePrice']
#Tahmin edeceğiimiz hedef değişkeni artık veri setimizden sile biliriz

del egitim['SalePrice']
#X ve Y değerlerini belirleyelim

X = egitim.values

y = y.values
#Şimdi de verilerimizi eğitim ve test verileri olmak üzere ikiye ayıralım (%80 eğitim, % 20 test)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
#Modeli önce eğitelim

from sklearn import linear_model

model = linear_model.LinearRegression()
#Eğittiğimiz modele verilerimizi uygulayalım

model.fit(X_train, y_train)
#Şimdi de eğittiğimiz modelimiz üzerinden tahminler yapalım

print("Tahmin edilen değer : " + str(model.predict([X_test[142]])))

print("Gerçek değer : " + str(y_test[142]))
#Bakalım ne kadar doğrulukla tahmin yapabilmişiz

print("Doğruluk oranı :  ", model.score(X_test, y_test)* 100)
#Modeli Eğitelim yine

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 1000)
#Şimdi veri setimizi model uygulayalım

model.fit(X_train, y_train)
#Modelimizin doğruluğunu kontrol edelim

print("Doğruluk oranı :", model.score(X_test, y_test)*100)
#Son kez modelimizi eğitelim

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators = 100, max_depth = 4)
#Eğittiğimiz modele eğitim verilerimizi  uygulayalım

GBR.fit(X_train, y_train)
#Doğruluk oranımızı da kontrol edelim son olarak

print('Doğruluk Oranı: ', GBR.score(X_test, y_test)*100)