import numpy as np 

import pandas as pd 



# egitim verisi 

egitim = pd.read_csv('../input/train.csv')

print(egitim.columns)
# ucretler

ucret = egitim.SalePrice

print(ucret.head())
# belirleyicileri tanimlama

belirleyici = ['YearBuilt','1stFlrSF','2ndFlrSF','LotArea','YrSold','FullBath','MSSubClass','TotRmsAbvGrd','BedroomAbvGr']
# belirleyici tablosu

veri_belirleyici = egitim[belirleyici]

veri_belirleyici.head()
from sklearn.tree import DecisionTreeRegressor

# model tanimlama

veri_modeli = DecisionTreeRegressor()

# modeli fitleme

veri_modeli.fit(veri_belirleyici, ucret)

print(veri_modeli.predict(veri_belirleyici.head()))
from sklearn.metrics import mean_absolute_error

# ortalama mutlak hatanin bulunmasi

tahmin = veri_modeli.predict(veri_belirleyici)

mean_absolute_error(ucret, tahmin)
# test verisi kullanilarak fiyat tahmini

test = pd.read_csv('../input/test.csv')

test_belirleyici = test[belirleyici]

tahmini_ucret = veri_modeli.predict(test_belirleyici)

print(tahmini_ucret)
# submission hazirlama

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': tahmini_ucret})

submission.to_csv('submission.csv', index=False)