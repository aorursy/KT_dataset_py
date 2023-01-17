import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor 

#Verileri Okuyoruz

train = pd.read_csv('../input/train.csv')

#Verileri hedeflenen y ve tahmincimiz X' e çekiyoruz

train_y = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

#Eğitici tahmin edicileri oluşturuyoruz

train_X = train[predictor_cols]

my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)
#Test datalarını okuyoruz

test = pd.read_csv('../input/test.csv')

#Training data için yaptığımız işlemi Test datası içinde yapacağız, aynı sütünları çekeceğiz

test_X = test[predictor_cols]

#Tahmin işlemi için modelimizi kullanıyoruz

predicted_prices = my_model.predict(test_X)

#Önce bir yazdırıp sonuçların durumuna bakalım.

print(predicted_prices)
my_submission =pd.DataFrame({'Id' : test.Id, 'SalePrice' : predicted_prices})

my_submission.to_csv('submission.csv', index = False)