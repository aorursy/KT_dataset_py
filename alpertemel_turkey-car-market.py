# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("/kaggle/input/turkey-car-market-2020/turkey_car_market.csv")

data.head()
değişkenler = data.columns

değişkenler
data.isnull().sum()/len(data)
for i in değişkenler:



    print(i, len(data.loc[data[i] == "Bilmiyorum"]))
hp = data.loc[data["Beygir Gucu"] != "Bilmiyorum"]

hp.head()



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



cols = ['Marka', 'Arac Tip Grubu', 'Arac Tip','Yakıt Turu', 'Vites', 'CCM', 'Beygir Gucu', 'Renk', 'Kasa Tipi',

       'Kimden', 'Durum']



for i in  cols:

    hp[i] = le.fit_transform(hp[i])



corr = hp.corr()

corr["Beygir Gucu"]
q1 = data["Fiyat"].quantile(0.25)

q3 = data["Fiyat"].quantile(0.75)



IOC = q3 - q1



alt_sınır = q1 - 1.5*IOC

üst_sınır = q3 + 1.5*IOC



kontrol = (data["Fiyat"] < alt_sınır) | (data["Fiyat"] > üst_sınır)

data["Asırı_Deger"] = kontrol

data["Asırı_Deger"].value_counts()

data = data.loc[data["Asırı_Deger"] == False]



data.head()
round(pd.pivot_table(data = data, columns = "Marka", values = "Fiyat")).T
fig, ax = plt.subplots(figsize=(20, 6))

sns.barplot(x = data.loc[data["Marka"] == "Opel"]["Model Yıl"], y = data["Fiyat"])
fig, ax = plt.subplots(figsize=(20, 6))

sns.barplot(x = data.loc[data["Marka"] == "Renault"]["Model Yıl"], y = data["Fiyat"])
fig, ax = plt.subplots(figsize=(20, 6))

sns.barplot(x = data.loc[data["Marka"] == "Hyundai"]["Model Yıl"], y = data["Fiyat"])
fig, ax = plt.subplots(figsize=(20, 6))

sns.barplot(x = data.loc[data["Marka"] == "Honda"]["Model Yıl"], y = data["Fiyat"])
honda = data.loc[data["Marka"] == "Honda"]

honda_2006 = honda.loc[honda["Model Yıl"] == 2006]

honda_2006
arazi = data.loc[data["Kasa Tipi"] == "Arazi Aracı"]

arazi_2006 = arazi.loc[arazi["Model Yıl"] == 2006]

arazi_2006
ccm = data.loc[data["CCM"] == "1601-1800 cc"]

ccm_2006 = ccm.loc[ccm["Model Yıl"] == 2006]

ccm_2006
data = data.loc[data.index != 3014]



fig, ax = plt.subplots(figsize=(20, 6))

sns.barplot(x = data.loc[data["Marka"] == "Honda"]["Model Yıl"], y = data["Fiyat"])
sns.barplot(x = data["Yakıt Turu"], y = data["Fiyat"], data = data)
fig, ax = plt.subplots(figsize = (20, 6))

sns.barplot(x = data["Kasa Tipi"], y = data["Fiyat"], data = data)
sns.barplot(x = data["Vites"], y = data["Fiyat"], data = data)
sns.barplot(x = data["Durum"], y = data["Fiyat"], data = data)
sns.barplot(x = data["Kimden"], y = data["Fiyat"], data = data)
fig, ax = plt.subplots(figsize = (25, 6))

sns.barplot(x = data["Renk"], y = data["Fiyat"])
turuncu = data.loc[data["Renk"] == "Turuncu "]

turuncu["Kasa Tipi"].value_counts()
df = pd.read_csv("/kaggle/input/turkey-car-market-2020/turkey_car_market.csv")



fig, ax = plt.subplots(figsize = (25, 6))

sns.barplot(x = df["Renk"], y = df["Fiyat"])
data.head()
del data["Asırı_Deger"]



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



for i in cols:

    data[i] = le.fit_transform(data[i])

    

y = data["Fiyat"]

x = data.iloc[:, 1:14]



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 45)



x_train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score





rf = RandomForestRegressor()

rf.fit(x_train, y_train)

rf_tahmin = rf.predict(x_test)



r2_skor = r2_score(y_test, rf_tahmin)

hata_skor = np.sqrt(mean_squared_error(y_test, rf_tahmin))



print("R2_skoru: ", r2_skor)

print("Hata Kare: ", hata_skor)
sonuc = pd.DataFrame({'Gerçek Değerler': np.array(y_test).flatten(), 'Tahminler': rf_tahmin.flatten()})

sonuc.head(10)
import statsmodels.api as sm

X_1 = sm.add_constant(x)



model = sm.OLS(y,X_1).fit()

model.pvalues



cols2 = list(x.columns)

pmax = 1

while (len(cols2)>0):

    p= []

    X_1 = x[cols2]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols2)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols2.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols2

print(selected_features_BE)
Arac_Yası = 2020 - data["Model Yıl"]



# standart sapma 1 olacak şekilde



KM2 = data["Km"] / np.std(data["Km"])

KM2.std()

data["Arac Yası"] = Arac_Yası



data["Km"] = KM2



del data["Model Yıl"]



data.head()
df = data.copy()



del df["İlan Tarihi"]



for i in cols:

    df[i] = le.fit_transform(df[i])



y = df["Fiyat"]

x = df.iloc[:, 0:12]

x2 = df.iloc[:, 13:14]

x = pd.concat([x, x2], axis = 1)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 123)



rf.fit(x_train, y_train)

rf_tahmin_2 = rf.predict(x_test)



r2_skor2 = r2_score(y_test, rf_tahmin_2)

hata_skor2 = np.sqrt(mean_squared_error(y_test, rf_tahmin_2))



print("İlk R2_skoru: ", r2_skor)

print("İlk Hata Kare: ", hata_skor)

print("Yeni R2_skoru: ", r2_skor2)

print("Yeni Hata Kare: ", hata_skor2)
from xgboost import XGBRegressor



xgb = XGBRegressor()

xgb.fit(x_train, y_train)

xgb_tahmin = xgb.predict(x_test)



r2_skor2 = r2_score(y_test, xgb_tahmin)

hata_skor2 = np.sqrt(mean_squared_error(y_test, xgb_tahmin))



print(r2_skor2)

print(hata_skor2)
sonuc = pd.DataFrame({'Gerçek Değerler': np.array(y_test).flatten(), 'Tahminler': xgb_tahmin.flatten()})

sonuc.head(50)
from sklearn.model_selection import GridSearchCV



xgb_params = {



        'learning_rate': [0.01, 0.1, 0.05],

        'max_depth': [3, 5, 7, 10],

        'min_child_weight': [1, 3, 5, 7, 9],

        'subsample': [0.5, 0.7, 0.6, 0.4],

        'colsample_bytree': [0.5, 0.7, 0.9],

        'objective': ['reg:squarederror']    



}



xgb_model = XGBRegressor()

xgb_tune = GridSearchCV(xgb_model, xgb_params, cv = 10, n_jobs = -1, verbose = 2)



xgb_tune.fit(x_train, y_train)
xgb_tune.best_params_
xgb = XGBRegressor(colsample_bytree = 0.7,

 learning_rate = 0.1,

 max_depth = 10,

 min_child_weight = 7,

 objective = 'reg:squarederror',

 subsample = 0.7)



xgb.fit(x_train, y_train)

xgb_tune_tahmin = xgb.predict(x_test)



r2_skor = r2_score(y_test, xgb_tune_tahmin)

hata_skor = np.sqrt(mean_squared_error(y_test, xgb_tune_tahmin))



print(r2_skor)

print(hata_skor)
sonuc = pd.DataFrame({'Gerçek Değerler': np.array(y_test).flatten(), 'Tahminler': xgb_tahmin.flatten()})

sonuc.head(50)