# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")
test = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv")
store = pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv")

train.head()
train = pd.merge(train, store, on = "Store", how = "left")
test_data = pd.merge(test, store, on = "Store", how = "left")

train.head()
date = train["Date"]
date = pd.DataFrame(date)
date["Date"] = pd.to_datetime(date["Date"])

y = []
i = 0
while i < 1017210:
    x = date["Date"][i].month
    y.append(x)
    i += 1

y = pd.DataFrame(y, columns = ["Month"])
y["Month"].value_counts()

train["Month"] = y["Month"]

y = []
i = 0
while i < 1017210:
    x = date["Date"][i].year
    y.append(x)
    i += 1
y = pd.DataFrame(y, columns = ["Year"])
y["Year"].value_counts()
train["Year"] = y["Year"]

y = []
i = 0
while i < 1017210:
    x = date["Date"][i].day
    y.append(x)
    i += 1
y = pd.DataFrame(y, columns = ["Day"])
train["Day"] = y["Day"]
aylar = pd.pivot_table(train, values='Sales', index = ["Month"],
                        aggfunc=np.sum)

aylar["Month"] = [1,2,3,4,5,6,7,8,9,10,11,12]

import seaborn as sns

sns.barplot(y = "Sales", x = "Month", data = aylar)
ornek_magaza = train.loc[train["Store"] == 13]
x = ornek_magaza.loc[ornek_magaza["Month"] == 3].sum()
y = ornek_magaza.loc[ornek_magaza["Month"] == 11].sum()

print("=======================mart ayı=======================", x, "\n", "=======================kasım ayı=======================",y)
magaza = pd.pivot_table(train, values = "Open", index = ["Store"], aggfunc = np.sum)
max(magaza["Open"])
magaza = magaza.loc[magaza["Open"] == 942]

hep_acik = pd.DataFrame()

for i in magaza.index:
    magazalar = train.loc[train["Store"] == i]
    hep_acik = pd.concat([hep_acik, magazalar], axis = 0)

aylar = pd.pivot_table(hep_acik, values = "Sales", index = ["Month"], aggfunc = np.sum)

aylar["Month"] = [1,2,3,4,5,6,7,8,9,10,11,12]

sns.barplot(x = "Month", y = "Sales", data = hep_acik)
günler = pd.pivot_table(hep_acik, values = "Sales", index = ["DayOfWeek"], aggfunc = np.sum)

günler["Day"] = ["P.tesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "C.tesi", "Pazar"]

sns.barplot(x = "Day", y = "Sales", data = günler)
train = train.loc[train["Open"] == 1]
train["günlük_ort"] = train["Sales"] / train["Customers"]

train.head()
train["CompetitionMonths"] = 12 * (train.Year - train.CompetitionOpenSinceYear) + (train.Month - train.CompetitionOpenSinceMonth)
train["CompetitionMonths"] = train["CompetitionMonths"].replace(24187, 0)

eksiler = train.loc[train["CompetitionMonths"] > 24000]
train = train.loc[train["CompetitionMonths"] < 24000]

eksiler["CompetitionMonths"] = 0
train = pd.concat([train, eksiler], axis = 0)
train.head()
train = train.replace(np.nan, 0)

del train["Date"]
del train["CompetitionOpenSinceMonth"]
del train["CompetitionOpenSinceYear"]
del train["Open"]

train.loc[train['StoreType'] == 'a', 'StoreType'] = '1'
train.loc[train['StoreType'] == 'b', 'StoreType'] = '2'
train.loc[train['StoreType'] == 'c', 'StoreType'] = '3'
train.loc[train['StoreType'] == 'd', 'StoreType'] = '4'
train['StoreType'] = train['StoreType'].astype(float)

train.loc[train['Assortment'] == 'a', 'Assortment'] = '1'
train.loc[train['Assortment'] == 'b', 'Assortment'] = '2'
train.loc[train['Assortment'] == 'c', 'Assortment'] = '3'
train['Assortment'] = train['Assortment'].astype(float)

train.loc[train['StateHoliday'] == 'a', 'StateHoliday'] = '1'
train.loc[train['StateHoliday'] == 'b', 'StateHoliday'] = '2'
train.loc[train['StateHoliday'] == 'c', 'StateHoliday'] = '3'
train['StateHoliday'] = train['StateHoliday'].astype(float)

train["PromoInterval"].value_counts()
train.loc[train['PromoInterval'] == 'Jan,Apr,Jul,Oct', 'PromoInterval'] = '1'
train.loc[train['PromoInterval'] == 'Feb,May,Aug,Nov', 'PromoInterval'] = '2'
train.loc[train['PromoInterval'] == 'Mar,Jun,Sept,Dec', 'PromoInterval'] = '3'
train['PromoInterval'] = train['PromoInterval'].astype(float)

train.isnull().sum()
train.head()
df = train.copy()
del df["Sales"]
y = pd.DataFrame(train["Sales"])
x = df


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(x_train, y_train)
xgb_tahmin = xgb.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
print("R^2 Skoru :", round(r2_score(y_test , xgb_tahmin), 4))
print("Hata Kareler Ortalaması: ", round(np.sqrt(mean_squared_error(y_test , xgb_tahmin))))