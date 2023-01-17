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
train = pd.read_csv("/kaggle/input/carinsurance/carInsurance_train.csv")

train.head()
train.columns
train.describe()
train.isnull().sum() / len(train)
import seaborn as sns



sns.pairplot(data = train, hue = "CarInsurance")
yeni_degiskenler = []



i = 0



while i < 4001:

    

    if train["LastContactMonth"][i] == "jan":

        yeni_degiskenler.append(1)

    elif train["LastContactMonth"][i] == "feb":

        yeni_degiskenler.append(1)

    elif train["LastContactMonth"][i] == "mar":

        yeni_degiskenler.append(1)

    elif train["LastContactMonth"][i] == "apr":

        yeni_degiskenler.append(2)

    elif train["LastContactMonth"][i] == "may":

        yeni_degiskenler.append(2)

    elif train["LastContactMonth"][i] == "jun":

        yeni_degiskenler.append(2)

    elif train["LastContactMonth"][i] == "jul":

        yeni_degiskenler.append(3)

    elif train["LastContactMonth"][i] == "aug":

        yeni_degiskenler.append(3)

    elif train["LastContactMonth"][i] == "sep":

        yeni_degiskenler.append(3)

    elif train["LastContactMonth"][i] == "oct":

        yeni_degiskenler.append(4)

    elif train["LastContactMonth"][i] == "nov":

        yeni_degiskenler.append(4)

    elif train["LastContactMonth"][i] == "dec":

        yeni_degiskenler.append(4)

        

    i += 1
len(yeni_degiskenler)
konusma = pd.to_datetime(train["CallEnd"]) - pd.to_datetime(train["CallStart"])



konusma = pd.to_numeric(konusma) / 1000000000

konusma = pd.DataFrame(konusma)



a = pd.concat([train.CarInsurance, konusma], names = ("sonuc", "saniye"), axis = 1)



sns.pairplot(data = a, hue = "CarInsurance")



konusma.loc[(konusma[0] >= 0) & (konusma[0] <= 126), "dakika"] = 1

konusma.loc[(konusma[0] > 126) & (konusma[0] <= 232), "dakika"] = 2

konusma.loc[(konusma[0] > 232) & (konusma[0] <= 460), "dakika"] = 3

konusma.loc[(konusma[0] > 460) & (konusma[0] <= 3253), "dakika"] = 4
dakika = pd.DataFrame(konusma["dakika"])

yeni_degiskenler = pd.DataFrame(yeni_degiskenler, columns = ["quarter"])



yeni_degiskenler = pd.concat([yeni_degiskenler, dakika], axis = 1)

yeni_degiskenler.head()
i = 0

yas = []



while i < 4000:

    

    if train["Age"][i] > 0 and train["Age"][i] < 26:

        yas.append(1)

    elif train["Age"][i] >=26 and train["Age"][i] < 35:

        yas.append(2)

    elif train["Age"][i] >= 35 and train["Age"][i] < 45:

        yas.append(3)

    elif train["Age"][i] >= 45:

        yas.append(4)

        

    i += 1



yas = pd.DataFrame(yas, columns = ["yas"])

yeni_degiskenler = pd.concat([yeni_degiskenler, yas], axis = 1)

yeni_degiskenler.head()
train = pd.concat([yeni_degiskenler, train], axis = 1)

train.head()
management = train.loc[train["Job"] == "management" ]

blue_collar = train.loc[train["Job"] == "blue-collar"]

technician = train.loc[train["Job"] == "technician"]

admin = train.loc[train["Job"] == "admin."]

services = train.loc[train["Job"] == "services"]

retired = train.loc[train["Job"] == "retired"]

self_employed = train.loc[train["Job"] == "self-employed"]

student = train.loc[train["Job"] == "student"]



management["Education"].value_counts() # tertiary 751 / secondary 94 / primary 22     

blue_collar["Education"].value_counts() # secondary 430 / primary 281 / tertiary 17

technician["Education"].value_counts() # secondary 446 / tertiary 177 / primary 16

admin["Education"].value_counts() # secondary 374 / tertiary 62 / primary 9            

services["Education"].value_counts() # secondary 269 / primary 27 / tertiary 25      

retired["Education"].value_counts() # secondary 100 / primary 93 / tertiary 37

self_employed["Education"].value_counts() # tertiary 84 / secondary 43 / primary 6

student["Education"].value_counts() # secondary 67 / tertiary 31 / primary 8



management["Age"].mean() #40

blue_collar["Age"].mean() #40

technician["Age"].mean() #39

admin["Age"].mean() # 39

services["Age"].mean() # 38

retired["Age"].mean() #63

self_employed["Age"].mean() # 41

student["Age"].mean() #25



train["Education"] = train["Education"].replace(np.nan, "?")





i = 0



while i < 4000:

    if train["Age"][i] >= 35 and train["Job"][i] == "management" and train["Education"][i] == "?":

        train["Education"][i] = train["Education"][i].replace("?", "tertiary")

    elif train["Age"][i] < 35 and train["Job"][i] == "management" and train["Education"][i] == "?":

        train["Education"][i] = train["Education"][i].replace("?", "secondary")

    i += 1



i = 0



while i < 4000:

    if train["Job"][i] == "admin." and train["Education"][i] == "?":

        train["Education"][i] = train["Education"][i].replace("?", "secondary")

    

    i += 1



i = 0



while i < 4000:

    if train["Job"][i] == "services" and train["Education"][i] == "?":

        train["Education"][i] = train["Education"][i].replace("?", "secondary")

    

    i += 1



i = 0



while i < 4000:

    if train["Job"][i] == "services" and train["Education"][i] == "?":

        train["Education"][i] = train["Education"][i].replace("?", "secondary")

    

    i += 1



i = 0



while i < 4000:

    if train["Age"][i] >= 28 and train["Job"][i] == "student" and train["Education"][i] == "?":

        train["Education"][i] = train["Education"][i].replace("?", "tertiary")

    elif train["Age"][i] < 28 and train["Job"][i] == "student" and train["Education"][i] == "?":

        train["Education"][i] = train["Education"][i].replace("?", "secondary")

    i += 1



train["Education"] = train["Education"].replace("?", "secondary")
train.isnull().sum()
train["Communication"] = train["Communication"].fillna("cellular")

train.isnull().sum()
train = train.drop("Outcome", axis = 1)

train = train.drop("CallEnd", axis = 1)

train = train.drop("CallStart", axis = 1)

train = train.drop("Id", axis = 1)



train = train.dropna()

train.isnull().sum()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



cols = ["Job", "Marital", "Education", "Communication", "LastContactMonth"]



for i in cols:

    train[i] = le.fit_transform(train[i])





x = train.iloc[:, 0:17]

y = train.iloc[:, 17:18]





from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

rf.fit(x_train, y_train)

rf_tahmin = rf.predict(x_test)



rf_skor = accuracy_score(y_test, rf_tahmin)

rf_skor
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(x_train, y_train)

xgb_tahmin = xgb.predict(x_test)

accuracy_score(y_test, xgb_tahmin)
from lightgbm import LGBMClassifier



lgb = LGBMClassifier()

lgb.fit(x_train, y_train)

lgb_tahmin = lgb.predict(x_test)

accuracy_score(y_test, lgb_tahmin)