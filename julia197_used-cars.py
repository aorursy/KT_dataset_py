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
from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from sklearn import metrics, linear_model, model_selection
df = pd.read_csv("/kaggle/input/used-cars-database/autos.csv", encoding = "Latin1")

df.head()
df = df.drop('nrOfPictures',1)

df = df.drop('postalCode',1)

df = df.drop('dateCrawled',1)

df = df.drop('lastSeen',1)

df = df.drop('seller',1)

df = df.drop('offerType',1)

df = df.drop('name',1)

df = df.drop('abtest',1)

df = df[df.price < 500000]

df = df[df.price != 0]

df = df[(df.powerPS > 0) & (df.powerPS < 1000)]

df = df.dropna()
#Umwandeln der Jahre in Alter des Autos

df["age"] = df["dateCreated"].str[:4].astype(int) - df["yearOfRegistration"] + (df["dateCreated"].str[5:7].astype(int)-df["monthOfRegistration"])/12

df = df.drop(["dateCreated","yearOfRegistration","monthOfRegistration"], 1)



#Umwandeln in numerische Werte

df = pd.get_dummies(df, columns = ["gearbox", "vehicleType"])

df = df.drop("gearbox_manuell", 1)

df = df.drop("vehicleType_andere",1)

df = df.replace(value=0, to_replace="nein")

df = df.replace(value = 1, to_replace = "ja")



#Umwandeln von Modell und Marke in einen Wert

df["Markenmodell"] = df["brand"] + " " + df["model"]

df = df.drop(["brand", "model"],1)

df = pd.get_dummies(df, columns = ['Markenmodell'])

df = df.drop("Markenmodell_volvo xc_reihe",1)



#Nur Benzin und Dieselautos

df = df[(df.fuelType == 'benzin') | (df.fuelType == 'diesel')]

df = pd.get_dummies(df, columns = ["fuelType"])

df = df.drop("fuelType_benzin", 1)
df.head()
x_train, x_test, y_train, y_test = model_selection.train_test_split(df.drop("price",1), df[["price"]], random_state = 0)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

model.score(x_test,y_test)