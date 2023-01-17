# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.tree import DecisionTreeRegressor # DecisionTree (karar agacimizi) regressiyonu import ettik





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/hitters/Hitters.csv")

df = df.dropna() # EKSIK HUCRELERI SILDIK
df.info()
df.describe().T
x = df.iloc[:,5].values.reshape(-1,1)    # yurume degerlerinin 

y = df.iloc[:,1].values.reshape(-1,1)   # basket atma basarisini iliskilendirdik
import seaborn as sns
# plot the heatmap

corr = df.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)

# veri setimiz %80 train ve %20 tes olarak ikiye ayrildi,

# Modeli kurma

tree_reg = DecisionTreeRegressor()   #ornek modeli kurduk

tree_reg.fit(x_train,y_train)   #x ve y  degiskenlerimizi modele gonderdik. modelimiz artik hazir



print(tree_reg.predict([[40]])) # yurume degeri 40 olan basketcinin basket atma basarisi
y_test_1 =np.array(range(0,len(y_train)))



#Index ekseni
y_head = tree_reg.predict(x_train)
from sklearn.metrics import mean_squared_error, r2_score

# r2 degerimiziz : 

r2_degeri = r2_score(y_train, y_head)

print("dogrulanmamis r2 degerimiz = ",r2_degeri)



plt.plot(y_test_1,y_train, color="red")

plt.plot(y_test_1, y_head, color = "green")

plt.xlabel("yurume")

plt.ylabel("basket")

plt.show()
#test hatamiz

rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

# hata_payi = egitim(test) setimizdeki gercek y ile tahmin edilen bgmsiz 

#...degiskenlerinin hatakarelerinin ortalamalarinin karekokunu aldik.

print("test hatasi - ", rmse) 
