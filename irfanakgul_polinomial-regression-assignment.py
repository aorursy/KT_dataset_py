# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict





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
df.corr()
import seaborn as sns



# plot the heatmap

corr = df.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
x = df.iloc[:,3].values.reshape(-1,1)    # kosu degerlerinin 

y = df.iloc[:,1].values.reshape(-1,1)   # basket atma basarisini iliskilendirdik
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)

# veri setimiz %80 train ve %20 tes olarak ikiye ayrildi,

############# Polynomial reg ile  ####################



pol_reg = PolynomialFeatures(degree = 8)  # degree ile tahminimizin ayrinti derececesini belirliyoruz. cok buyuk olursa bozulur. veriye gore belirlemek lazim



level_poly = pol_reg.fit_transform(x_train) #polynoma gore x tanimladik



lm2 = LinearRegression()  #yeni x e gore tekrar linear reg olusturduk

lm2.fit(level_poly,y_train)

tahmin_poly = lm2.predict(pol_reg.fit_transform([[20]]))

tahmin_poly                          

 # kosu degeri 20 olan basketcinin basket atma basarisi
y_head = lm2.predict(pol_reg.fit_transform(x_train))

y_head[:10]
#X y cizgisi olustura bilmek icin, bir nevi indexleme yaptik

y_test_1 =np.array(range(0,len(y_train)))
# r2 skorumuz... 

from sklearn.metrics import r2_score

r2_degeri = r2_score(y_train, y_head)

print("Modelin tutarliligi =", r2_degeri)



plt.scatter(y_test_1,y_train, color="red")

plt.scatter(y_test_1, y_head, color = "blue")

plt.xlabel("yurume")

plt.ylabel("basket")

plt.show()

plt.plot(y_test_1,y_train, color="red")

plt.plot(y_test_1, y_head, color = "blue")

plt.xlabel("yurume")

plt.ylabel("basket")

plt.show()