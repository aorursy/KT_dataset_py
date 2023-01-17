# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/used-car-price-prediction/automobileEDA.csv'

df=pd.read_csv(path)

df.head()
# Liner regresyon modülünü yükleyelim

from sklearn.linear_model import LinearRegression
# Linear Regresyon Objesi yaratalım

lm=LinearRegression()

lm
X=df[["highway-mpg"]] # Bunu bir dataframe olarak ürettim.

Y=df["price"] # Tahmin edilecek değeri ise bir seri olarak ürettim.
# Modelimizi eğitelim

lm.fit(X,Y)
# Eğitilen modelimizin tahminlerini aşağıdaki gibi görebiliriz.

Yhat=lm.predict(X)

Yhat[0:5]
# Simple Linear Regresyon için temel değerle olan intercept ve slope değerlerinin ne olduğunu görmek için 

sabit=lm.intercept_ # intercept

egim=lm.coef_ # slope

print("intercept = %f slope= %f" % (sabit,egim))
Z = df[['horsepower','curb-weight','engine-size','highway-mpg']]

lm.fit(Z,df["price"])

# Sabit değer 

lm.intercept_
# Eğimler. Burada birden fazla eğim olacağı için bir array döner

lm.coef_
lm.predict(Z)[0:5]
import seaborn as sns

%matplotlib inline 



width = 12

height = 10

plt.figure(figsize=(width, height))

sns.regplot(x="highway-mpg", y="price", data=df)

plt.ylim(0,)
plt.figure(figsize=(width, height))

sns.regplot(x="peak-rpm", y="price", data=df)

plt.ylim(0,)
df.corr() # Tüm alanların birbiri ile korealasyonun getirir. Değerler 
df[['price','horsepower','curb-weight','engine-size','highway-mpg']].corr() # Belirli kolonları alalım
#En yüksek korelasyona sahip ilk beş parametre

df.corr()[["price"]].abs().sort_values(by="price",ascending=False)[1:].head(5) 
width = 12

height = 10

plt.figure(figsize=(width, height))

sns.residplot(df['highway-mpg'], df['price'])

plt.show()
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))





ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")

sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)





plt.title('Actual vs Fitted Values for Price')

plt.xlabel('Price (in dollars)')

plt.ylabel('Proportion of Cars')



plt.show()

plt.close()
# Aşağıdaki polinom fonksiyonunun grafiğini çizdirmek için kullanacağız.

def PlotPolly(model, independent_variable, dependent_variabble, Name):

    x_new = np.linspace(15, 55, 100)

    y_new = model(x_new)



    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')

    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')

    ax = plt.gca()

    ax.set_facecolor((0.898, 0.898, 0.898))

    fig = plt.gcf()

    plt.xlabel(Name)

    plt.ylabel('Price of Cars')



    plt.show()

    plt.close()
x = df['highway-mpg']

y = df['price']
f=np.polyfit(x,y,3)

print(f)
p=np.poly1d(f)

print(p)
PlotPolly(p,x,y,'highway-mpg')
f1=np.polyfit(x,y,11)

p1=np.poly1d(f1)

print(p1)

PlotPolly(p1,x,y,'highway-mpg')
from sklearn.preprocessing import PolynomialFeatures
#2 dereceden Polinom bir obje yaratalım

pr=PolynomialFeatures(degree=2)

pr

Z.shape # Orjinal daha 201 örnek 4 feature
Z_pr=pr.fit_transform(Z)
Z_pr.shape # Dönüştürülmüş data 201 örnek 15 feature
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
input=[('scale',StandardScaler()),('polynominal',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
#Yukarıdaki input listesini kullanarak pipeline constructor yaratırız.

pipe=Pipeline(input)

pipe
#Tahmin ve dataların normalize edilmesini eş zamanlı olarak yapabiliriz.

pipe.fit(Z,y)
ypipe=pipe.predict(Z)

ypipe[0:4]
lm.fit(X,Y)

#R^2 hesaplama

print("R-square =",lm.score(X,Y))
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df["price"],Yhat)

print("Mean Squared Error = ",mse)
lm.fit(Z,df["price"])

#R^2 hesaplama

print("R-square = ",lm.score(Z,df["price"]))
Y_predict_multi=lm.predict(Z)

print("Mean Squared Error = ", mean_squared_error(df["price"],Y_predict_multi))
from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))

print("R-square = ", r_squared)
mean_squared_error(df["price"],p(x))