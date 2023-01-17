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
#weatherHIstory datasini okuyoruz

data = pd.read_csv('/kaggle/input/szeged-weather/weatherHistory.csv')

data=pd.DataFrame(data)
data.head()
#nem ve sicaklik iliskisi inceleneceginden bu iki veriyi ayri bir degiskene atiyoruz

hum_degree=data[['Humidity', 'Temperature (C)']]
hum_degree.head()
#iki ozellik arasindaki bagintiyi gozlemlemek icin scatter grafige dokuyoruz

from matplotlib import pyplot as plt

import seaborn as sns

plt.figure(figsize=(13, 9))

plt.scatter(hum_degree["Humidity"], hum_degree["Temperature (C)"],s=65)

plt.xlabel('Humidity',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('Humidity-Temp',fontsize=25)

plt.show()
#datayi incelemeyi hizlandirmak ve daha anlamli halde gorebilmek icin 750 veriyi aliyoruz

new_hum_degree = hum_degree[:][:750]

len(new_hum_degree)
#datamiz icindeki Nan degerleri tespit edip temizliyoruz

new_hum_degree["Humidity"].isna().value_counts()

new_hum_degree["Temperature (C)"].isna().value_counts()

new_hum_degree = new_hum_degree.dropna(axis=0, how="any")
#datamiz icindeki tekrar eden degerleri siliyoruz

new_hum_degree = new_hum_degree.drop_duplicates(subset = ["Humidity", "Temperature (C)"])

len(new_hum_degree)
new_hum_degree.columns=["Hum","Tmp"]
#732 veriyi tekrar scatter ile gozlemliyoruz 

plt.figure(figsize=(12, 12))

plt.scatter(new_hum_degree["Hum"], new_hum_degree["Tmp"],s=65)

plt.xlabel('Hum',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('Hum-Temp',fontsize=25)

plt.show()
from sklearn.linear_model import LinearRegression 
# Tmp ve hum adli iki degiskende kolon degerlerini tutuyoruz

Hum = new_hum_degree.iloc[:, 0:1].values  

Tmp = new_hum_degree.iloc[:, -1].values  
#regresyon modelimizde girilen nem degerine gore sicaklik degeri aliyoruz

lin_reg=LinearRegression()

lin_reg.fit(Hum,Tmp)
#scatter grafiginde lineer degisimi gosteriyoruz

sns.set(font_scale=2)

plt.figure(figsize=(15, 15))

plt.scatter(Hum,Tmp,s=65)

plt.plot(Hum,lin_reg.predict(Hum), color='red', linewidth='6')

plt.xlabel('Hum',fontsize=25)

plt.ylabel('Tmp',fontsize=25)

plt.title('nem degerlerine gore temp tahmin gosterimi',fontsize=25)

plt.show()
#nem degerine gore tahmini hava sicakligi tahmini yaptiriyoruz

degree_lin = lin_reg.predict([[0.3]])

degree_lin
#r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

from sklearn.metrics import mean_squared_error,r2_score

Tmp_head_lin=lin_reg.predict(Hum)

print("Linear Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_lin))

degerlendirme={}

degerlendirme["Linear Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_lin)
m_lin_reg = LinearRegression()

m_lin_reg = m_lin_reg.fit(Hum,Tmp)

m_lin_reg.intercept_       # constant b0

m_lin_reg.coef_   
#nem degerine gore tahmini hava sicakligi tahmini yaptiriyoruz

degree_m_lin = m_lin_reg.predict([[0.3]])

degree_m_lin
#scatter grafiginde m-lineer degisimi gosteriyoruz

import operator

plt.scatter(Hum, Tmp, s=65)

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(Hum, Tmp), key=sort_axis)

X_test, y_pred = zip(*sorted_zip)

plt.plot(Hum, Tmp, color='g')

plt.show()
#r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Tmp_head_m_lin=m_lin_reg.predict(Hum)

print("Multiple Linear Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_m_lin))

degerlendirme["Multiple Linear Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_m_lin)
from sklearn.preprocessing import PolynomialFeatures 
pol = PolynomialFeatures(degree = 3) 

Hum_pol = pol.fit_transform(Hum) 

pol.fit(Hum_pol, Tmp) 

lin_reg2 = LinearRegression() 

lin_reg2.fit(Hum_pol, Tmp)
#tuz degerine gore  hava sicakligi tahmini yaptiriyoruz

Predict_Hum_pol = lin_reg2.predict(pol.fit_transform([[0.3]])) 

Predict_Hum_pol
sns.set(font_scale=1.6)

plt.figure(figsize=(13, 9))

x_grid = np.arange(min(Hum), max(Hum), 0.1)

x_grid = x_grid.reshape(-1,1)

plt.scatter(Hum,Tmp,s=65)

plt.plot(x_grid,lin_reg2.predict(pol.fit_transform(x_grid)) , color='red', linewidth = '6')

plt.xlabel('Hum',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('nem degerlerine gore temp tahmin gosterimi',fontsize=25)

plt.show()
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Tmp_head_pol=lin_reg2.predict(Hum_pol)

print("Polynomial Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_pol))

degerlendirme["Polynomial Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_pol)
from sklearn.tree import DecisionTreeRegressor



Hum_ = new_hum_degree.iloc[:,0].values.reshape(-1, 1)

Tmp_ = new_hum_degree.iloc[:,1].values.reshape(-1, 1)

dt_reg = DecisionTreeRegressor()      

dt_reg.fit(Hum_,Tmp_)
dt_reg.predict([[0.3]])
Tmp_head=dt_reg.predict(Hum_)
plt.scatter(Hum_,Tmp_, color="red")                         

plt.plot(Hum_,Tmp_head,color="green")

plt.xlabel("Nem")

plt.ylabel("Tmp")

plt.title("Decision Tree Model")

plt.show()
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Tmp_head_dt=dt_reg.predict(Hum_)

print("Decision Tree Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_dt))

degerlendirme["Decision Tree Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_dt)
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=100,random_state=42)

rf_reg.fit(Hum_,Tmp_)

rf_reg.predict([[0.3]])
Tmp_head=rf_reg.predict(Hum_)
plt.scatter(Hum_,Tmp_,color="red")

plt.plot(Hum_,Tmp_head,color="green")

plt.xlabel("Nem")

plt.ylabel("Tmp")

plt.title("Random Forest Model")

plt.show()
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Tmp_head_rf=rf_reg.predict(Hum_)

print("Random Forest Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_rf))

degerlendirme["Random Forest Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_rf)
degerlendirme