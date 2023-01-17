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
#bottle datasini okuyoruz

data = pd.read_csv('/kaggle/input/calcofi/bottle.csv')

data=pd.DataFrame(data)
data.head()
#tuzluluk ve sicaklik iliskisi inceleneceginden bu iki veriyi ayri bir degiskene atiyoruz

salt_degree=data[['Salnty', 'T_degC']]
salt_degree.head()
#iki ozellik arasindaki bagintiyi gozlemlemek icin scatter grafige dokuyoruz

from matplotlib import pyplot as plt

import seaborn as sns

plt.figure(figsize=(13, 9))

plt.scatter(salt_degree["Salnty"], salt_degree["T_degC"],s=65)

plt.xlabel('Slnty',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('slnty-Temp',fontsize=25)

plt.show()
#datayi incelemeyi hizlandirmak ve daha anlamli halde gorebilmek icin 750 veriyi aliyoruz

new_salt_degree = salt_degree[:][:750]

len(new_salt_degree)

#datamiz icindeki Nan degerleri tespit edip temizliyoruz

new_salt_degree["Salnty"].isna().value_counts()

new_salt_degree["T_degC"].isna().value_counts()

new_salt_degree = new_salt_degree.dropna(axis=0, how="any")

#datamiz icindeki tekrar eden degerleri siliyoruz

new_salt_degree = new_salt_degree.drop_duplicates(subset = ["Salnty", "T_degC"])

len(new_salt_degree)
#717 veriyi tekrar scatter ile gozlemliyoruz 

plt.figure(figsize=(12, 12))

plt.scatter(new_salt_degree["Salnty"], new_salt_degree["T_degC"],s=65)

plt.xlabel('Slnty',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('Slnty-Temp',fontsize=25)

plt.show()
from sklearn.linear_model import LinearRegression 
# Tmp ve Slt adli iki degiskende kolon degerlerini tutuyoruz

Slt = new_salt_degree.iloc[:, 0:1].values  

Tmp = new_salt_degree.iloc[:, -1].values  
#regresyon modelimizde girilen tuz degerine gore sicaklik degeri aliyoruz

lin_reg=LinearRegression()

lin_reg.fit(Slt,Tmp)
Slt
Tmp
#scatter grafiginde lineer degisimi gosteriyoruz

sns.set(font_scale=2)

plt.figure(figsize=(15, 15))

plt.scatter(Slt,Tmp,s=65)

plt.plot(Slt,lin_reg.predict(Slt), color='red', linewidth='6')

plt.xlabel('Slt',fontsize=25)

plt.ylabel('Tmp',fontsize=25)

plt.title('salt degerlerine gore temp tahmin gosterimi',fontsize=25)

plt.show()
#tuz degerine gore tahmini hava sicakligi tahmini yaptiriyoruz

degree_lin = lin_reg.predict([[33]])

degree_lin
#r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

from sklearn.metrics import mean_squared_error,r2_score

Tmp_head_lin=lin_reg.predict(Slt)

print("Linear Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_lin))

degerlendirme={}

degerlendirme["Linear Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_lin)
m_lin_reg = LinearRegression()

m_lin_reg = m_lin_reg.fit(Slt,Tmp)

m_lin_reg.intercept_       # constant b0

m_lin_reg.coef_         

#scatter grafiginde m-lineer degisimi gosteriyoruz

import operator

plt.scatter(Slt, Tmp, s=65)

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(Slt, Tmp), key=sort_axis)

X_test, y_pred = zip(*sorted_zip)

plt.plot(Slt, Tmp, color='g')

plt.show()
from sklearn.preprocessing import PolynomialFeatures 
pol = PolynomialFeatures(degree = 3) 

Slt_pol = pol.fit_transform(Slt) 

pol.fit(Slt_pol, Tmp) 

lin_reg2 = LinearRegression() 

lin_reg2.fit(Slt_pol, Tmp)
#tuz degerine gore  hava sicakligi tahmini yaptiriyoruz

Predict_Tmp_pol = lin_reg2.predict(pol.fit_transform([[33]])) 

Predict_Tmp_pol
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Tmp_head_pol=lin_reg2.predict(Slt_pol)

print("Polynomial Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_pol))

degerlendirme["Polynomial Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_pol)
sns.set(font_scale=1.6)

plt.figure(figsize=(13, 9))

x_grid = np.arange(min(Slt), max(Slt), 0.1)

x_grid = x_grid.reshape(-1,1)

plt.scatter(Slt,Tmp,s=65)

plt.plot(x_grid,lin_reg2.predict(pol.fit_transform(x_grid)) , color='red', linewidth = '6')

plt.xlabel('Slt',fontsize=25)

plt.ylabel('Temp',fontsize=25)

plt.title('salt degerlerine gore temp tahmin gosterimi',fontsize=25)

plt.show()
from sklearn.tree import DecisionTreeRegressor



Slt_ = new_salt_degree.iloc[:,0].values.reshape(-1, 1)

Tmp_ = new_salt_degree.iloc[:,1].values.reshape(-1, 1)

dt_reg = DecisionTreeRegressor()      

dt_reg.fit(Slt_,Tmp_)
dt_reg.predict([[33]])
Tmp_head=dt_reg.predict(Slt_)
plt.scatter(Slt_,Tmp_, color="red")                         

plt.plot(Slt_,Tmp_head,color="green")

plt.xlabel("Slnty")

plt.ylabel("Tmp")

plt.title("Decision Tree Model")

plt.show()
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Tmp_head_dt=dt_reg.predict(Slt_)

print("Decision Tree Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_dt))

degerlendirme["Decision Tree Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_dt)
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=100,random_state=42)

rf_reg.fit(Slt_,Tmp_)

rf_reg.predict([[33]])

Tmp_head=rf_reg.predict(Slt_)
plt.scatter(Slt_,Tmp_,color="red")

plt.plot(Slt_,Tmp_head,color="green")

plt.xlabel("Slnty")

plt.ylabel("Tmp")

plt.title("Random Forest Model")

plt.show()
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Tmp_head_rf=rf_reg.predict(Slt_)

print("Random Forest Regression R_Square Score: " ,r2_score(Tmp,Tmp_head_rf))

degerlendirme["Random Forest Regression R_Square Score:"]=r2_score(Tmp,Tmp_head_rf)
degerlendirme