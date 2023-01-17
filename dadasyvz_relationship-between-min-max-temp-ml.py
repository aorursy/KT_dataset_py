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
#Summary of Weather datasini okuyoruz

data = pd.read_csv('/kaggle/input/weatherww2/Summary of Weather.csv')

data=pd.DataFrame(data)
data.head()
#min ve max sicaklik iliskisi inceleneceginden bu iki veriyi ayri bir degiskene atiyoruz

min_max_degree=data[['MaxTemp', 'MinTemp']]
min_max_degree.head()
min_max_degree.columns=["Max","Min"]
#iki ozellik arasindaki bagintiyi gozlemlemek icin scatter grafige dokuyoruz

from matplotlib import pyplot as plt

import seaborn as sns

plt.figure(figsize=(13, 9))

plt.scatter(min_max_degree["Min"], min_max_degree["Max"],s=65)

plt.xlabel('Min',fontsize=25)

plt.ylabel('Max',fontsize=25)

plt.title('Min-Max-Temp',fontsize=25)

plt.show()
#datayi incelemeyi hizlandirmak ve daha anlamli halde gorebilmek icin 750 veriyi aliyoruz

new_min_max_degree = min_max_degree[:][:750]

len(new_min_max_degree)
#datamiz icindeki Nan degerleri tespit edip temizliyoruz

new_min_max_degree["Max"].isna().value_counts()

new_min_max_degree["Min"].isna().value_counts()

new_min_max_degree = new_min_max_degree.dropna(axis=0, how="any")

len(new_min_max_degree)
#750 veriyi tekrar scatter ile gozlemliyoruz 

plt.figure(figsize=(13, 9))

plt.scatter(new_min_max_degree["Min"], new_min_max_degree["Max"],s=65)

plt.xlabel('Min',fontsize=25)

plt.ylabel('Max',fontsize=25)

plt.title('Min-Max-Temp',fontsize=25)

plt.show()
from sklearn.linear_model import LinearRegression 
# max ve min adli iki degiskende kolon degerlerini tutuyoruz

Max = new_min_max_degree.iloc[:, -2].values  

Min = new_min_max_degree.iloc[:,1:2].values  
Max

Min
#regresyon modelimizde girilen min degerine gore sicaklik degeri aliyoruz

lin_reg=LinearRegression()

lin_reg.fit(Min,Max)
#scatter grafiginde lineer degisimi gosteriyoruz

sns.set(font_scale=2)

plt.figure(figsize=(15, 15))

plt.scatter(Min,Max,s=65)

plt.plot(Min,lin_reg.predict(Min), color='red', linewidth='6')

plt.xlabel('Min',fontsize=25)

plt.ylabel('Max',fontsize=25)

plt.title('min degerlerine gore temp tahmin gosterimi',fontsize=25)

plt.show()
#min degerine gore  tahmini max hava sicakligi tahmini yaptiriyoruz

degree_lin = lin_reg.predict([[20]])

degree_lin
#r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

from sklearn.metrics import mean_squared_error,r2_score

Max_head_lin=lin_reg.predict(Min)

print("Linear Regression R_Square Score: " ,r2_score(Max,Max_head_lin))

degerlendirme={}

degerlendirme["Linear Regression R_Square Score:"]=r2_score(Max,Max_head_lin)
m_lin_reg = LinearRegression()

m_lin_reg = m_lin_reg.fit(Min,Max)

m_lin_reg.intercept_       # constant b0

m_lin_reg.coef_         

#scatter grafiginde m-lineer degisimi gosteriyoruz

import operator

plt.scatter(Min, Max, s=65)

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(Min, Max), key=sort_axis)

X_test, y_pred = zip(*sorted_zip)

plt.plot(Min, Max, color='g')

plt.show()
from sklearn.preprocessing import PolynomialFeatures 
pol = PolynomialFeatures(degree = 3) 

Min_pol = pol.fit_transform(Min) 

pol.fit(Min_pol, Max) 

lin_reg2 = LinearRegression() 

lin_reg2.fit(Min_pol, Max)
#min degerine gore max hava sicakligi tahmini yaptiriyoruz

Predict_Max_pol = lin_reg2.predict(pol.fit_transform([[20]])) 

Predict_Max_pol
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Max_head_pol=lin_reg2.predict(Min_pol)

print("Polynomial Regression R_Square Score: " ,r2_score(Max,Max_head_pol))

degerlendirme["Polynomial Regression R_Square Score:"]=r2_score(Max,Max_head_pol)
sns.set(font_scale=1.6)

plt.figure(figsize=(13, 9))

x_grid = np.arange(min(Min), max(Min), 0.1)

x_grid = x_grid.reshape(-1,1)

plt.scatter(Min,Max,s=65)

plt.plot(x_grid,lin_reg2.predict(pol.fit_transform(x_grid)) , color='red', linewidth = '6')

plt.xlabel('Min',fontsize=25)

plt.ylabel('Max',fontsize=25)

plt.title('Min degerlerine gore max temp tahmin gosterimi',fontsize=25)

plt.show()
from sklearn.tree import DecisionTreeRegressor



Max_ = new_min_max_degree.iloc[:,0].values.reshape(-1, 1)

Min_ = new_min_max_degree.iloc[:,1].values.reshape(-1, 1)

dt_reg = DecisionTreeRegressor()      

dt_reg.fit(Min_,Max_)
dt_reg.predict([[20]])
Max_head=dt_reg.predict(Min_)
plt.scatter(Min_,Max_, color="red")                         

plt.plot(Min_,Max_head,color="green")

plt.xlabel("Min")

plt.ylabel("Max")

plt.title("Decision Tree Model")

plt.show()
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Max_head_dt=dt_reg.predict(Min_)

print("Decision Tree Regression R_Square Score: " ,r2_score(Max,Max_head_dt))

degerlendirme["Decision Tree Regression R_Square Score:"]=r2_score(Max,Max_head_dt)
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=100,random_state=42)

rf_reg.fit(Min_,Max_)

rf_reg.predict([[20]])

Tmp_head=rf_reg.predict(Min_)
plt.scatter(Min_,Max_,color="red")

plt.plot(Min_,Max_head,color="green")

plt.xlabel("Min")

plt.ylabel("Max")

plt.title("Random Forest Model")

plt.show()
##r_square ile tahminlerimizin dogruluk degerini tespit ediyoruz

Max_head_rf=rf_reg.predict(Min_)

print("Random Forest Regression R_Square Score: " ,r2_score(Max,Max_head_rf))

degerlendirme["Random Forest Regression R_Square Score:"]=r2_score(Max,Max_head_rf)
degerlendirme