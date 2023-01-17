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
data3=pd.read_csv("/kaggle/input/greenhouse-gas-emissions-in-the-netherlands/IPCC_emissions.csv", sep=";")
keydata3=pd.read_csv("/kaggle/input/greenhouse-gas-emissions-in-the-netherlands/IPCC_key_description.csv", sep=";")
data3.tail()
keydata3.tail()
data3.info()
keydata3.info()
keydata3.drop(['Description', 'L1.1','L2.1', 'L3.1', 'L4.1'], axis=1,inplace=True)
keydata3.head()
keydata3.info()
keydata3.isnull().sum()
data3.isnull().sum()
data3 = data3.rename(columns={'Bronnen': 'Key'})
data3 = data3.merge(keydata3, on='Key')
data3.head()
data3.drop("Hierarchy key", axis=1,inplace=True)

data3 = data3.rename(columns={'CO2_1': 'CO2', 'CH4_2': 'CH4', 'N2O_3': 'N2O'})
data3.info()
data3
data3.L1.unique()
data3.L2.unique()
data3.L3.unique()
data3.L4.unique()
data3.isnull().sum()
data3 = data3.rename(columns={'Perioden': "Year"})
data3.Year.unique()
(data3[data3['L2'] == 'Stationaire bronnen, totaal']).count()
(data3[data3['L2'] == 'Mobiele bronnen; totaal']).count()
data_Stationary_sources=data3[data3['L2'] == 'Stationaire bronnen, totaal']
data_Mobile_resources=data3[data3['L2'] == 'Mobiele bronnen; totaal']
data_Stationary_sources
data_Stationary_sources.L3.unique()
data_Mobile_resources.L3.unique()
data_Stationary_sources_Industry_noenergysector=data_Stationary_sources[data_Stationary_sources['L3'] == 'Nijverheid (geen energiesector)']
data_Stationary_sources_AEGUOthersectorsstationary=data_Stationary_sources[data_Stationary_sources['L3'] == 'A,E,G-U Overige sectoren (stationair)']
data_Mobile_resources_Transport=data_Mobile_resources[data_Mobile_resources["L3"]=='Vervoer']
data_Mobile_resources_Other_mobile_resources=data_Mobile_resources[data_Mobile_resources["L3"]=='Overige mobiele bronnen, totaal']
data_Stationary_sources_Industry_noenergysector.L4.unique()
data_Stationary_sources_AEGUOthersectorsstationary.L4.unique()
data_Mobile_resources_Transport.L4.unique()
data_Mobile_resources_Other_mobile_resources.L4.unique()
data_Stationary_sources_Industry_noenergysector
examine1_Services_waste_waterr=data_Stationary_sources_AEGUOthersectorsstationary.copy()
examine1_Services_waste_waterr.drop(["Key","Title","L1","L2","L3"],axis=1, inplace=True)
examine1_Services_waste_waterr

examine1_Services_waste_waterr.drop(["ID","L4"],axis=1, inplace=True)
examine1_Services_waste_waterr
examine1_Services_waste_waterr.Year
examine1_Services_waste_watercpp=examine1_Services_waste_waterr.copy()
examine1_Services_waste_watercpp['Year'] = examine1_Services_waste_watercpp['Year'].str[:4]

examine1_Services_waste_watercpp
examine1_Services_waste_watercpp.info()
import matplotlib.pyplot as plt
plt.scatter(examine1_Services_waste_watercpp.CH4,examine1_Services_waste_watercpp.N2O)
plt.xlabel("CH4")
plt.ylabel("N2O")
#%% linear regression
# sklearn library
from sklearn.linear_model import LinearRegression
# linear regression model
linear_reg = LinearRegression()
x = examine1_Services_waste_watercpp.CH4.values.reshape(-1,1)
y = examine1_Services_waste_watercpp.N2O.values.reshape(-1,1)
linear_reg.fit(x,y)
#%% prediction
b0 = linear_reg.predict([[0]])
print("b0: ",b0)
b0_ =linear_reg.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept
b1 = linear_reg.coef_
print("b1: ",b1)   # egim slope
print(linear_reg.predict([[119]]))
# visualize line
array = np.array([100,200,300,400,500,600,700,800,900,1000]).reshape(-1,1)  # deneyim
plt.scatter(x,y)
y_head = linear_reg.predict(array) 
plt.plot(array, y_head,color = "red")
plt.show()
# sklearn kütüphanesini kullanarak verileri test ve eğitim olarak böleceğimiz fonksiyonu import ettik.
from sklearn.model_selection   import train_test_split

x = examine1_Services_waste_watercpp.CH4.values.reshape(-1,1)
y = examine1_Services_waste_watercpp.N2O.values.reshape(-1,1)
x_trainn, x_test, y_trainn, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
import statsmodels.api as sm
x=sm.add_constant(x)
model = sm.OLS(y,x).fit()
print_model = model.summary()
print(print_model)
lr = LinearRegression()

# Train veri kümelerini vererek makineyi eğitiyoruz.
lr.fit(x_trainn,y_trainn)
# CH4'ın test kümesini vererek N2O'ı tahmin etmesini sağlıyoruz. Üst satırda makinemizi eğitmiştik.
tahmin = lr.predict(x_test)
#Modelimizin Root Mean Squared Error (RMSE) degeri 

# RMSE
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(tahmin,y_test)))
#Multiple Linear Regression
examine1_Services_waste_watercpp_p=examine1_Services_waste_watercpp.drop(["N2O","Year"],axis=1) 
examine1_Services_waste_watercpp_p
k=sm.add_constant(examine1_Services_waste_watercpp_p)
model = sm.OLS(examine1_Services_waste_watercpp.N2O,k).fit()
print_model = model.summary()
print(print_model)
examine1_Services_waste_watercpp
examine1_Services_waste_watercpp['year'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
examine1_Services_waste_watercpp
examine1_Services_waste_watercpp_pp=examine1_Services_waste_watercpp.drop("Year",axis=1) 
examine1_Services_waste_watercpp_pp.info()
examine1_Services_waste_watercpp_multi=examine1_Services_waste_watercpp_pp.drop("N2O",axis=1) 
examine1_Services_waste_watercpp_multi
z=sm.add_constant(examine1_Services_waste_watercpp_multi)
model = sm.OLS(examine1_Services_waste_watercpp_pp.N2O,z).fit()
print_model = model.summary()
print(print_model)
examine1_Services_waste_watercpp_pp_drop=examine1_Services_waste_watercpp_pp.drop("CO2",axis=1)
x=examine1_Services_waste_watercpp_pp_drop.drop("N2O",axis=1)
x_trainn, x_test, y_trainn, y_test = train_test_split(x,examine1_Services_waste_watercpp_pp_drop.N2O,test_size=0.33,random_state=42)
lrmulti = LinearRegression()

# Train veri kümelerini vererek makineyi eğitiyoruz.
lrmulti.fit(x_trainn,y_trainn)
# CH4'ın ve N2O test kümesini vererek year'ı tahmin etmesini sağlıyoruz. Üst satırda makinemizi eğitmiştik.
tahminn = lrmulti.predict(x_test)
#Modelimizin Root Mean Squared Error (RMSE) degeri 

# RMSE
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(tahmin,y_test)))
lrmulti.predict([[119.69,20]])
#polinomal lineer
a = examine1_Services_waste_watercpp.CH4.values.reshape(-1,1)
b = examine1_Services_waste_watercpp.N2O.values.reshape(-1,1)
c=examine1_Services_waste_watercpp.CO2.values.reshape(-1,1)
plt.scatter(a,b)
plt.ylabel("CH4")
plt.xlabel("N2O")
plt.scatter(a,c)
plt.ylabel("CH4")
plt.xlabel("CO2")
plt.scatter(b,c)
plt.ylabel("N2O")
plt.xlabel("CO2")

lr = LinearRegression()

lr.fit(a,b)

#%% predict
b_headd = lr.predict(a)

plt.plot(b,b_headd,color="red",label ="linear")


#print("13454 of CH4: ",lr.predict(13454))

# %%
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 5)

a_polynomial = polynomial_regression.fit_transform(a)

# %% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(a_polynomial,b)

# %%
#print("13454 of CH4 tahmin(poly): ",linear_regression2.predict(a_polynomial,[13454]))
b_head2 = linear_regression2.predict(a_polynomial)
print(b_head2)
plt.plot(a,b_head2,color= "green",label = "poly")
plt.legend()
plt.show()