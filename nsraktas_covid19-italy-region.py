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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv("/kaggle/input/covid19-in-italy/covid19_italy_region.csv")
data.head()
data.columns
data.info()
data.describe()
data.corr()
df = data.groupby('Date')[['HospitalizedPatients', 'IntensiveCarePatients',
       'TotalHospitalizedPatients', 'HomeConfinement', 'CurrentPositiveCases',
       'NewPositiveCases', 'Recovered', 'Deaths', 'TotalPositiveCases',
       'TestsPerformed']].sum()
df["day"]=range(1,len(df.Deaths)+1)
df.head()
plt.figure(figsize=(10,10))
sns.barplot(x=df['day'], y=df['Deaths'])
plt.xticks(rotation= 90)
plt.xlabel('Days')
plt.ylabel('Deaths')

data1 = df[['day','CurrentPositiveCases']]
sns.pairplot(data1,kind="reg");
#Linear

# Pandas kütüphanesinden iloc fonksiyonu yardımıyla "day" kolonunun değerlerini bir değişkene atadık

day= df.iloc[:,-1].values.reshape(-1,1)

# CurrentPositiveCases kolonunun değerlerini bir değişkene atadık
CurrentPositiveCases= df.iloc[:,4:5].values.reshape(-1,1)

# sklearn library
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(day,CurrentPositiveCases,test_size=0.33,random_state=0)

# sklearn kütüphanesini kullanarak LinearRegression sınıfını dahil ediyoruz.
from sklearn.linear_model import LinearRegression

#Sınıftan bir nesne oluşturuyoruz.
lr = LinearRegression()

# Train veri kümelerini vererek makineyi eğitiyoruz.
lr.fit(x_train,y_train)

# gunlerin'ın test kümesini vererek CurrentPositiveCases'leri tahmin etmesini sağlıyoruz
tahmin = lr.predict(x_test)

x_train=np.sort(x_train)
y_train=np.sort(y_train)



b0 = lr.predict([[0]])
print("b0: ",b0)     # y eksenini kestigi nokta predict ile

b0_ =lr.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept ile

b1 = lr.coef_
print("b1: ",b1)     # egim slope

# Grafik şeklinde ekrana basıyoruz.
plt.scatter(x_train,y_train)
plt.plot(x_test,tahmin,color="red")
plt.show()

#tahmin ornek...100 gun sonra 
print('100 gun sonra :',lr.predict([[100]]))


# day kolonunun değerlerini bir değişkene atadık
day= df.iloc[:,-1].values.reshape(-1,1)

# Deaths kolonunun değerlerini bir değişkene atadık
Deaths= df.iloc[:,7:8].values.reshape(-1,1)

# sklearn kütüphanesini kullanarak LinearRegression sınıfını dahil ediyoruz.
from sklearn.linear_model import LinearRegression

# LinearRegression sınıftan bir nesne oluşturuyoruz.
lr = LinearRegression()

#PolynomialFeatures sınıfini import ettik
from sklearn.preprocessing import PolynomialFeatures

# PolynomialFeatures sınıfından bir nesne ürettik.
poly = PolynomialFeatures(degree=5)

# Makineyi eğitmeden önce day kolonundaki değerleri PolynomialFeatures ile dönüşüm yapıyoruz.
day_poly = poly.fit_transform(day)

# Makineyi eğitiyoruz.
lr.fit(day_poly, Deaths)

# Makineyi eğittikten sonra bir tahmin yaptırtıyoruz.
predict = lr.predict(day_poly)


plt.scatter(day, Deaths, color='red')
plt.plot(day, predict, color='blue')
plt.xlabel('days')
plt.ylabel('Deaths')
plt.show()
#df[['day','Deaths']]
# Polinom

CurrentPositiveCases= df.iloc[:,4:5].values.reshape(-1,1)
recorved= df.iloc[:,6:7].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
lr = LinearRegression()

poly = PolynomialFeatures(degree=5)

CurrentPositiveCases_poly = poly.fit_transform(CurrentPositiveCases)

lr.fit(CurrentPositiveCases_poly, recorved)

predict = lr.predict(CurrentPositiveCases_poly)

plt.scatter(CurrentPositiveCases, recorved, color='red')
plt.plot(CurrentPositiveCases, predict, color='blue')
plt.show()

# Kurtulan sayisinda evde,hastanede ve yogun bakimdakilerin sayisini kullandik
x = df.iloc[:,[0,1,3]].values
y = df.Recovered.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2,b3: ",multiple_linear_regression.coef_)




