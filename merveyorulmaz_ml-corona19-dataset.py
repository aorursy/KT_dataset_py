# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df.head()
data = df[ df["Country/Region"] =="Turkey"]
x = np.array(data.loc[:,"Confirmed"]).reshape(-1,1)
y = np.array(data.loc[:,"Deaths"]).reshape(-1,1)
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(x,y)
y_head = linReg.predict(x)
#Scatter

plt.plot(x,y_head,color="red")
plt.scatter(x,y)
plt.xlabel="confirmed"
plt.ylabel="death"
plt.show()
# herhangi bir confirmed sayısına göre predict edelim

y_ = linReg.predict([[6000]])
print("6k teşhis sonucu ölen kişi sayısı:",y_)
# yapılan predictin ne kadar dogru oldgunu değerlendirme RSQUARE

from sklearn.metrics import r2_score 
print("rscore: ",r2_score(y,y_head))
df2 = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")
df2.head(2)
data = df2 [df2["country"]=="China"]
data.head(2)
data_ =data["country"].value_counts(dropna=False)
data_
data1_ =df2["case_in_country"].fillna(-1)
data2_ =df2["age"].fillna(-1)
data3_ =df2["from Wuhan"].fillna(-1)
#x = np.array(df2.iloc[:,[8,14]]).reshape(-1,1) #Age and from whan attr. index num
#y = np.array(data1_.loc[:,"case_in_country"]).reshape(-1,1)
print("data1_.size",data1_.size)
print("data2_.size",data2_.size)
print("data3_.size",data3_.size)
x_ = np.array(data1_).reshape(-1, 1) #case_in_country
y_ = np.array(data2_).reshape(-1, 1) #age
from sklearn.linear_model import LinearRegression

mlReg = LinearRegression()

mlReg.fit(x_, y_) 

y_head = mlReg.predict(x_)

# y_head=mlReg.predict( dataMin, dataMax)
plt.scatter(x_,y_,color="pink")
plt.plot(x_, y_head, 'p--')
plt.show()
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df.head()
x = np.array(df["Deaths"]).reshape(-1,1)
y = np.array(df["Confirmed"]).reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures

polyReg = PolynomialFeatures(degree=2)

x_poly = polyReg.fit_transform(x)

linReg = LinearRegression()

linReg.fit(x_poly,y)

y_head = linReg.predict(x_poly)

plt.plot(x,y_head,color="orange")
plt.show()
x = np.array(df["Deaths"]).reshape(-1,1)
y = np.array(df["Confirmed"]).reshape(-1,1)

x_2 = x[10000:11000]
y_2 = y[10000:11000]


from sklearn.tree import DecisionTreeRegressor

treeReg = DecisionTreeRegressor()

treeReg.fit(x_2,y_2)

x2 = np.arange(min(x_2),max(x_2),0.01).reshape(-1,1)

y_head = treeReg.predict(x_2).reshape(-1,1)

plt.scatter(x_2,y_2,color="blue")
plt.plot(x_2,y_head,color="green")
plt.show()
x = np.array(df["Deaths"]).reshape(-1,1)
y = np.array(df["Confirmed"]).reshape(-1,1)

x_2 = x[10000:11000]
y_2 = y[10000:11000]
from sklearn.ensemble import RandomForestRegressor

randFReg = RandomForestRegressor(n_estimators=100, random_state=40)

randFReg.fit(x_2,y_2)

temp = randFReg.predict([[120]])

print(temp, "confirmed sayısından 120 kişi ölmüştür")

# görselleştirme

x2 = np.arange(min(x_2),max(x_2),0.01).reshape(-1,1)

y_head = treeReg.predict(x_2).reshape(-1,1)

plt.scatter(x_2,y_2,color="blue")
plt.plot(x_2,y_head,color="green")
plt.show()

#desicion tree regresyondan farkı 1 yerine 100 tane tree kullanılmasıdır
