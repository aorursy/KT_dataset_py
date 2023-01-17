# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #add yourself



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/fifa19/data.csv")
data.columns
data.head(10)
data[['Unnamed: 0','Overall','Wage']].head(10)
data.info()
dataBelgium=data[data.Nationality =="Belgium"]



dataBelgiumtop10=dataBelgium.head(10)[::-1]



Wage=[ 77,170,105,150,155,135,230,240,340,355]



a=0

for i in dataBelgiumtop10.Wage:

    dataBelgiumtop10.Wage=dataBelgiumtop10.Wage.replace(i,Wage[a])

    a=a+1



# plot data

plt.scatter(dataBelgiumtop10.Overall,dataBelgiumtop10.Wage)

plt.xlabel("Overall")

plt.ylabel("Wage")

plt.show()

# sklearn library

from sklearn.linear_model import LinearRegression



# linear regression model

linear_reg = LinearRegression()



x = dataBelgiumtop10.Overall.values.reshape(-1,1)

y = dataBelgiumtop10.Wage.values.reshape(-1,1)



linear_reg.fit(x,y)



# prediction



#starting point 

#Method 1

b0 = linear_reg.predict([[0]])

print("b0: ",b0)

#Method 2

b0_ = linear_reg.intercept_

print("b0_: ",b0_)   # The point where the y-axis intersects intercept



b1 = linear_reg.coef_

print("b1: ",b1)   # slope



# visualize line

array = dataBelgiumtop10.Overall.values.reshape(-1,1)  # Overall



plt.scatter(x,y)



y_head = linear_reg.predict(array)  # Wage



plt.plot(array, y_head,color = "red")



linear_reg.predict([[100]])
print(linear_reg.predict([[84]]),"-",linear_reg.predict([[91]]))
from sklearn.linear_model import LinearRegression



df1=dataBelgiumtop10[['Unnamed: 0','Overall','Wage']]



x = df1.iloc[:,[0,1]].values

y = df1.Wage.values.reshape(-1,1)



# %% fitting data

multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)

print("b1,b2: ",multiple_linear_regression.coef_)



# predict

multiple_linear_regression.predict(np.array([[84,150],[91,150]]))
import pandas as pd

import matplotlib.pyplot as plt



dataEngland=data[data.Nationality =="England"]



dataEnglandtop10=dataEngland.head(10)[::-1]



a=0

for i in dataEnglandtop10.Wage:

    dataEnglandtop10.Wage=dataEnglandtop10.Wage.replace(i,Wage[a])

    a=a+1

    

    

x = dataEnglandtop10.Overall.values.reshape(-1,1)

y = dataEnglandtop10.Wage.values.reshape(-1,1)



# plot data

plt.scatter(x,y)

plt.xlabel("Overall")

plt.ylabel("Wage")

plt.show()
# %% linear regression



from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x,y)





#%% predict



y_head = lr.predict(x)





plt.plot(x,y_head,color="red",label ="linear")

plt.show()
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n



from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)



x_polynomial = polynomial_regression.fit_transform(x)





# fit

linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head2 = linear_regression2.predict(x_polynomial)





plt.plot(x,y_head2,color= "green",label = "poly")

plt.title("polinomal grafik")

plt.legend()

plt.show()
# final shape formed

# point scatters

plt.scatter(x,y)

plt.xlabel("Overall")

plt.ylabel("Wage")



#Linear line

plt.plot(x,y_head,color="red",label ="linear")



#polynomial line

plt.plot(x,y_head2,color= "green",label = "poly")

plt.title("polynomial graph")

plt.legend()



plt.show()
df1
x = df1.iloc[:,0].values.reshape(-1,1)

y = df1.iloc[:,1].values.reshape(-1,1)



# decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()   # random sate = 0

tree_reg.fit(x,y)

'''

tree_reg.predict(5.5)

'''

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree_reg.predict(x_)



# visualize

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color = "green")

plt.xlabel("Unnamed")

plt.ylabel("Overall")

plt.show()
x = df1.iloc[:,0].values.reshape(-1,1)

y = df1.iloc[:,1].values.reshape(-1,1)



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x,y)



x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf.predict(x_)
# visualize

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color="green")

plt.xlabel("Unnamed")

plt.ylabel("Overall")

plt.show()


x = df1.iloc[:,0].values.reshape(-1,1)

y = df1.iloc[:,1].values.reshape(-1,1)

    

# plot data

plt.scatter(x,y)

plt.xlabel("Unnamed")

plt.ylabel("Overall")

plt.show()
#%% linear regression



# sklearn library

from sklearn.linear_model import LinearRegression



# linear regression model

linear_reg = LinearRegression()



linear_reg.fit(x,y)



y_head = linear_reg.predict(x)  # maas



#slope line

plt.plot(x, y_head,color = "red")





#average of points on the y-axis

m=sum(y)/10

x1=[min(x),max(x)]

y1=[m,m]  

plt.plot(x1,y1,color = "orange")  

plt.show()
# final shape formed

plt.scatter(x,y)

plt.xlabel("Unnamed")

plt.ylabel("Overall")



plt.plot(x, y_head,color = "red")

 

plt.plot(x1,y1,color = "orange")  

plt.show()