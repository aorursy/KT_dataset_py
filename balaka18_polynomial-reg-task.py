# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import math as m
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as ml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
%matplotlib inline
ml.style.use('fivethirtyeight')

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/position-salaries/Position_Salaries.csv')
print("SHAPE = ",data.shape)
print("\nINFORMATION : \n")
print(data.info())
print("\nDESCRIPTION : \n\n",data.describe())
data.head()
print("No. of unique positions = ",data.Position.nunique())
print("\nUnique positions : \n\n",data.Position.unique())

# Label encode : As these are positions of varying importance, so can be considered as ranks (ordinal)
le = LabelEncoder()
vals = le.fit_transform(data.Position)
data['Position_enc'] = vals
data.drop(columns=['Position'],axis=1,inplace=True)
data
# Splitting into X and Y
X = np.array(data.iloc[:,0:3:2].values)
Y = np.array(data.Salary.values)
print("X shape = ",X.shape)
print("Y shape = ",Y.shape)
print("\nFeatures : \n",X)
print("\nLabels : \n",Y)
# Initializing fig and axes
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111,projection='3d')
# Scatter plot
ax.scatter(data.Level.values, data.Position_enc.values, data.Salary.values, zdir='z', s = 180, c = 'red', depthshade=True)
ax.set_xlabel('LEVEL')
ax.set_ylabel('POSITION')
ax.set_zlabel('SALARY')
ax.set_title('3D REPRESENTATION OF THE DATA')
plt.show()

plt.figure(figsize=(5,3))
plt.scatter(data.Level,data.Salary,c = 'blue')
plt.xlabel("LEVEL")
plt.ylabel('SALARY')
plt.title('SALARY VS LEVEL')
plt.show()
plt.figure(figsize=(5,3))
plt.scatter(data.Position_enc,data.Salary,c = 'green')
plt.xlabel("POSITION")
plt.ylabel('SALARY')
plt.title('SALARY VS POSITION')
plt.show()
def PolyReg(trainx=X,trainy=Y,degree=2):
    testx = X
    testy = Y
    pol = PolynomialFeatures(degree=degree,order='C')
    X_poly_train = pol.fit_transform(trainx)
    X_poly_test = pol.fit_transform(testx)
    lr = LinearRegression()
    lr.fit(X_poly_train,trainy)
    y_pred = lr.predict(X_poly_test)
    r2 = r2_score(y_pred,testy)
    mse = mean_squared_error(y_pred,testy)
    rmse = m.sqrt(mean_squared_error(y_pred,testy))
    
    return r2,mse,rmse
# Sample Run
r2,mse,rmse = PolyReg(X,Y,degree=5)
print("Mean squared error = {} , Root mean squared error = {} , R2 Score = {}".format(mse,rmse,r2))
# Plotting r2 scores to get optimum value of degree of polynomial.
iterate,rmse_it = [],[]
for i in range(1,11):
    r2_i,mse_i,rmse_i = PolyReg(X,Y,degree = i)
    iterate.append(r2_i)
    rmse_it.append(rmse_i)
    
xaxis = np.arange(10) + 1

plt.plot(xaxis,iterate,label="TEST")
plt.legend()
plt.xlabel("DEGREE OF THE POLYNOMIAL IN POLYNOMIAL REGRESSION")
plt.ylabel("r2-SCORE")
plt.title("R2-SCORE BASED COMPARISON");
plt.show()

plt.plot(xaxis,rmse_it,c = 'red', label="TEST")
plt.legend()
plt.xlabel("DEGREE OF THE POLYNOMIAL IN POLYNOMIAL REGRESSION")
plt.ylabel("RMSE")
plt.title("RMSE BASED COMPARISON");
plt.show()
# Test for optimum degree = 4
r21,mse1,rmse1 = PolyReg(X,Y,degree=2)
print("Root mean squared error = {} , R2 Score = {}\n".format(rmse1,r21))

# Define function for plotting graphs for different test degrees
def plotGraph(data,trainx = X,trainy = Y, degree = 2):
    testx = X
    testy = Y
    pol = PolynomialFeatures(degree=degree,order='C')
    X_poly_train = pol.fit_transform(trainx)
    X_poly_test = pol.fit_transform(testx)
    lr = LinearRegression()
    lr.fit(X_poly_train,trainy)
    y_pred = lr.predict(X_poly_test)
    
    # Plot
    plt.scatter(data.Level,Y,label="DATA",color='black')
    plt.plot(data.Level,y_pred,label="MODEL",color='red')
    plt.legend()
    plt.show()
    
    # Accuracy
    acc = r2_score(y_pred,testy)*100
    print("Accuracy of this model = {} %".format(acc))
    
# Sample test for degree = 5
plotGraph(data,X,Y,degree = 5)
# Sample test for degree = 2
plotGraph(data,X,Y,degree = 2)
