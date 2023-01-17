# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



from sklearn.model_selection import KFold

from sklearn.preprocessing import PolynomialFeatures



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/position-salaries/Position_Salaries.csv')
data
plt.scatter(data.Level,data.Salary);
X=data.iloc[:,1:2].values

y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

L=LinearRegression()
L.fit(X_train,y_train)
y_pred=L.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error



print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
from sklearn.metrics import r2_score,mean_squared_error



print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
def extended(ax, x, y, **args):

    xlim = ax.get_xlim()

    ylim = ax.get_ylim()



    x_ext = np.linspace(xlim[0], xlim[1], 100)

    p = np.polyfit(x, y , deg=1)

    y_ext = np.poly1d(p)(x_ext)

    ax.plot(x_ext, y_ext, **args)

    ax.set_xlim(xlim)

    ax.set_ylim(ylim)

    return ax



plt.figure(figsize=(10,10))

ax = plt.subplot()

ax.scatter(X,y, label="Data",color='g', marker='x', s=200)

ax = extended(ax,X_test.reshape(len(X_test)), y_pred,  color="r", lw=3, label="Predicted Line of Best Fit (extended)")

ax.plot(X_test,y_pred, lw=3, label="Predicted Line of Best Fit")

plt.xlabel("Position Level")

plt.ylabel("Salary earned")

plt.title("Plotting Line of Best Fit")

ax.legend()
class PolynomialRegression:

    def __init__(self,degree=5):

        self.degree=degree

        self.L=None

                

    def fit(self,X,y):



        X_poly = PolynomialFeatures(degree=self.degree).fit_transform(X)

        L = LinearRegression()

        

        L.fit(X_poly,y)

        

        self.L=L

    

    def predict(self,X):

        

        X_test_poly = PolynomialFeatures(degree=self.degree).fit_transform(X)

        return self.L.predict(X_test_poly)
# Function to plot R2 for a range of Degrees

def pRegression_R2(X,y,k_low=1,k_high=3):

    training_score=[]

    test_score=[]

    

    kf = KFold(n_splits=3)

    

    for k in range(k_low,k_high+1):

    



        train_s=[]

        test_s=[]

        

        pl=PolynomialRegression(degree=k)

        

        for train,test in kf.split(y):

            X_train, X_test, y_train, y_test =  X[train], X[test], y[train], y[test]

            

            pl.fit(X_train,y_train)

            

            train_s.append( r2_score(y_train, pl.predict(X_train)) )

            test_s.append( r2_score(y_test,pl.predict(X_test)) )

            

        training_score.append(np.mean(train_s))

        test_score.append(np.mean(test_s))



    plt.figure(figsize=(10,10))

    x=np.arange(k_high - k_low + 1)

    plt.plot(x,training_score,label="Training")

    plt.plot(x,test_score,label="Testing")

    plt.legend()

    plt.xlabel("Degree")

    plt.ylabel("r2-Score")

    # plt.xlim((1, 15))

    # plt.ylim((-3, 2))

    plt.title("R2-Score");

    plt.show()
pRegression_R2(X,y,k_high=10)
# Trainning model with degree=7

pl=PolynomialRegression(degree=7)

pl.fit(X_train,y_train)
# Plotting Line of Best Fit

plt.figure(figsize=(10,10))

plt.scatter(X,y, label="Data",color='g', marker='x', s=200)

plt.scatter(np.linspace(1,10,1000), pl.predict(np.linspace(1,10,1000).reshape(1000,1)), label="Line of Best Fit", marker='+', s=2)

plt.xlabel("Position Level")

plt.ylabel("Salary earned")

plt.title("Plotting Line of Best Fit")

ax.legend()

plt.show()