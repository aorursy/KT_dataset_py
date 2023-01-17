# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14,10]



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.
# Reading data

data = pd.read_csv('../input/headbrain/headbrain.csv')

print(data.shape)

data.head()
#Collecting X and Y

X = data['Head Size(cm^3)'].values

Y = data['Brain Weight(grams)'].values



#Mean_x and mean_y (media)

mean_x = np.mean(X)

mean_y = np.mean(Y)



#Total Number of values

m = len(X)



#Using the formula to calculate b1 and b2

numer=0

denom=0

for i in range(m):

    numer += (X[i]- mean_x) * (Y[i]- mean_y)

    denom += (X[i]- mean_x) ** 2

b1 = numer / denom

b0 = mean_y - (b1 * mean_x)



#Print coefficent

print(b1,b0)
#Plotting values and Regression Line

max_x = np.max(X) + 100

min_x = np.min(X) - 100



#Calculating line values x and y

x = np.linspace(min_x,max_x,1000)

y = b0+b1*x



#Plotting Line

plt.plot(x,y, color='#58b970',label='Regression Line')

#Plotting Scatter Points

plt.scatter(X,Y, c='#ef5423', label='Scatter Plot')



plt.xlabel('Head size in cm3')

plt.ylabel('Brain Weight in grams')

plt.legend()

plt.show()
ss_t = 0

ss_r = 0

for i in range(m):

    y_pred = b0 + b1 * X[i]

    ss_t += (Y[i] - mean_y) ** 2

    ss_r += (Y[i] - y_pred) ** 2

r2 = 1 - (ss_r/ss_t)

print(r2)

    
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



#Cannot use Rank 1 Matrix in sklear

X = X.reshape((m,1))

#Creating Model

reg = LinearRegression()

#Fitting training data

reg = reg.fit(X,Y)

#Y prediction

Y_pred = reg.predict(X)



#Calculating R2 score



r2_score = reg.score(X,Y)



print(r2_score)
#Ridge Regression

from sklearn.linear_model import Ridge



alphas = [0, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000, 10000, 100000]



coefs = []

plt.plot(X,Y,"o",label="Training Set")

for a in alphas:

    ridge = Ridge(alpha=a)

    ridge.fit(X, Y)

    y_prediction = ridge.predict(X)

    plt.plot(X,y_prediction,label="Alpha="+str(a))

    coefs.append(ridge.coef_)

plt.legend()

plt.show()

    
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures



alphas = [0, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000, 10000, 100000]



for a in alphas:

    for degree in range(1,6):

        plt.figure()

        plt.title("alpha="+str(a)+" degree="+str(degree),fontsize=20,fontweight="bold")

        plt.plot(X,Y,"o",label="Training Set")

        polynomialRidgeModel = make_pipeline(PolynomialFeatures(degree), Ridge(a))

        polynomialRidgeModel.fit(X,Y)

        res = polynomialRidgeModel.predict(X)

        plt.plot(X, res, label="Alpha="+str(a)+" Degree="+str(degree))

        plt.legend()

        plt.show()