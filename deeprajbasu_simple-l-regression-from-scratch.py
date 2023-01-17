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
#load data into a csv

df =  pd.read_csv('/kaggle/input/insurance.txt', sep='\t', header=0)

df.head(10)

#changing y so that it it float instead of str

df['Y'] = df['Y'].str.replace(',','.')

df["Y"] = pd.to_numeric(df["Y"])
#store the columns as separate lists

x = list(df["X"])

y = list(df["Y"])



pd.to_numeric(y)



type(y[0])
len(x),len(y),type(x),type(y)
a = 5

type(1.0*a)
#method for calculating mean

#this method takes a list X and provides the mean

def mean(X):

    return sum(X)/(1.00*len(X))





# B1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum( (x(i) - mean(x))^2 )

# variance  =  sum( (x(i) - mean(x))^2



#method for calculating variance 

#this method accepts a list X and its mean m and returns a variance value

def variance(X,m) : 

    return sum(list((i-m)**2 for i in X))
variance(x,mean(x))
# B1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum( (x(i) - mean(x))^2 )



# covariance  = sum((x(i) - mean(x)) * (y(i) - mean(y)))   



# method for calculating covariance 

def covariance(X,Y ,m_x,m_y):

	C = 0

	for i in range(len(x)):

		C = C+ (X[i] - m_x) * (Y[i] - m_y)

	return C
def coeffcients (X,Y):

    X_mean = mean(X)

    Y_mean = mean(Y)

    

    B1 = covariance(X,Y,X_mean,Y_mean) / variance(X,X_mean)

    B0 = Y_mean - (B1 * X_mean)

    

    return [B0,B1]
b0, b1 = coeffcients(x,y)

b0, b1



pred_for = [i for i in range(0,130,3)]

prediction =[]

for yhat in pred_for:

    prediction.append(b0 + b1 * yhat)

prediction
#visualising our system

import numpy as np 

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import matplotlib.lines as mlines



plt.figure(figsize=(26, 15))

ax = sns.scatterplot(x='X', y='Y', data=df,s=444,color="y")



kk=150

ii = [i for i in range(22+kk,3205+kk)]



SS = abs(np.sin(ii)*500)



plt.scatter(x=pred_for, y=prediction, color='r',s=SS)



XX = 79.2

YY = b0 + b1 * XX

YY,ax.scatter(x=XX, y=YY, color='#ffcc5f',s = 16000)

import numpy as np 



# B1 = covariance(X,Y,X_mean,Y_mean) / variance(X,X_mean)

# B0 = Y_mean - (B1 * X_mean)





b1,b0 =0,0

print(b1,b0)





b1 = np.cov(x, y)[0][1]/ np.var(x)

b0 = np.mean(y) - (b1 * np.mean(x))





XX = 79.2

YY = b0 + b1 * XX



YY