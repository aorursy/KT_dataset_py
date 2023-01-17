# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
headbrain=pd.read_csv('/kaggle/input/headbrain/headbrain.csv')

headbrain.head()
headbrain.columns #display all columns
X=headbrain['Head Size(cm^3)'].values #Assigning X

Y=headbrain['Brain Weight(grams)'].values #Assigning Y

X_mean=np.mean(X) # mean of X

Y_mean=np.mean(Y) # mean of Y

m=len(X) # setting size of samples
mp.scatter(X,Y,color='red')

mp.xlabel('Head Size(cm^3)')

mp.ylabel('Brain Weight(grams)')
#Formulating for Linear Regression

#Linear regression hypothesis = (h=thetha0 + thetha1 * x) or (y=c+mx)



#Calculating slope/b1/thetha1

numer=0

deno=0

for i  in range(m):

    numer+=(X[i]-X_mean)*(Y[i]-Y_mean)

    deno+=(X[i]-X_mean)**2

    

slope=(numer/deno) 



#calculating intercept/thetha0/b0

intercept=(Y_mean-(slope*X_mean))



#Display intercept and slope

print(intercept,slope)


#generating new values for x

x_max=max(X)+100

x_min=min(X)-100



#Calculatin values of new x (testing set) and y based on testing set of x

x=np.linspace(x_min,x_max,1000) #testing set of new x

y=intercept+(slope*x) #predicting new y



#plotting the regression line and scatter plot



mp.plot(x,y,color='black',label='Regression Line')

mp.scatter(X,Y,color='orange',label='Scatter Plot')

mp.xlabel('Head Size(cm^3)')

mp.ylabel('Brain Weight(grams)')

mp.legend(loc=2)

mp.show()
#Calculating the squared error or the fitment



ss_r=0

ss_t=0



for i in range(m):

    y_pred=intercept+(slope*X[i])

    ss_r+=(Y[i]-y_pred)**2

    ss_t+=(Y[i]-Y_mean)**2

    

r2=1-(ss_r/ss_t) #mean squared error = sum(testing Y- predicted y)^2 / sum(testing Y- mean of testing Y)^2

print(r2)#mean squared error



    