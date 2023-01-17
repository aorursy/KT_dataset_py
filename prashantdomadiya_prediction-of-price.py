# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

data=pd.read_csv('/kaggle/input/agricultural-raw-material-prices-19902020/agricultural_raw_material.csv',index_col='Month')



########################################

#  Data  Cleaning 

#

data=data.dropna()

data=data.iloc[1:-2]

I,J=data.shape

for i in range(I):

    for j in range(J):

        data.iloc[i,j]=str((data.iloc[i,j])).replace('%','')

        data.iloc[i,j]=(data.iloc[i,j]).replace(',','')

data=data.astype('float')



Dl=list(range(1,len(data.columns),2))

clm=data.columns

data=data.drop(columns=clm[Dl])









X=data.loc[:'Mar-17']#data.loc[:'Mar-17',Product]

Y=data.loc['Jun-90':]#data.loc['Jun-90':,Product]



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# finding exact degree of polynomial 

from sklearn.model_selection import validation_curve

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression as Lr

import matplotlib.pyplot as plt



def PolynomialRegression(degree=2, **kwargs):

    return make_pipeline(PolynomialFeatures(degree),Lr(**kwargs))



degree = np.arange(0, 21)

##########################################################

# User Input 

Product='Cotton Price' # Assign procuct name to be predicted

##########################################################

Xvld=X[Product]

Yvld=Y[Product]



train_score, val_score = validation_curve(PolynomialRegression(), Xvld[:,None], Yvld[:,None],'polynomialfeatures__degree',degree, cv=7)

plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')

plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')

plt.legend(loc='best')

plt.ylim(0, 1)

plt.xlabel('degree')

plt.ylabel('score')

plt.show()
# Polynomial Regression 

# Here, I choose degree 2 based on validation curve (Above)

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression as Lr

from sklearn.pipeline import make_pipeline



Model=[]

Clm=X.columns

Ypr_test=np.zeros((len(X_test),len(Clm)))

Ypr_train=np.zeros((len(X_train),len(Clm)))

for i in range(len(X.columns)):

    Model.append(make_pipeline(PolynomialFeatures(2,include_bias=True),Lr()))

    Xin=X_train[Clm[i]]

    Yin=Y_train[Clm[i]]

    Model[i].fit(Xin[:,None],Yin)

    Xtstin=X_test[Clm[i]]

    Ypr_test[:,i]=Model[i].predict(Xtstin[:,None])

    Ypr_train[:,i]=Model[i].predict(Xin[:,None])

Ypr_test = pd.DataFrame(Ypr_test,columns=Clm,index=Y_test.index) 

Ypr_train = pd.DataFrame(Ypr_train,columns=Clm,index=Y_train.index) 

# Error

print("Mean train error", (Y_test-Ypr_test).abs().mean(axis=0))

print("Mean test error", (Y_test-Ypr_test).abs().mean(axis=0))



print("Max train error", (Y_train-Ypr_train).abs().mean(axis=0))

print("Max test error", (Y_test-Ypr_test).abs().mean(axis=0))



print("Min train error", (Y_train-Ypr_train).abs().mean(axis=0))

print("Min test error", (Y_test-Ypr_test).abs().mean(axis=0))
import matplotlib.pyplot as plt



Product='Copra Price'



ax = plt.axes()

ax.plot(Y_train[Product],'r',label='price')

ax.plot(Ypr_train[Product],'g',label='Predicted Price')

ax.legend(loc='upper left')

ax.set_title('Prediction (Training)')

ax.xaxis.set_major_locator(plt.MaxNLocator(3))

ax.set_xlabel("Month")

ax.set_ylabel(Product)

plt.show()
import matplotlib.pyplot as plt

Product='Copra Price'

ax = plt.axes()

ax.plot(Y_test[Product],'r',label='price')

ax.plot(Ypr_test[Product],'g',label='Predicted Price')

ax.legend(loc='upper left')

ax.set_title('Prediction (Testing)')

ax.xaxis.set_major_locator(plt.MaxNLocator(3))

ax.set_xlabel("Month")

ax.set_ylabel(Product)

plt.show()
#  Predict future valueses

###########################################################

# User Input

Xnew=X_test.iloc[50]

print(Y_test.iloc[50])

###########################################################



NumProd=len(X_test.columns)

Ypredict=np.zeros([NumProd])

for i in range(NumProd):

    Ypredict[i]=Model[i].predict(np.reshape(Xnew[i],(1,1)))

print(pd.Series(Ypredict,index=Xnew.index))
