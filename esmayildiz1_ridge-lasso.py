

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn.datasets import load_boston

from sklearn import preprocessing

from sklearn.linear_model import Lasso, Ridge

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
boston=load_boston()

boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)

print(boston_df.info())
boston_df.isnull().sum()
boston_df.head()
plt.figure(figsize = (12,6))

sns.heatmap(boston_df.corr(),annot = True)
boston_df['Price']=boston.target

print( boston_df.head())
newX=boston_df.drop('Price',axis=1)

print(newX[0:3])  # check 

newY=boston_df['Price']
print( type(newY))# pandas core frame
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)

print (len(X_test), len(y_test))
lr = LinearRegression()

lr.fit(X_train, y_train)
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely

rr = Ridge(alpha=0.01) 



# restricted and in this case linear and ridge regression resembles

rr.fit(X_train, y_train)
rr100 = Ridge(alpha=100) #  comparison with alpha value

rr100.fit(X_train, y_train)
train_score=lr.score(X_train, y_train)

test_score=lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)

Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)

Ridge_test_score100 = rr100.score(X_test, y_test)
print ("linear regression train score:", train_score)

print ("linear regression test score:", test_score)

print ("ridge regression train score low alpha:", Ridge_train_score)

print ("ridge regression test score low alpha:", Ridge_test_score)

print ("ridge regression train score high alpha:", Ridge_train_score100)

print ("ridge regression test score high alpha:", Ridge_test_score100)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)

plt.ylabel('Coefficient Magnitude',fontsize=16)

plt.legend(fontsize=13,loc=4)

plt.show()
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=31)
lasso= Lasso()

lasso.fit(X_train,y_train)

train_score=lasso.score(X_train,y_train)

test_score =lasso.score(X_test,y_test)

coeff_used = np.sum(lasso.coef_!=0)
print("traning score:", train_score)

print("test score: ", test_score)

print("number of used feature: ", coeff_used)
lasso001=Lasso(alpha=0.01, max_iter=10e5)

lasso001.fit(X_train,y_train)

train_score001=lasso001.score(X_train,y_train)

test_score001=lasso001.score(X_test,y_test)

coeff_used001=np.sum(lasso001.coef_)
print("traning score001:", train_score001)

print("test score001: ", test_score001)

print("number of used feature001: ", coeff_used001)
lasso00001=Lasso(alpha=0.0001, max_iter=10e5)

lasso00001.fit(X_train,y_train)

train_score00001=lasso00001.score(X_train,y_train)

test_score00001=lasso00001.score(X_test,y_test)

coeff_used00001=np.sum(lasso00001.coef_)
print("traning score00001:", train_score00001)

print("test score00001: ", test_score00001)

print("number of used feature00001: ", coeff_used00001)
lr= LinearRegression()

lr.fit(X_train,y_train)

lr_train_score = lr.score(X_train,y_train)

lr_test_score = lr.score(X_test,y_test)


print("Lr training score: ", lr_train_score)

print("Lr test score: ", lr_test_score)
plt.subplot(1,2,1)

plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.xlabel('Coefficient Index',fontsize=16)

plt.ylabel('Coefficient Magnitude',fontsize=16)

plt.legend(fontsize=13,loc=4)
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)

plt.ylabel('Coefficient Magnitude',fontsize=16)

plt.legend(fontsize=13,loc=4)

plt.tight_layout()

plt.show()
