import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
df_gender=pd.read_csv('../input/weight-height.csv')
df_gender=df_gender.replace('Male','0')
df_gender=df_gender.replace('Female','1')
df_gender.head()
#2 variables X1=H and x2=W
#target variable gender and this is binary
y=df_gender['Gender']
df_gender.drop( ['Gender'],axis = 1,inplace = True)
X=df_gender
import matplotlib.pyplot as plt
plt.scatter(df_gender.Weight, df_gender.Height, color="green")
plt.xlabel('weight (lb)')
plt.ylabel('height (in)')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression

#creating a linearRegression object 'lm'
lm=LinearRegression()
print(lm)
lm.fit(X_train, y_train)
print(lm.intercept_)
print(lm.coef_)
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_train,lm.predict(X_train))
MSE
MSE=mean_squared_error(y_test,lm.predict(X_test))
MSE
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train,lm.predict(X_train)),r2_score(y_test,lm.predict(X_test))))
from sklearn.preprocessing import PolynomialFeatures
degrees=range(4)
MSE_train=np.empty(len(degrees))
MSE_test=np.empty(len(degrees))
for d in degrees:
    Xd_train=PolynomialFeatures(d).fit_transform(X_train)
    lmd=LinearRegression()
    lmd.fit(Xd_train,y_train)
    MSE_train[d]=mean_squared_error(y_train,lmd.predict(Xd_train))
    Xd_test=PolynomialFeatures(d).fit_transform(X_test)
    MSE_test[d]=mean_squared_error(y_test,lmd.predict(Xd_test))
    
mindeg=np.argmin(MSE_test)

plt.plot(degrees,MSE_train,marker='o',linewidth='2',label='train data error') 
plt.plot(degrees,MSE_test,marker='d',linewidth='2',label='test data error') 
plt.plot(mindeg,MSE_test[mindeg],marker='s',color='red',alpha=0.5,label='min test error') 
plt.legend(loc='upper left')
plt.xlabel('degree')
plt.ylabel('MSE')
plt.yscale("log")
plt.show()
X_itrain, X_test, y_itrain, y_test=train_test_split(X,y, test_size=0.2,random_state=2)
X_train, X_valid, y_train, y_valid=train_test_split(X_itrain,y_itrain, train_size=0.75,random_state=2)

degrees=range(30)

MSE_train = np.empty(len(degrees))
MSE_valid = np.empty(len(degrees))
MSE_test = np.empty(len(degrees))

for d in degrees:
    Xd_train=PolynomialFeatures(d).fit_transform(X_train)
    lmd=LinearRegression()
    lmd.fit(Xd_train, y_train)
    
    MSE_train[d]=mean_squared_error(y_train,lmd.predict(Xd_train))
    Xd_valid=PolynomialFeatures(d).fit_transform(X_valid)
    MSE_train[d]=mean_squared_error(y_valid,lmd.predict(Xd_valid))
        
    #Calculate the degree at which validation error is minimized
    
    mindeg=np.argmin(MSE_valid)
    
 #fit on the whole training set now
X_comb_train=np.concatenate((X_train, X_valid),axis=0)
Y_comb_train=np.concatenate((y_train, y_valid),axis=0)

clm=LinearRegression()
clm.fit(X_comb_train, Y_comb_train) #fit

#pedict on the test et now and calculate error
pred = clm.predict(X_test)
MSE_test=mean_squared_error(y_test,pred)

plt.plot(degrees,MSE_train, marker='o', linewidth='2', label='Train set error')
plt.plot(degrees,MSE_valid, marker='d', linewidth='2', label='Validation set error')
plt.plot(mindeg, MSE_test, marker='s', color='red', alpha=0.5, label='test error')

plt.legend(loc='upper left')
plt.ylabel('mean squared error')
plt.xlabel('degree')
plt.yscale("log")
plt.show()
print(MSE_test)
from sklearn.model_selection  import cross_val_score
X_intr, X_test, y_intr, y_test=train_test_split(X,y, test_size=0.2,random_state=3)

avg_scores = []
degrees= range(25)
for d in degrees:
    lnrreg=LinearRegression()
    Xd_train=PolynomialFeatures(d).fit_transform(X_train)
    scores=cross_val_score(lnrreg, Xd_train, y_train, cv=4, scoring='neg_mean_squared_error')
    avg_scores.append(-scores.mean())
    
    
from sklearn.model_selection  import cross_val_score
X_intr, X_test, y_intr, y_test=train_test_split(X,y, test_size=0.2,random_state=3)

avg_scores = []
degrees= range(25)
for d in degrees:
    lnrreg=LinearRegression()
    Xd_train=PolynomialFeatures(d).fit_transform(X_train)
    scores=cross_val_score(lnrreg, Xd_train, y_train, cv=4, scoring='neg_mean_squared_error')
    avg_scores.append(-scores.mean())

#Calculate the degree at which CV error is minimized
mindeg=np.argmin(avg_scores)

# fit the model on the complete train set
lnrreg.fit(PolynomialFeatures(mindeg).fit_transform(X_train), y_train)

#predict on the test set now and calculate error
Xd_test=PolynomialFeatures(mindeg).fit_transform(X_test)
pred=lnrreg.predict(Xd_test)
MSE_test=mean_squared_error(y_test, pred)

plt.plot(degrees, avg_scores, marker='d', color='red', linewidth='2', label='cross validation error')
plt.ylabel('Mean Squared Error')
plt.legend(loc='upper left')
plt.xlabel('Degree')
plt.yscale("log")
plt.show()
MSE_test