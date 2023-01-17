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
import itertools

import numpy as np

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import preprocessing

import operator

import random

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

#warnings.filterwarnings(action='once')

%matplotlib inline
df = pd.read_csv('/kaggle/input/ManUtd_All_Season_Data.csv')   

df.head(100)
df.shape
df.isnull().sum()
df = df.drop(['Division'], axis = 1)

df.head()
df.dropna(subset = ["P"], inplace=True)

df.isnull().sum()
sns.relplot(x="Season", y="Pos", ci=None, kind="line",dashes = False, markers=True, data=df, height = 5, aspect = 3)
X = df[['Season', 'P', 'W', 'D', 'L', 'F', 'A', 'Pts']]

y = df[['Pos']]

# x1 and y1 will be used for fitting and training data that is not pre-processed

X1 = df[['Season', 'P', 'W', 'D', 'L', 'F', 'A', 'Pts']]

y1 = df[['Pos']]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

lin_df = LinearRegression()  

lin_df.fit(X_train, y_train)

lr_pred = lin_df.predict(X_test)   

lr_pred[0:5]
linrgr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

linrgr_r2 = r2_score(y_test, lr_pred)

print("RMSE Score for Test set: ",linrgr_rmse)

print("R2 Score: ",linrgr_r2)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn import linear_model

reg = linear_model.Ridge()

reg.fit(X_train, y_train)

rdg_pred = reg.predict(X_test)

print("The coefficients after ridge regression is :", reg.coef_)

print("The intercept after ridge regression is :", reg.intercept_)

rdg_pred[0:5]
rdgrgr_rmse = np.sqrt(mean_squared_error(y_test, rdg_pred))

rdgrgr_r2 = r2_score(y_test, rdg_pred)

print("RMSE Score for Test set: ",rdgrgr_rmse)

print("R2 Score: ",rdgrgr_r2)
# Finding the best value of alpha using cross validation

from sklearn.linear_model import LassoCV

regr = LassoCV()

regr.fit(X, y)

print(regr.alpha_)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn import linear_model

clf = linear_model.Lasso(alpha=regr.alpha_)

clf.fit(X_train, y_train)

las_pred = clf.predict(X_test)

print("The coefficients after lasso regression is :", clf.coef_)

print("The intercept after lasso regression is :", clf.intercept_)

las_pred[0:5]
lasrgr_rmse = np.sqrt(mean_squared_error(y_test, las_pred))

lasrgr_r2 = r2_score(y_test, las_pred)

print("RMSE Score for Test set: ",lasrgr_rmse)

print("R2 Score: ",lasrgr_r2)
# Finding the best value of alpha using cross validation

from sklearn.linear_model import ElasticNetCV

regr = ElasticNetCV()

regr.fit(X, y)

print(regr.alpha_)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

# Actual regression on test set

from sklearn.linear_model import ElasticNet

reg = ElasticNet(alpha=regr.alpha_)

reg.fit(X_train, y_train)

en_pred = reg.predict(X_test)

print("The coefficients after elasticnet regression is :", reg.coef_)

print("The intercept after elasticnet regression is :", reg.intercept_)

en_pred[0:5]
enrgr_rmse = np.sqrt(mean_squared_error(y_test, en_pred))

enrgr_r2 = r2_score(y_test, en_pred)

print("RMSE Score for Test set: ",enrgr_rmse)

print("R2 Score: ",enrgr_r2)
# Finding the best value of alpha using cross validation

from sklearn.linear_model import LassoLarsCV

regr = LassoLarsCV()

regr.fit(X, y)

print(regr.alpha_)
from sklearn import linear_model

reg = linear_model.LassoLars(alpha=regr.alpha_)

reg.fit(X_train, y_train)

lslr_pred = reg.predict(X_test)

print("The coefficients after elasticnet regression is :", reg.coef_)

print("The intercept after elasticnet regression is :", reg.intercept_)

lslr_pred[0:5]
lslrrgr_rmse = np.sqrt(mean_squared_error(y_test, lslr_pred))

lslrrgr_r2 = r2_score(y_test, lslr_pred)

print("RMSE Score for Test set: ",lslrrgr_rmse)

print("R2 Score: ",lslrrgr_r2)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn.preprocessing import PolynomialFeatures



from sklearn import linear_model



plrgr_rmse = np.zeros(9)

plrgr_r2 = np.zeros(9)



for i in range(1,10):

    poly_df = PolynomialFeatures(degree = i)

    transform_poly = poly_df.fit_transform(X_train)

    clf = linear_model.LinearRegression()

    clf.fit(transform_poly,y_train)

    polynomial_predict = clf.predict(poly_df.fit_transform(X_test))

    plrgr_rmse[i-1] = np.sqrt(mean_squared_error(y_test,polynomial_predict))

    plrgr_r2[i-1] = r2_score(y_test,polynomial_predict)

    print("\nThe predicted values with degree = ",i," is \n",polynomial_predict[0:5])

    print("\nRMSE Score of Test set for degree ", i," is: ",plrgr_rmse[i-1])

    print("R2 RMSE Score of Test set for degree ", i," is: ",plrgr_r2[i-1]) 



print("\nThe best RMSE score of Test Set is ", plrgr_rmse.min(), " with degree = ",plrgr_rmse.argmin()+1)

print("The max R2 score of Test Set is ", plrgr_r2.max(), " with degree = ",plrgr_r2.argmax()+1)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



k = ['mse', 'friedman_mse', 'mae']

dt_rmse = np.zeros(3)

dt_r2 = np.zeros(3)

n = 0



from sklearn.tree import DecisionTreeRegressor

for i in k:

    dt_reg = DecisionTreeRegressor(criterion = i)          # create  DecisionTreeReg with sklearn

    dt_reg.fit(X_train,y_train)

    dt_predict = dt_reg.predict(X_test)

    dt_rmse[n] = np.sqrt(mean_squared_error(y_test,dt_predict))

    dt_r2[n] = r2_score(y_test,dt_predict)

    print("\nThe predicted values for Test Set using criterion = ",i," is: ",dt_predict[0:5])

    print("\nThe RMSE score for Test Set using criterion = ",i," is: ",dt_rmse[n])

    print("The R2 score for Test Set using criterion = ",i," is: ",dt_r2[n])

    n += 1   

print("\nThe best RMSE score for Test Set is ", dt_rmse.min())

print("The max R2 score of Test Set is ", dt_r2.max())   
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



k = ['mse', 'mae']

n = 0

rf_rmse = np.zeros(2)

rf_r2 = np.zeros(2)



from sklearn.ensemble import RandomForestRegressor

for i in k:

    rf_reg = RandomForestRegressor(criterion = i)

    rf_reg.fit(X_train,y_train)

    rf_pred = rf_reg.predict(X_test)

    rf_rmse[n] = np.sqrt(mean_squared_error(y_test,rf_pred))

    rf_r2[n] = r2_score(y_test,dt_predict)

    print("\nThe predicted values for Test Set using criterion = ",i," is: ",rf_pred[0:5])

    print("\nThe RMSE score for Test Set using criterion = ",i," is: ",rf_rmse[n])

    print("The R2 score for Test Set using criterion = ",i," is: ",rf_r2[n])

    n += 1   

    

print("\nThe best RMSE score for Test Set is ", rf_rmse.min())

print("The max R2 score of Test Set is ", rf_r2.max())       
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



k = ['linear', 'square', 'exponential']

n = 0

adb_rmse = np.zeros(3)

adb_r2 = np.zeros(3)



from sklearn.ensemble import AdaBoostRegressor

for i in k:

    ada_regr = AdaBoostRegressor(loss = i)

    ada_regr.fit(X_train,y_train)

    ada_pred = ada_regr.predict(X_test)

    adb_rmse[n] = np.sqrt(mean_squared_error(y_test,ada_pred))

    adb_r2[n] = r2_score(y_test,ada_pred)

    print("\nThe predicted values for Test Set using loss = ",i," is: ",ada_pred[0:5])

    print("\nThe RMSE score for Test Set using loss = ",i," is: ",adb_rmse[n])

    print("The R2 score for Test Set using loss = ",i," is: ",adb_r2[n])

    n += 1   

print("\nThe best RMSE score for Test Set is ", adb_rmse.min())

print("The max R2 score of Test Set is ", adb_r2.max())           
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



k = ['ls', 'lad', 'huber', 'quantile']

n = 0

gdb_rmse = np.zeros(4)

gdb_r2 = np.zeros(4)



from sklearn.ensemble import GradientBoostingRegressor



for i in k:

    reg = GradientBoostingRegressor(loss = i)

    reg.fit(X_train, y_train)

    grdbst_pred = reg.predict(X_test)

    gdb_rmse[n] = np.sqrt(mean_squared_error(y_test,grdbst_pred))

    gdb_r2[n] = r2_score(y_test,grdbst_pred)

    print("\nThe predicted values for Test Set using loss = ",i," is: ",grdbst_pred[0:5])

    print("\nThe RMSE score for Test Set using loss = ",i," is: ",gdb_rmse[n])

    print("The R2 score for Test Set using loss = ",i," is: ",gdb_r2[n])

    n += 1   

print("\nThe best RMSE score for Test Set is ", gdb_rmse.min())

print("The max R2 score of Test Set is ", gdb_r2.max())    
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn.neighbors import KNeighborsRegressor



n = 0

knn_rmse = np.zeros(9)

knn_r2 = np.zeros(9)



for i in range(1,10):

    neigh = KNeighborsRegressor(n_neighbors=i)

    neigh.fit(X_train, y_train)

    knn_pred = neigh.predict(X_test)

    knn_rmse[n] = np.sqrt(mean_squared_error(y_test,knn_pred))

    knn_r2[n] = r2_score(y_test,knn_pred)

    print("\nThe predicted values for Test Set with neighbor k = ",i," is: \n",knn_pred[0:5])

    print("\nThe RMSE score for Test Set with neighbor k = ",i," is: ",knn_rmse[n])

    print("The R2 score for Test Set with neighbor k = ",i," is: ",knn_r2[n])

    n += 1   

print("\nThe best RMSE score for Test Set is ", knn_rmse.min(), " with neighbor k = ", knn_rmse.argmin()+1)

print("The max R2 score of Test Set is ", knn_r2.max(), " with neighbor k = ", knn_r2.argmax()+1)   
X= preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn.linear_model import SGDRegressor

reg = SGDRegressor()

reg.fit(X_train, y_train)

sgd_pred = reg.predict(X_test)

sgd_pred[0:5]
sgd_rmse = np.sqrt(mean_squared_error(y_test,sgd_pred))

sgd_r2 = r2_score(y_test,sgd_pred)

print("RMSE Score for Test set: ",sgd_rmse)

print("R2 Score for Test set: ",sgd_r2)
#X= preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn.svm import SVR

k = ['linear', 'poly', 'rbf', 'sigmoid']

n = 0

svr_rmse = np.zeros(4)

svr_r2 = np.zeros(4)



for i in k:

    reg = SVR(kernel = i)

    reg.fit(X_train, y_train)

    svr_pred = reg.predict(X_test)

    svr_rmse[n] = np.sqrt(mean_squared_error(y_test,svr_pred))

    svr_r2[n] = r2_score(y_test,svr_pred)

    print("\nThe predicted values for Test Set with kernel = ",i," is: ",svr_pred[0:5])

    print("\nThe RMSE score for Test Set with kernel = ",i," is: ",svr_rmse[n])

    print("The R2 score for Test Set with kernel = ",i," is: ",svr_r2[n])

    n += 1   

print("\nThe best RMSE score for Test Set is ", svr_rmse.min())

print("The max R2 score of Test Set is ", svr_r2.max())         
#X= preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn.svm import NuSVR

k = ['linear', 'poly', 'rbf', 'sigmoid']

n = 0

nusvr_rmse = np.zeros(4)

nusvr_r2 = np.zeros(4)



for i in k:

    reg = NuSVR(kernel = i)

    reg.fit(X_train, y_train)

    nusvr_pred = reg.predict(X_test)

    nusvr_rmse[n] = np.sqrt(mean_squared_error(y_test,nusvr_pred))

    nusvr_r2[n] = r2_score(y_test,nusvr_pred)

    print("\nThe predicted values for Test Set with kernel = ",i," is: ",nusvr_pred[0:5])

    print("\nThe RMSE score for Test Set with kernel = ",i," is: ",nusvr_rmse[n])

    print("The R2 score for Test Set with kernel = ",i," is: ",nusvr_r2[n])

    n += 1   

print("\nThe best RMSE score for Test Set is ", nusvr_rmse.min())

print("The max R2 score of Test Set is ", nusvr_r2.max())       
#X= preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)



from sklearn.svm import LinearSVR



reg = LinearSVR()

reg.fit(X_train, y_train)

linsvr_pred = reg.predict(X_test)

linsvr_rmse = np.sqrt(mean_squared_error(y_test,linsvr_pred))

linsvr_r2 = r2_score(y_test,linsvr_pred)

print("\nThe predicted values for Test Set is: ",linsvr_pred[0:5])

print("\nThe RMSE score for Test Set is: ",linsvr_rmse)

print("The R2 score for Test Set is: ",linsvr_r2)  
# rmse = root mean squared score.......r2 = R2-Score

# 1.Linear Models

# 1.1. Linear Regression

linrgr_rmse

linrgr_r2



# 1.2. Ridge Regression

rdgrgr_rmse

rdgrgr_r2



# 1.3. Lasso

lasrgr_rmse

lasrgr_r2



# 1.4. ElasticNet

enrgr_rmse

enrgr_r2



# 1.5. LarsLasso

lslrrgr_rmse

lslrrgr_r2



# 2.Polynomial Regression

plrgr_rmse = plrgr_rmse.min()

plrgr_r2 = plrgr_r2.max()



# 3.Decision Tree

dt_rmse = dt_rmse.min()

dt_r2 = dt_r2.max()



# 4.Ensemble Methods

# 4.1. Random Forest

rf_rmse = rf_rmse.min()

rf_r2 = rf_r2.max()



# 4.2. AdaBoost

adb_rmse = adb_rmse.min()

adb_r2 = adb_r2.max()



# 4.3. GradientBoost

gdb_rmse = gdb_rmse.min()

gdb_r2 = gdb_r2.max()



# 5.KNearestNeighbor

knn_rmse = knn_rmse.min()

knn_r2 = knn_r2.max()



# 6.Stochastic Gradient Descent

sgd_rmse

sgd_r2



# 7.Support Vector Machines

# 7.1. SVR

svr_rmse = svr_rmse.min()

svr_r2 = svr_r2.max()



# 7.2. NuSVR

nusvr_rmse = nusvr_rmse.min()

nusvr_r2 = nusvr_r2.max()

# 7.3. LinearSVR

linsvr_rmse = linsvr_rmse

linsvr_r2 = linsvr_r2



#max of all

min_rmse = [linrgr_rmse,rdgrgr_rmse,lasrgr_rmse,enrgr_rmse,lslrrgr_rmse,plrgr_rmse,dt_rmse,rf_rmse,adb_rmse,gdb_rmse,knn_rmse,sgd_rmse,svr_rmse,nusvr_rmse,linsvr_rmse]

max_r2 = [linrgr_r2,rdgrgr_r2,lasrgr_r2,enrgr_r2,lslrrgr_r2,plrgr_r2,dt_r2,rf_r2,adb_r2,gdb_r2,knn_r2,sgd_r2,svr_r2,nusvr_r2,linsvr_r2]
data = {'Algorithm':['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 'LarsLasso Regression', 'Polynomial Regression','Decision Tree Regression','Random Forest Regression','AdaBoost Regression','Gradient Boosting Regression','KNearest Neighbor Regression','Stochastic Gradient Regression','Support Vector Regression','Nu Support Vector Regression','Linear Support Vector Regression'], 

        'R2-Sore':max_r2, 'Root Mean Squared Error':min_rmse}

s = pd.DataFrame(data, index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

s
print("The best R-squared score is shown by Gradient Boosting Regression ")

print("\nR2 Score = ",gdb_r2)
fd = pd.read_csv('/kaggle/input/ManUtd_All_Season_Data.csv')  

fd.head(100)
fd = fd.drop(['Division'], axis = 1)

fd.dropna(subset = ["P"], inplace=True)
fd.isnull().sum()
# For the dataset pertaining to next 30 years I have used the standard deviation and mean of W to get a random integer

print("mean of W = ",round(fd.W.mean()))

print("std of W = ",round(fd.W.std()))

min1 = round(fd.W.mean())-round(fd.W.std())

max1 = round(fd.W.mean())+round(fd.W.std())

print("The random value which we require is between ",min1," and ",max1)
w = np.zeros(30)

for i in range(0,30):

    w[i] = random.randint(13,25)
print("mean of D = ",round(fd.D.mean()))

print("std of D = ",round(fd.D.std()))

min2 = round(fd.D.mean())-round(fd.D.std())

max2 = round(fd.D.mean())+round(fd.D.std())

print("The random value which we require is between ",min2," and ",max2)
d = np.zeros(30)

for i in range(0,30):

    d[i] = random.randint(6,12)
# No. of matches lost = Total matches played - (Matches won + matches drawn)

l = np.zeros(30)

for i in range(0,30):

    l[i] = (38 - (w[i]+d[i]))
print("mean of F = ",round(fd.F.mean()))

print("std of F = ",round(fd.F.std()))

min4 = round(fd.F.mean())-round(fd.F.std())

max4 = round(fd.F.mean())+round(fd.F.std())

print("The random value which we require is between ",min4," and ",max4)
f = np.zeros(30)

for i in range(0,30):

    f[i] = random.randint(52,84)
print("mean of A = ",round(df.A.mean()))

print("std of A = ",round(df.A.std()))

min5 = round(fd.A.mean())-round(fd.A.std())

max5 = round(fd.A.mean())+round(fd.A.std())

print("The random value which we require is between ",min5," and ",max5)
a = np.zeros(30)

for i in range(0,30):

    a[i] = random.randint(31,67)
# Total points earned = 3*Matches Won + Matches drawn 

p = np.zeros(30)

for i in range(0,30):

    p[i] = (3*w[i]+d[i])
n = 0

test_set = np.zeros((30,8))

#Total number of matches played in a season

s = 38

for i in range(2019,2049):

    test_set[n] = [i, s, w[n], d[n], l[n], f[n], a[n], p[n]]

    n += 1

test_set[0:5]    
from sklearn.ensemble import GradientBoostingRegressor



reg = GradientBoostingRegressor(loss = 'lad')

#fitting the data that is not pre processed

reg.fit(X1, y1)

grdbst_pred = reg.predict(test_set)

print("\nThe predicted values for Test Set is: ",grdbst_pred[0:30])
plpred = np.round(grdbst_pred)

print("\nThe predicted rounded values for Test Set is: ",plpred[0:30])