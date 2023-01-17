import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import  metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

# from lcp import plot_learning_curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import KFold
df = pd.read_excel("/kaggle/input/slr06.xls")
df.head()
df.info()
df.describe()
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))



ax1.set_title('Distribution of feature X i.e. Number of Claims')

sns.distplot(df.X,bins=50,ax=ax1)



ax2.set_title('Distribution of label Y i.e. Total Payment for Corresponding claims')

sns.distplot(df.Y,bins=50,ax=ax2)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))



ax1.set_ylim(-50,150)

ax1.set_title('Boxplot for X')

sns.boxplot(y='X',data=df,ax=ax1,)

sns.stripplot(y='X',color='green',data=df,jitter=True,ax=ax1,alpha=0.5)



ax2.set_ylim(-50,150)

ax2.set_title('Violinplot for X')

sns.violinplot(y='X',data=df,ax=ax2)

sns.stripplot(y='X',color='green',data=df,jitter=True,ax=ax2,alpha=0.5)
fig , (ax1) = plt.subplots(1,1,figsize=(10,4))

ax1.set_title('Scatter plot between feature and Label')

sns.regplot(data=df,x='X',y='Y',ax=ax1)
Y = df.Y

X = pd.DataFrame(df.X)

regr = linear_model.LinearRegression()

regr.fit(X,Y)

Y_pred = regr.predict(X)

mse = metrics.mean_squared_error(Y,Y_pred)

print('RMSE for Training set : %f' % (np.sqrt(mse)))
def meanSquarredError(y_test,y_pred):

    error = 0

    for i,j in zip(y_test,y_pred):

        error += (i-j)**2

    error /= len(y_test)

    return error



def rootMeanSquarredError(y_test,y_pred):

    mean_error = meanSquarredError(y_test,y_pred)

    return np.sqrt(mean_error)    
mse = meanSquarredError(Y,Y_pred)

print('MSE for Training set : %f' % (meanSquarredError(Y,Y_pred)))

print('RMSE for Training set : %f' % (rootMeanSquarredError(Y,Y_pred)))
regr_cv = linear_model.LinearRegression()

scores = cross_val_score(regr_cv,X,Y,cv=10,scoring='neg_mean_squared_error')

scores = scores*-1

print('Mean RMSE for Cross Validation : %f' % (np.mean(np.sqrt(scores))))
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)



fig, ax = plt.subplots()

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_title('Scatter plot showing train and test sample split')

ax.scatter(X_train,Y_train,marker='*',label='Train')

ax.scatter(X_test,Y_test,c='red',label='Test')
regr_fin = linear_model.LinearRegression()

regr_fin.fit(X_train,Y_train)

Y_pred = regr_fin.predict(X_test)

print('MSE for Testing set : %f' % (meanSquarredError(Y_pred,Y_test)))

print('RMSE for Testing set : %f' % (rootMeanSquarredError(Y_pred,Y_test)))
x = range(0,int(X.max()))

y = x*regr.coef_



fig, (ax,ax1) = plt.subplots(1,2,figsize=(12,6))

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_title('Plot for Regression line fit on Train and Test data')

ax.scatter(X_train,Y_train,marker='*',label='Train')

ax.scatter(X_test,Y_test,c='red',label='Test')

ax.legend()

ax.plot(x,y,c='black')



ax1.set_title('Scatter plot between feature and Label')

ax1.set_ylim(-100,500)

sns.regplot(data=df,x='X',y='Y',ax=ax1)