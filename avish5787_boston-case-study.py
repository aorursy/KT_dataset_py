# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#importing packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_rows = None

pd.options.display.max_columns = None
#Loading the data,dropping Unnamed column and shuffling the data

df = pd.read_csv('/kaggle/input/boston-data-set/boston.csv')

df = df.drop('Unnamed: 0',axis = 1)

df0 = df.sample(frac = 1,random_state = 3)

df.head()
#Checking Shape

df.shape
#To check the data-type of the dataframe

df.info()
#To check the descriptive stats of the dataset

df.describe()
#To check all the columns of dataset

df.columns
df.isna().sum()
plt.subplots(figsize=(12,9))

sns.distplot(df['Price'], fit=st.norm)



# Get the fitted parameters used by the function



(mu, sigma) = st.norm.fit(df['Price'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

st.probplot(df['Price'], plot=plt)

plt.show()
#we use log function which is in numpy

df['Price'] = np.log1p(df['Price'])



#Check again for more normal distribution



plt.subplots(figsize=(12,9))

sns.distplot(df['Price'], fit=st.norm)



# Get the fitted parameters used by the function



(mu, sigma) = st.norm.fit(df['Price'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

st.probplot(df['Price'], plot=plt)

plt.show()

corr = df.corr()['Price']

corr[np.argsort(corr, axis=0)[::-1]]
corrMatrix=df[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','Price']].corr()



sns.set(font_scale=1.10)

plt.figure(figsize=(10, 10))



sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='viridis',linecolor="white")

plt.title('Correlation between features');
plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="ZN", y="Price", data=df)
plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="TAX", y="Price", data=df)
sns.scatterplot(df['LSTAT'],df['Price'])
sns.scatterplot(df['AGE'],df['Price'])
#importing statsmodel library

import statsmodels.api as sm
#Taking X and y and adding constant

y = df['Price']

X = df.drop('Price',axis = 1)

Xc = sm.add_constant(X)
#Importing VIF and applying its function

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

pd.DataFrame([vif(Xc.values,i)for i in range(Xc.shape[1])],index = Xc.columns,columns=['VIF'])
#Building OLS model

model = sm.OLS(y,Xc).fit()

model.summary()
#Drop INDUS,AGE

Xc = Xc.drop(['INDUS','AGE'],axis = 1)

model = sm.OLS(y,Xc).fit()

model.summary()
from scipy.stats import norm

norm.fit(model.resid)

sns.distplot(model.resid,fit = norm)
import scipy.stats as st

st.probplot(model.resid,plot = plt)

plt.show()
#Jarque Bera Test to check normality

st.jarque_bera(model.resid)
#Applying Transformation

ly = np.log(y)
model = sm.OLS(ly,Xc).fit()

model.summary()
#Jarque-Bera test

st.jarque_bera(model.resid)
y_pred = model.predict(Xc)

resids = model.resid



sns.regplot(x = y_pred,y = resids,lowess=True,line_kws={'color':'red'})

plt.xlabel('y_pred')

plt.ylabel('resids')
#Goldfeld Test for Checking Homoscedasticity

import statsmodels.stats.api as sm

from statsmodels.compat import lzip



name = ['F-statstics','p-value']

test = sm.het_goldfeldquandt(model.resid,model.model.exog)

print(lzip(name,test))
import statsmodels.api as sm

y = df['Price']

X = df.drop('Price',axis = 1)

Xc = sm.add_constant(X)
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(model.resid,lags = 30)

acf.show()
import statsmodels.api as sm

y = df['Price']

X = df.drop('Price',axis = 1)

Xc = sm.add_constant(X)
sns.regplot(x = y_pred, y = resids,lowess=True,line_kws={'color':'red'})

plt.xlabel('y_pred')

plt.ylabel('residuals')
#Rainbow Test for Linearity of Residuals

sm.stats.diagnostic.linear_rainbow(res = model,frac = 0.5)
#Applying the dependent and independent variable

y = df['Price']

X = df.drop('Price',axis = 1)
#Splitting the dataframe in 70:30(train and test)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)
#printing the shape of train and test

print(X_train.shape,X_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error



lr = LinearRegression()

lr.fit(X_train,y_train)



y_test_pred = lr.predict(X_test)

y_train_pred = lr.predict(X_train)

print('r-square for train: ', r2_score(y_train,y_train_pred))

print('RMSE for train: ',np.sqrt(mean_squared_error(y_train,y_train_pred)))



print('\n')

print('r-square for test: ', r2_score(y_test,y_test_pred))

print('RMSE for test: ', np.sqrt(mean_squared_error(y_test,y_test_pred)))
#Splitting into X and y

y = df['Price']

X = df.drop('Price',axis = 1)
#importng LR,RFE,R2-SQUARE,MEAN SQUARE

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.metrics import r2_score,mean_squared_error
#Fitting rfe to lr 

lr = LinearRegression()

rfe = RFE(lr,n_features_to_select=13)

rfe.fit(X,y)



#WE ASKED ALGORITHM TO TELL US WHICH FEATURE IS WORSE BY PUTTING 'n_features_to_select = 13
#rfe support function

rfe.support_
pd.DataFrame(rfe.ranking_,index = X.columns,columns=['SELECT'])
#importing train test split method and splitting data into 70:30 ratio

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)
no_of_cols = X_train.shape[1]

r2score = []

rmse = []



lr = LinearRegression()  #estimator



for i in range(no_of_cols):

    rfe = RFE(lr,n_features_to_select=i+1)

    rfe.fit(X_train,y_train)

    y_test_pred = rfe.predict(X_test)

    

    #r2score

    r2 = r2_score(y_test,y_test_pred)

    r2score.append(r2)

    

    #rmse

    rms = np.sqrt(mean_squared_error(y_test,y_test_pred))

    rmse.append(rms)
plt.plot(range(1,14),r2score)
r2score
plt.plot(range(1,14),rmse)
rmse
#Importing KFold and GridSearchCV

from sklearn.model_selection import KFold,GridSearchCV



params = {'n_features_to_select': list(range(1,14))}

lr = LinearRegression()

rfe = RFE(lr)



kf = KFold(n_splits=3,random_state=True)



gsearch = GridSearchCV(rfe,param_grid=params,scoring = 'r2',cv = kf,return_train_score=True)

gsearch.fit(X,y)
#Finding the best parameters

gsearch.best_params_
#Displaying the results

pd.DataFrame(gsearch.cv_results_)
#importing SFS and fitting it

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

sfs1 = sfs(lr,k_features=13,scoring='r2',cv = 3,verbose=2)

sfs1.fit(X,y)
#Making dataframe for subset of sfs and transforming it

sf = pd.DataFrame(sfs1.subsets_).T

sf
plt.plot(sf['avg_score'])
cols = list(sfs1.k_feature_names_)

cols
X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.3, random_state=3)



lr.fit(X_train, y_train)

y_test_pred = lr.predict(X_test)



plt.scatter(y_test, y_test_pred)

plt.plot(y_test, y_test,'r')
#importing lasso library

from sklearn.linear_model import Lasso,LassoCV
#Applying Lasso function and fitting it

alphas = np.linspace(0.0001,1,100)

lasso_cv = LassoCV(alphas = alphas,cv = 3,random_state=3)

lasso_cv.fit(X,y)

lasso_cv
#the best alpha

lasso_cv.alpha_
#Finding the R-square for train and test using Lasso

lasso = Lasso(alpha=lasso_cv.alpha_,random_state=3)



lasso.fit(X_train,y_train)

y_train_pred = lasso.predict(X_train)

y_test_pred = lasso.predict(X_test)



print('r-square of Train: ',r2_score(y_train,y_train_pred))

print('r-square of Test: ',r2_score(y_test,y_test_pred))
#Coeffiecient of Lasso

lasso.coef_
#importing ridge library

from sklearn.linear_model import Ridge,RidgeCV
#Applying Ridge function and fitting it

alphas = np.logspace(0,1,200)

ridge_cv = RidgeCV(alphas = alphas,scoring = 'r2',cv = 3)

ridge_cv.fit(X,y)

ridge_cv
#the best alpha

ridge_cv.alpha_
#Finding the R-square for train and test using Lasso

ridge = Ridge(alpha =ridge_cv.alpha_,random_state=3 )   #this is ridge



ridge.fit(X_train, y_train)

y_train_pred = ridge.predict(X_train)

y_test_pred = ridge.predict(X_test)





print('r-sqaured of Train', r2_score(y_train, y_train_pred))     #this part we are putting the best alpha and then finding the predicted value

print('r-sqaured of Test', r2_score(y_test, y_test_pred))
#Coefficient for Ridge

ridge_cv.coef_
#importing elastic net library

from sklearn.linear_model import ElasticNet,ElasticNetCV
alphas = np.logspace(-4,0,100)

en_cv = ElasticNetCV(alphas = alphas, cv = 3, random_state=3)

en_cv.fit(X,y)

en_cv
#best alpha for elasticnet

en_cv.alpha_