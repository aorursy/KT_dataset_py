# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import scipy.stats as st

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#importing housing dataset and shuffling the data
df = pd.read_csv('/kaggle/input/housing-dataset/Housing.csv')
#Dropping the unnamed columns
df  =df.drop('Unnamed: 0',axis = 1)
#Shuffling the data
df = df.sample(frac = 1,random_state = 3)
df.head()
#To check the shape of dataset(row,columns)
df.shape
# Let's look at some statistical information about our dataframe.
df.describe()
# What type of values are stored in the columns?
df.info()
#Checking all the numerical variables
df.select_dtypes(exclude = object).columns
obj_cols = list(df.select_dtypes(include = object).columns)
obj_cols
for col in obj_cols:
    print(df[col].value_counts())
#Use get dummies to further categorize the Categorical variable
df = pd.get_dummies(data = df,columns = obj_cols,drop_first=True)
df.info()
#all the object type got converted to unsigned integer format.
#To find the % of missing values
df.isna().sum()*100/546
#We wanted to see which particular column has missing data
sns.heatmap(df.isnull())
#Describe for price
df['price'].describe()
#Boxplot for price before removing outlier
sns.boxplot(df.price)
#Removing outlier for Price
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3-Q1
IQR

upper = Q3+1.5*IQR
print(upper)

c_df = df[df['price']<upper]
#Boxplot for price after removing outlier
sns.boxplot(c_df.price)
#Describe for lotsize
c_df['lotsize'].describe()
#Boxplot for lotsize before removing outlier
sns.boxplot(c_df.lotsize)
#Removing outlier for lotsize
Q1 = c_df['lotsize'].quantile(0.25)
Q3 = c_df['lotsize'].quantile(0.75)
IQR = Q3-Q1
IQR

upper = Q3+1.5*IQR
print(upper)

c_df = c_df[c_df['lotsize']<upper]
c_df.head()
#Boxplot for price after removing outlier
sns.boxplot(c_df.lotsize)
#Final shape for clean data
c_df.shape
df.head()
plt.subplots(figsize=(12,9))
sns.distplot(df['price'], fit=st.norm)

# Get the fitted parameters used by the function

(mu, sigma) = st.norm.fit(df['price'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
st.probplot(df['price'], plot=plt)
plt.show()

#we use log function which is in numpy
df['price'] = np.log1p(df['price'])

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(df['price'], fit=st.norm)

# Get the fitted parameters used by the function

(mu, sigma) = st.norm.fit(df['price'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
st.probplot(df['price'], plot=plt)
plt.show()
corr = df.corr()['price']
corr[np.argsort(corr, axis=0)[::-1]]
corrMatrix=df[['price','lotsize','bedrooms','bathrms','stories','garagepl','driveway_yes','recroom_yes','fullbase_yes','gashw_yes','airco_yes','prefarea_yes']].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');
sns.scatterplot(df['lotsize'],df['price'])
plt.title("Lotsize Vs Price ")
plt.ylabel("Price")
plt.xlabel("Lotsize in sq feet");
import statsmodels.api as sm
#Transform the dataset
#Separate the input and target variables into X and Y.
y = c_df['price']
X = c_df.drop('price',axis = 1)
Xc = sm.add_constant(X)
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
pd.DataFrame([vif(Xc.values,i) for i in range(Xc.shape[1])],index = Xc.columns, columns=['VIF'])
model = sm.OLS(y,Xc).fit()
model.summary()
#Drop bedroom
Xc = Xc.drop('bedrooms',axis = 1)
model = sm.OLS(y,Xc).fit()
model.summary()
from scipy.stats import norm
norm.fit(model.resid)
sns.distplot(model.resid,fit = norm)
import scipy.stats as st

st.probplot(model.resid,plot = plt)
plt.show()
#Jarque-Bera (JB) test
st.jarque_bera(model.resid)
#Transform the data using log
ly = np.log(y)
#Building model with log
model = sm.OLS(ly,Xc).fit()
model.summary()
#Jarque-Bera (JB) test
st.jarque_bera(model.resid)
y_pred = model.predict(Xc)

sns.regplot(x = y_pred,y = model.resid,lowess = True,line_kws = {'color':'red'})
plt.show()
#Goldfeld Test for Checking Homoscedasticity
import statsmodels.stats.api as sms
test = sms.het_goldfeldquandt(y = model.resid,x = Xc)
test 
import statsmodels.tsa.api as smt
acf = smt.graphics.plot_acf(model.resid,lags = 30)
acf.show()
sns.regplot(x = y_pred, y = ly ,lowess = True,line_kws = {'color':'red'})
plt.show()
import statsmodels.api as sm
sm.stats.diagnostic.linear_rainbow(res = model,frac = 0.5)
#Assumptions completed
plt.scatter(ly, y_pred)
plt.plot(ly,ly,'r')
plt.show()
#Taking dependent and independent variable.
y = c_df['price']
X = c_df.drop('price',axis = 1)
#importing libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
#Fitting RFE
lr = LinearRegression()
rfe = RFE(lr,n_features_to_select=13)
rfe.fit(X,y)

#WE ASKED ALGORITHM TO TELL US WHICH FEATURE IS WORSE BY PUTTING 'n_features_to_select = 13
rfe.support_

pd.DataFrame(rfe.ranking_,index = X.columns,columns = ['Select'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)
no_of_cols = X_train.shape[1]
r2score = []
rmse = []

lr = LinearRegression()   #estimator

for i in range(no_of_cols):
    rfe = RFE(lr,n_features_to_select=i+1)
    rfe.fit(X_train,y_train)
    y_test_pred = rfe.predict(X_test)
    
    #for r2score 
    r2 = r2_score(y_test,y_test_pred)
    r2score.append(r2)
    
    #now for mean square error
    rms = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmse.append(rms)
plt.plot(range(1,12),r2score)
r2score
plt.plot(range(1,12),rmse)
rmse
from sklearn.model_selection import KFold,GridSearchCV

params = {'n_features_to_select': list(range(1,13))}
lr = LinearRegression()
rfe = RFE(lr)

kf = KFold(n_splits=3,random_state=3)

gsearch = GridSearchCV(rfe,param_grid=params,cv = kf,return_train_score=True,scoring='r2')
gsearch.fit(X,y)
gsearch.best_params_
pd.DataFrame(gsearch.cv_results_)
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
lr = LinearRegression()
sfs1 = sfs(lr,k_features=11,cv = 3,scoring='r2',verbose = 2)

sfs1.fit(X,y)

sf = pd.DataFrame(sfs1.subsets_).T
sf
plt.plot(sf['avg_score'])
plt.show()
cols = list(sfs1.k_feature_names_)
cols
X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.3, random_state=3)

lr.fit(X_train, y_train)
y_test_pred = lr.predict(X_test)

plt.scatter(y_test, y_test_pred)
plt.plot(y_test, y_test,'r')
y = df['price']
X = df.drop('price',axis = 1)
#random_state is the seed used by the random number generator. It can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()

# fit the model to the training data
lr.fit(X_train,y_train)
# print the intercept
print(lr.intercept_)
# Let's see the coefficient
cof_df = pd.DataFrame(lr.coef_,X_test.columns,columns=['Coefficient'])
cof_df
# Making predictions using the model
y_pred = lr.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error

y_test_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)
print('r-square for train: ', r2_score(y_train,y_train_pred))
print('RMSE for train: ',np.sqrt(mean_squared_error(y_train,y_train_pred)))

print('\n')
print('r-square for test: ', r2_score(y_test,y_test_pred))
print('RMSE for test: ', np.sqrt(mean_squared_error(y_test,y_test_pred)))
