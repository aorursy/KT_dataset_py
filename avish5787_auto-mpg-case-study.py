# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#importing Libraries
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
#importing libraries and shuffling the data
df = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')
df = df.sample(frac = 1,random_state = 3)
df.head()
#Finding the unqiue values
df['origin'].unique()

#here we see there is '?' here.
#To check the shape of Dataframe
df.shape
#To check the data-type of the dataframe
df.info()
#To check descriptive statistics of data frame
df.describe()
df.isna().sum()*100/398
#Removing ?, replacing with mean value and conver the datatype to integer
df['horsepower'] = df['horsepower'].replace('?',np.nan)
df['horsepower'] = df['horsepower'].astype(float)
df['horsepower'] = df['horsepower'].replace(np.nan,df['horsepower'].mean())
df['horsepower'] = df['horsepower'].astype(int)

df['car name'].value_counts().nlargest(15).plot(kind='bar', figsize=(15,5))
plt.title("Number of vehicles by car brand name")
plt.ylabel('Number of vehicles')
plt.xlabel('car name');
df['mpg'].hist(bins = 5,color = 'red')
plt.title("Miles per gallon of vehicle")
plt.ylabel('Number of vehicles')
plt.xlabel('Miles per gallon');
df['cylinders'].hist(bins = 5,color = 'red')
plt.title("Number of Cylinders in a vehicle")
plt.ylabel('Number of vehicles')
plt.xlabel('Numer of Cylinder');
df['displacement'].hist(bins = 5,color = 'red')
plt.title("Displacement of the vehicle")
plt.ylabel('Number of vehicles')
plt.xlabel('Displacement');
df['weight'].hist(bins = 5,color = 'red')
plt.title("Weight of vehicle")
plt.ylabel('Number of vehicles')
plt.xlabel('Weight');
df['acceleration'].hist(bins = 5,color = 'red')
plt.title("Acceleration of vehicle")
plt.ylabel('Number of vehicles')
plt.xlabel('Acceleration');
df['model year'].hist(bins = 5,color = 'red')
plt.title("Model year of vehicle")
plt.ylabel('Number of vehicles')
plt.xlabel('Year');
df[['mpg','cylinders','displacement','weight','acceleration','model year','origin']].hist(figsize=(10,8),bins=6,color='Y')
plt.tight_layout()
plt.show()
plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="mpg", y="weight", data=df)
plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="mpg", y="displacement", data=df)
plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="horsepower", y="weight", data=df)
sns.scatterplot(x = 'weight',y = 'acceleration',data = df)
corr = df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
a = sns.heatmap(corr, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
#import library
import statsmodels.api as sm
#drop car name
df = df.drop('car name',axis = 1)
#Taking target variable and adding constant
y = df['mpg']
X = df.drop('mpg',axis = 1)
Xc = sm.add_constant(X)
#importing vif
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
pd.DataFrame([vif(Xc.values,i) for i in range(Xc.shape[1])],index = Xc.columns,columns=['VIF'])
#Fitting model
model = sm.OLS(y,Xc).fit()
model.summary()
#Dropping cylinders,horsepower,acceleration and fiting into model
Xc = Xc.drop(['cylinders','horsepower','acceleration'],axis = 1)
model = sm.OLS(y,Xc).fit()
model.summary()
#importing library and creating distplot
from scipy.stats import norm
norm.fit(model.resid)
sns.distplot(model.resid,fit = norm)
#importing library and creating probplot
import scipy.stats as st
st.probplot(model.resid,plot = plt)
plt.show()
#Jarque Bera Test to check normality
st.jarque_bera(model.resid)
#Applying Transformation
ly = np.log(y)
#Building model
model = sm.OLS(ly,Xc).fit()
model.summary()
#Again creating distplot after applying log
from scipy.stats import norm
norm.fit(model.resid)
sns.distplot(model.resid,fit = norm)
#Again creating probplot after applying log
import scipy.stats as st
st.probplot(model.resid,plot = plt)
plt.show()
#Again performing JB Test
st.jarque_bera(model.resid)
#Splitting X and y and adding constant
y = df['mpg']
X = df.drop('mpg',axis = 1)
Xc = sm.add_constant(X)
model = sm.OLS(y,Xc).fit()
y_pred = model.predict(Xc)
resids = model.resid

sns.regplot(x = y_pred,y = resids,lowess=True,line_kws={'color':'red'})
#Goldfeld Test for Checking Homoscedasticity
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ['F-Statistics','p-value']
test = sms.het_goldfeldquandt(model.resid,model.model.exog)
print(lzip(name,test))
#Splitting X and y and adding constant
y = df['mpg']
X = df.drop('mpg',axis = 1)
Xc = sm.add_constant(X)
#Fitiing in model and applying autocorrleation
model = sm.OLS(y,Xc).fit()
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(model.resid,lags = 30,alpha = 0.05)
model.summary()
#Splitting X and y and adding constant
y = df['mpg']
X = df.drop('mpg',axis = 1)
Xc = sm.add_constant(X)
#Fitting to model and applying linearity
model = sm.OLS(y,Xc).fit()
y_pred = model.predict(Xc)
resids = model.resid

sns.regplot(x = y_pred,y = resids,lowess=True,line_kws={'color':'red'})
#Fitting to model and applying linearity
model = sm.OLS(y,Xc).fit()
y_pred = model.predict(Xc)
resids = model.resid

sns.regplot(x = y,y = y_pred,lowess=True,line_kws={'color':'red'})
#Rainbow test for Linearity
import statsmodels.api as sm
sm.stats.diagnostic.linear_rainbow(res = model,frac = 0.5)
#Applying in target variables
y = df['mpg']
X = df.drop('mpg',axis = 1)
#Splitting the dataframe in 70:30(train and test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)
#printing the shape of train and test
print(X_train.shape,X_test.shape)
#importing the library and building the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

lr = LinearRegression()
lr.fit(X_train,y_train)

y_test_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)

print('r-square of Train :',r2_score(y_train,y_train_pred))
print('rmse for Train: ',np.sqrt(mean_squared_error(y_train,y_train_pred)))

print('\n')
print('r-square of Test :',r2_score(y_test,y_test_pred))
print('rmse for Test: ',np.sqrt(mean_squared_error(y_test,y_test_pred)))

#importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score,mean_squared_error
#Fitting lr to rfe
lr = LinearRegression()
rfe = RFE(lr,n_features_to_select = 8)
rfe.fit(X,y)
#Checking the rfe support function
rfe.support_
pd.DataFrame(rfe.ranking_,index = X.columns,columns=['SELECT'])
#Applying train test split method and splitting the data in 70:30 ratio
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
    
    #for r2score
    r2 = r2_score(y_test,y_test_pred)
    r2score.append(r2)
    
    #for rmse
    rms = np.sqrt(mean_squared_error(y_test,y_test_pred))
    rmse.append(rms)
#plot for r2score
plt.plot(range(1,8),r2score)
#printing r2score
r2score
#plot for rmse
plt.plot(range(1,8),rmse)
#printing rmse
rmse
from sklearn.model_selection import KFold,GridSearchCV

params = {'n_features_to_select':list(range(1,8))}
lr = LinearRegression()
rfe = RFE(lr)  #estimator

kf = KFold(n_splits=3,random_state=3)

gsearch = GridSearchCV(rfe,param_grid=params,scoring = 'r2',cv = kf,return_train_score=True)
#to find the best parameter for our data
gsearch.get_params
#importing the SFS
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
sfs1 = sfs(lr,k_features=7,scoring='r2',cv = 3,verbose = 2)
sfs1.fit(X,y)
#Making the dataframe as subset and transforming it
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
