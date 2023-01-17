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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head()
sns.scatterplot(df.age,df.charges)

plt.show()
df.groupby('sex')['charges'].mean().plot.bar()

plt.show()
sns.scatterplot(df.bmi,df.charges)

plt.show()
sns.boxplot(y='charges',x='children',data=df)

plt.show()
df.groupby('smoker')['charges'].mean().plot.bar()

plt.show
sns.boxplot(x='region',y='charges',data=df)

plt.show()
df.groupby('region')['charges'].median().sort_values(ascending=False)
# Replacing region categories with labels as per the median values

#Region with highest median will get the highest numerical values

df.region=df.region.map({'northeast':4,'southeast':3,'northwest':2,'southwest':1})
#Creating dummies for smoker with drop_first=True

df=pd.get_dummies(df,columns=['sex','smoker'],drop_first=True)
import statsmodels.api as  sm
X=df.drop(columns='charges')

Y=df.charges
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,random_state=0)
X_train_const=sm.add_constant(X_train)
model=sm.OLS(Y_train,X_train_const).fit()

model.summary()
X_test.shape,X_train.shape
X_test_const=sm.add_constant(X_test)

y_pred=model.predict(X_test_const)
SSE=np.sum((Y_test-y_pred)**2)

SSR=np.sum((y_pred-Y_test.mean())**2)

SST=SSE+SSR
R2=SSR/SST

R2
N=len(X_test)# test data size

p=len(X_test.columns)

Adj_R2=1-(((1-R2)*(N-1))/(N-p-1))

Adj_R2
rmse=np.sqrt(np.sum((y_pred-Y_test)**2)/N)

rmse
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X_train,Y_train)
y_pred_lr=lr.predict(X_test)
lr.score(X_test,Y_test)
from sklearn.metrics import mean_squared_error

rmse_lr=np.sqrt(mean_squared_error(Y_test,y_pred_lr))

rmse_lr
model.summary()
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(model.resid, lags=40 , alpha=0.05)

acf.show()
from scipy.stats import shapiro

shapiro(model.resid)

#Since p value is almost 0 so we reject null hypothesis and residual distribution is not normal
#Here we can see visually also that residuals are not normally distributed and there is high skewness

sns.distplot(model.resid)

plt.show()
%matplotlib inline

%config InlineBackend.figure_format ='retina'

import seaborn as sns 

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



def linearity_test(model, y):

    '''

    Function for visually inspecting the assumption of linearity in a linear regression model.

    It plots observed vs. predicted values and residuals vs. predicted values.

    

    Args:

    * model - fitted OLS model from statsmodels

    * y - observed values

    '''

    fitted_vals = y_pred

    resids = y_pred-Y_test



    fig, ax = plt.subplots(1,2)

    

    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})

    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)

    ax[0].set(xlabel='Predicted', ylabel='Observed')



    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})

    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)

    ax[1].set(xlabel='Predicted', ylabel='Residuals')

    

linearity_test(model, Y_test)  
import statsmodels.api as sm

sm.stats.diagnostic.linear_rainbow(res=model, frac=0.5)
np.mean(model.resid)

# mean is close to 0 so residual are linear
from statsmodels.compat import lzip

from statsmodels.compat import lzip

%matplotlib inline

%config InlineBackend.figure_format ='retina'

import statsmodels.stats.api as sms

sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



model = model

fitted_vals = model.predict()

resids = model.resid

resids_standardized = model.get_influence().resid_studentized_internal

fig, ax = plt.subplots(1,2)



sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})

ax[0].set_title('Residuals vs Fitted', fontsize=16)

ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})

ax[1].set_title('Scale-Location', fontsize=16)

ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')



name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(model.resid, model.model.exog)

lzip(name, test)
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_train_const.values, i) for i in range(X_train_const.shape[1])]

pd.DataFrame({'vif': vif}, index=X_train_const.columns)