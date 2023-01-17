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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV

from sklearn.metrics import r2_score
pd.set_option('display.max_columns',None)
df=pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')

df.head()
df.shape
df.info()
df.isna().sum()
cols=df.columns

percent_of_null=[]

for i in cols:

    p=(df[i].isnull().sum()/df.shape[0])*100

    percent_of_null.append(p)



for l,n in zip(df.columns,percent_of_null):

    print(l,':',n)
df['Status']=df['Status'].replace('Developing',1)

df['Status']=df['Status'].replace('Developed',0)
df=df.drop(['Country','Year'],axis=1)
df=df.fillna(df.median())
df.isna().sum()   # rechecking missing values after treatment
df.describe()
col={'Life expectancy ':1,'Adult Mortality':2,'infant deaths':3,'Alcohol':4,'percentage expenditure':5,'Hepatitis B':6, 

        'Measles ':7,' BMI ':8,'under-five deaths ':9,'Polio':10, 'Total expenditure':11,'Diphtheria ':12, 

     ' HIV/AIDS':13,'GDP':14,'Population':15,

        ' thinness  1-19 years':16,' thinness 5-9 years':17,'Income composition of resources':18,'Schooling':19}
plt.figure(figsize=(20,30))



for variable,i in col.items():

                     plt.subplot(5,4,i)

                     plt.boxplot(df[variable])

                     plt.title(variable)



plt.show()
df=df.transform(lambda x : x**0.5)  # Outliers treatment
sns.distplot(df['Life expectancy '],kde=True)
sns.barplot(data=df, x='Life expectancy ',y='Status',orient = 'h')
disease_cols=df[['Life expectancy ','Alcohol','Hepatitis B','Measles ',' BMI ','Polio','Diphtheria ',' HIV/AIDS']]
sns.pairplot(disease_cols,diag_kind='kde')
disease_cols.corr()
measures_cols=df[['Life expectancy ','Adult Mortality','infant deaths','under-five deaths ',' thinness  1-19 years',' thinness 5-9 years','Schooling']]
sns.pairplot(measures_cols,diag_kind='kde')
measures_cols.corr()
income_exp_cols=df[['Life expectancy ','percentage expenditure','Total expenditure','GDP','Population',

                    'Income composition of resources']]
sns.pairplot(income_exp_cols, diag_kind='kde')
income_exp_cols.corr()
X=df.drop('Life expectancy ',axis=1)

y=df['Life expectancy ']
X_constant = sm.add_constant(X)

lin_reg = sm.OLS(y,X_constant).fit()

lin_reg.summary()
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(lin_reg.resid, alpha=0.05)

acf.show()
from scipy import stats

print(stats.jarque_bera(lin_reg.resid))
sns.distplot(lin_reg.resid)
sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



def linearity_test(model, y):

    pred_vals = model.predict()

    resids = model.resid



    fig, ax = plt.subplots(1,2)

    

    sns.regplot(x=pred_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})

    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)

    ax[0].set(xlabel='Predicted', ylabel='Observed')



    sns.regplot(x=pred_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})

    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)

    ax[1].set(xlabel='Predicted', ylabel='Residuals')

    

linearity_test(lin_reg, y)  
lin_reg.resid.mean()
import statsmodels.stats.api as sms

from statsmodels.compat import lzip



model = lin_reg

pred_vals = model.predict()

resids = model.resid

resids_standardized = model.get_influence().resid_studentized_internal

fig, ax = plt.subplots(1,2)



sns.regplot(x=pred_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})

ax[0].set_title('Residuals vs Fitted', fontsize=16)

ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

sns.regplot(x=pred_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})

ax[1].set_title('Scale-Location', fontsize=16)

ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')



name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(model.resid, model.model.exog)

lzip(name, test)
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred=lr.predict(X_test)
coll=['Status','Adult Mortality','infant deaths','Alcohol','percentage expenditure','Hepatitis B', 

        'Measles ',' BMI ','under-five deaths ','Polio', 'Total expenditure','Diphtheria ', 

     ' HIV/AIDS','GDP','Population',

        ' thinness  1-19 years',' thinness 5-9 years','Income composition of resources','Schooling']
coefficients = pd.Series(lr.coef_, index= coll)

print(coefficients)
print('Intercept: ',lr.intercept_)

print('Mean absolute error for test: ',mean_absolute_error(y_test,y_pred))

print('Mean Squared error for test: ',mean_squared_error(y_test,y_pred))

print('Root mean squared error for test: ',np.sqrt(mean_squared_error(y_test,y_pred)))

print('Accuracy for train: ',lr.score(X_train, y_train))

print('Accuracy for test: ',lr.score(X_test, y_test))

print('R square of test: ',r2_score(y_test,y_pred))
rcv=RidgeCV(cv=5)

rcv.fit(X_train,y_train)

y_pred_rr=rcv.predict(X_test)
print('Optimal alpha: ',rcv.alpha_)
coefficients_rcv=pd.Series(rcv.coef_,index=coll)

print(coefficients_rcv)

print('Intercept: ',rcv.intercept_)

print('Mean absolute error for test: ',mean_absolute_error(y_test,y_pred_rr))

print('Mean Squared error for test: ',mean_squared_error(y_test,y_pred_rr))

print('Root mean squared error for test: ',np.sqrt(mean_squared_error(y_test,y_pred_rr)))

print('Train accuracy: ',rcv.score(X_train,y_train))

print('Test accuracy: ',rcv.score(X_test,y_test))

print('R square of test:',r2_score(y_test,y_pred_rr))
lassocv=LassoCV(cv=5,random_state=3)

lassocv.fit(X_train,y_train)

y_pred_lasso=lassocv.predict(X_test)
print('Optimal alpha: ',lassocv.alpha_)

print('No. of interations: ',lassocv.n_iter_)
coefficients_lasso=pd.Series(lassocv.coef_,index=coll)

print(coefficients_lasso)

print('Intercept: ',lassocv.intercept_)

print('Mean absolute error for test: ',mean_absolute_error(y_test,y_pred_lasso))

print('Mean Squared error for test: ',mean_squared_error(y_test,y_pred_lasso))

print('Root mean squared error for test: ',np.sqrt(mean_squared_error(y_test,y_pred_lasso)))

print('Train accuracy: ',lassocv.score(X_train,y_train))

print('Test accuracy: ',lassocv.score(X_test,y_test))

print('R square of test:',r2_score(y_test,y_pred_lasso))
en_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True, 

                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, 

                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
en_cv.fit(X_train,y_train)

y_pred_en=en_cv.predict(X_test)
print('Optimal alpha: ',en_cv.alpha_)

print('Optimal l1_ratio: ',en_cv.l1_ratio_)

print('Number of iterations: ',en_cv.n_iter_)
coefficients_en=pd.Series(en_cv.coef_,index=coll)

print(coefficients_en)

print('Intercept: ',en_cv.intercept_)

print('Mean absolute error for test: ',mean_absolute_error(y_test,y_pred_en))

print('Mean Squared error for test: ',mean_squared_error(y_test,y_pred_en))

print('Root mean squared error for test: ',np.sqrt(mean_squared_error(y_test,y_pred_en)))

print('Train accuracy: ',en_cv.score(X_train,y_train))

print('Test accuracy: ',en_cv.score(X_test,y_test))

print('R square of test:',r2_score(y_test,y_pred_en))
coefficients_en[abs(coefficients_en)>0.05]