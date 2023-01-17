



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
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head()
#checking for missing values

dfna = df.isna()

for column in dfna.columns.values.tolist():

    print(column)

    print(dfna[column].value_counts(dropna = False))
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
#plotting a boxplot to assess if gender has an impact on insurance cost

sns.boxplot(x = 'sex',y = 'charges', data = df)
from scipy import stats

df_ftestsex = df[['sex','charges']].groupby(df['sex'])

df_ftestsex.head()
ftest_val,p_val = stats.f_oneway(df_ftestsex.get_group('female')['charges'],df_ftestsex.get_group('male')['charges'])

print('F-Value : {} , pvalue : {}'.format(ftest_val,p_val))
sns.boxplot(x = 'region', y = 'charges', data = df)
df_ftestregion = df[['region','charges']].groupby('region')

df_ftestregion.head()
ftest_valregion, p_valregion = stats.f_oneway(df_ftestregion.get_group('southeast')['charges'],df_ftestregion.get_group('southwest')['charges'],\

                                             df_ftestregion.get_group('northeast')['charges'],df_ftestregion.get_group('northwest')['charges'])

print('F-Value : {} , pvalue : {}'.format(ftest_valregion,p_valregion))
sns.boxplot(x = 'smoker', y = 'charges', data = df)
df_ftestsmoker = df[['smoker','charges']].groupby('smoker')

df_ftestsmoker.head()
ftest_valsmoker, p_valsmoker = stats.f_oneway(df_ftestsmoker.get_group('yes')['charges'],df_ftestsmoker.get_group('no')['charges'])

print('F-Value : {} , pvalue : {}'.format(ftest_valsmoker,p_valsmoker))
#correlation matrix

dfcorr = df.corr()

#correlation with premium charges

dfcorrcharges = dfcorr[['charges']]

dfcorrcharges
corrcoeff_age,pval_age = stats.pearsonr(df['age'],df['charges'])

print("The Pearson Correlation Coefficient is", corrcoeff_age, " with a P-value of P =", pval_age)  
sns.regplot(x = 'age', y = 'charges', data = df)
corrcoeff_bmi,pval_bmi = stats.pearsonr(df['bmi'],df['charges'])

print("The Pearson Correlation Coefficient is", corrcoeff_bmi, " with a P-value of P =", pval_bmi)  
sns.regplot(x = 'bmi', y = 'charges', data = df)
corrcoeff_child,pval_child = stats.pearsonr(df['children'],df['charges'])

print("The Pearson Correlation Coefficient is", corrcoeff_child, " with a P-value of P =", pval_child) 
sns.regplot(x = 'children', y = 'charges', data = df)
#smokerdf is the one-hot encoded dataframe

smokerdf = pd.get_dummies(df['smoker']) 

smokerdf.rename(columns = {'no':'non-smoker','yes':'smokes'}, inplace = True)

smokerdf.head()
#we are dropping the smoker column to replace it with the one-hot encoded dataframe "smokerdf"

df = pd.concat([df,smokerdf],axis = 1)

df.drop(['smoker'],axis = 1, inplace = True)

df.head()
sexdf = pd.get_dummies(df['sex'])

sexdf.head()
df = pd.concat([df,sexdf],axis = 1)

df.drop(['sex'],axis = 1, inplace = True)

df.head()
#independent variable dataframe 

x = df[['age','bmi','non-smoker','smokes','female','male']]

y = df[['charges']]

x.head()
#residual plot for a linear relationship

sns.residplot(x[['age']],y)


sns.residplot(x[['bmi']],y)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
SCALE = StandardScaler()

xforscaling = x[['age','bmi']]

SCALE.fit(xforscaling)
print('The mean values of Age and BMI are {} and {} respectively'.format(SCALE.mean_[0],SCALE.mean_[1]))
scaledxdata = pd.DataFrame(SCALE.transform(xforscaling))

scaledxdata.rename(columns = {0:'Age_scaled',1:'BMI_scaled'},inplace = True)

scaledxdata.head()
xtemp = x.drop(['age','bmi'],axis = 1)
xscaleddata = pd.concat([scaledxdata,xtemp],axis = 1)

xscaleddata.head()
x_train,x_test,y_train,y_test = train_test_split(xscaleddata,y,random_state = 0)
linreg = LinearRegression()

linreg.fit(x_train,y_train)

print(linreg.coef_,linreg.intercept_)

ypredict_test = linreg.predict(x_test)

ypredict_test[0:10,:]
print(r2_score(y_test,ypredict_test))
poly = PolynomialFeatures(2, include_bias = False)

xpoly_train = poly.fit_transform(x_train)

xpoly_train = pd.DataFrame(xpoly_train)

xpoly_train.head()
linregpoly = LinearRegression()

linregpoly.fit(xpoly_train,y_train)
ypolypredict = linregpoly.predict(poly.fit_transform(x_test))
print(r2_score(y_test,ypolypredict))
#residual curve for linear regression

ypredict_test = pd.DataFrame(ypredict_test)

ypredict_test.rename(columns = {0:'charges'}, inplace = True)

ypredict_test.head()
y_testresid = y_test.reset_index(drop = True)

y_testresid.head()
linreg_resid = y_testresid - ypredict_test

linreg_resid.reset_index(inplace = True)

linreg_resid.head()
sns.scatterplot(x = 'index', y = 'charges', data = linreg_resid).set_title('Residual Plot')
#residual curve for polynomial regression

ypolypredict_resid = pd.DataFrame(ypolypredict)

ypolypredict_resid.rename(columns = {0:'charges'},inplace = True)

ypolypredict_resid.head()




linregpoly_resid = y_testresid - ypolypredict_resid

linregpoly_resid.reset_index(inplace = True)

linregpoly_resid.head()

sns.scatterplot(x = 'index', y = 'charges', data = linregpoly_resid).set_title('Residual Plot_Polynomial')
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge
xscaleddata.head()
y.shape
polyorders = [2,3,4,5,6,7]

parameters = [{'alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,100,1000]}]

results = {}

for order in polyorders:

    #getting the polynomical object 

    poly =  PolynomialFeatures(degree = order, include_bias = False)

    #transforming scaled x data to polynomial format

    xpolydata = poly.fit_transform(xscaleddata)

    #ridge model to introduce the regularization term

    ridgemodel = Ridge()

    gridsearchcv = GridSearchCV(ridgemodel,parameters,cv = 10)

    gridsearchcv.fit(xpolydata,y)

    #putting the results into the 'results' dictionary

    results[order] = [gridsearchcv.best_params_,gridsearchcv.best_score_]

    
results
#this is the same test/train split used with Polynomial order of 2 previously

x_train,x_test,y_train,y_test = train_test_split(xscaleddata,y,random_state = 0)

poly3 = PolynomialFeatures(degree = 3)

x_trainpoly3 = poly3.fit_transform(x_train)

x_testpoly3 = poly3.fit_transform(x_test)

ridgepoly3 = Ridge(alpha = 5)

ridgepoly3.fit(x_trainpoly3,y_train)
#predicting the values from the Ridge Regression model for x_testpoly3

y_testpoly3predict = ridgepoly3.predict(x_testpoly3)

#calculating the R2 value on the test set using Ridge Regression model (Polynomial order = 3, Regularization Parameter = 5)

r2_score(y_test,y_testpoly3predict)
#5th order polynomial

poly5 = PolynomialFeatures(degree = 5)

x_trainpoly5 = poly5.fit_transform(x_train)

x_testpoly5 = poly5.fit_transform(x_test)

ridgepoly5 = Ridge(alpha = 100)

ridgepoly5.fit(x_trainpoly5,y_train)
#predicting the values from the Ridge Regression model for x_testpoly5

y_testpoly5predict = ridgepoly5.predict(x_testpoly5)
#calculating the R2 value on the test set using Ridge Regression model (Polynomial order = 5, Regularization Parameter = 100)

r2_score(y_test,y_testpoly5predict)
#final model

ridgepoly5