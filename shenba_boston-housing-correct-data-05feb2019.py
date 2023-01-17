# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/boston_train.csv')
train.head()
train.info()
import matplotlib.pyplot as plt
plt.hist(train['medv'])

plt.show()
from statsmodels.api import OLS
from sklearn.linear_model import LinearRegression
linear_reg1 = LinearRegression(fit_intercept=True)
input_vars = [col for col in train.columns]
input_vars.remove('medv')
input_vars.remove('ID')
input_vars
linear_reg1.fit(X=train[input_vars], y=train['medv'])
linear_reg1.intercept_
linear_reg1.coef_
linear_reg1.score(X=train[input_vars], y=train['medv'])
import statsmodels.api as sm
train = sm.add_constant(train)
train.head()
input_vars = [col for col in train.columns]

input_vars.remove('ID')

input_vars.remove('medv')
statsmodel1 = sm.OLS(train['medv'], train[input_vars])
statsmodel1.fit().summary()
#check for multi collinearity
import seaborn as sns
input_vars
corr_matrx_cols = input_vars.remove('const')
corr = train[input_vars].corr()



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, annot=True,square=True)



fig=plt.gcf()

fig.set_size_inches(12,9)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)
input_vars = [col for col in train.columns]

input_vars.remove('ID')

input_vars.remove('medv')

input_vars.remove('rad')
model2 = sm.OLS(train['medv'], train[input_vars])

result2 = model2.fit()

result2.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
train.shape
train.columns
X_train = train.iloc[:, :-1]

for i in range(X_train.shape[1]):

    print('column: {} has VIF: {}'.format(X_train.columns[i],variance_inflation_factor(exog=X_train.values, exog_idx=i)))
#one more way to calculate VIF
# Removing variable has threshold value of VIF above 5

print ("\nVariance Inflation Factor")

cnames = X_train.columns

for i in np.arange(0,len(cnames)):

    xvars = list(cnames)

    yvar = xvars.pop(i)

    mod = sm.OLS(X_train[yvar],(X_train[xvars]))

    res = mod.fit()

    vif = 1/(1-res.rsquared)

    print (yvar,round(vif,3))
#we see fromm results above as well as from the corr matrix that rad and tax are two cols that are highly correlated; 

#let's drop one and see the impact
X_train2 = X_train.copy()
X_train2.drop(columns=['tax'], inplace=True)
for i in range(X_train2.shape[1]):

    print('column: {} has VIF: {}'.format(X_train2.columns[i],variance_inflation_factor(exog=X_train2.values, exog_idx=i)))
#much better values for VIF; lets build a model with these cols
input_vars.remove('tax')
input_vars
statsmodel2 = sm.OLS(train['medv'], train[input_vars])

reg_model2 = statsmodel2.fit()
reg_model2.summary()
input_vars = [col for col in train.columns]

input_vars.remove('ID')

input_vars.remove('medv')

input_vars.remove('dis')

input_vars.remove('tax')
statsmodel2 = sm.OLS(train['medv'], train[input_vars])

statsmodel2.fit().summary()
#remove the vars that are not significant

input_vars = [col for col in train.columns]

input_vars.remove('ID')

input_vars.remove('medv')

input_vars.remove('dis')

input_vars.remove('tax')

input_vars.remove('crim')

input_vars.remove('zn')

input_vars.remove('indus')
statsmodel3 = sm.OLS(train['medv'], train[input_vars])

statsmodel3.fit().summary()
# Removing variable has threshold value of VIF above 5

print ("\nVariance Inflation Factor")

cnames = train.columns

for i in np.arange(0,len(cnames)):

    xvars = list(cnames)

    yvar = xvars.pop(i)

    mod = sm.OLS(train[yvar],(train[xvars]))

    res = mod.fit()

    vif = 1/(1-res.rsquared)

    print (yvar,round(vif,3))
#remove the vars that are not significant

input_vars = [col for col in train.columns]

input_vars.remove('ID')

input_vars.remove('medv')

input_vars.remove('tax')



statsmodel4 = sm.OLS(train['medv'], train[input_vars])

statsmodel4.fit().summary()
#remove the vars that are not significant

input_vars = [col for col in train.columns]

input_vars.remove('ID')

input_vars.remove('medv')

input_vars.remove('tax')

input_vars.remove('crim')

input_vars.remove('indus')

input_vars.remove('age')



statsmodel5 = sm.OLS(train['medv'], train[input_vars])

reg_model5 = statsmodel5.fit()

reg_model5.summary()
residuals = train['medv'] - reg_model5.predict(train[input_vars])
plt.scatter(x=reg_model5.predict(train[input_vars]), y=residuals)

plt.show()