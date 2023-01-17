##Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import statsmodels.api as sm
##Importing the dataset

data = pd.read_stata('../input/random-data-table/maketable1.dta')
data.head(3)

##droping the missing values
data = data.dropna()
data.head(3)
#Calculation of beta_hat using numpy
data['const'] = 1
X = data[['const', 'avexpr']]
y = data['logpgp95']
X = np.asarray(X)
y = np.asarray(y)
X_b = X
beta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(beta_best)
#Calculation of beta_hat using statsmodels
regsm = sm.OLS(endog=data['logpgp95'], exog=data[['const', 'avexpr']], \
    missing='drop')
type(regsm)
result = regsm.fit()
print(result.summary())