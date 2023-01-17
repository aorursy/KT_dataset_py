##Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import statsmodels.api as sm
##Importing the dataset

data = pd.read_stata('../input/maketable4/maketable4.dta')
data.head(3)
##Plotting the relationship between X and Y
#we can observe a strongly positive relationship between x='avexpr' & y='logpgp95'

plt.style.use('seaborn')
data.plot(x='avexpr', y='logpgp95', kind='scatter', color='purple')
plt.show()
##Plotting the relationship between 'logem4' and 'avexpr' to see if 'avexpr' may be an endogenous variable!
#we can observe a fairly negative relationship between 'logem4' and 'avexpr'.
plt.style.use('seaborn')
data.plot(x='logem4', y='avexpr', kind='scatter', color='orange')
plt.show()
##2SLS model##

data = data[data['baseco'] == 1]

#We add a constant column with value 1 to the current data to make the true dimension(i.e., beta0*1=beta0)
data['const'] = 1

###FIRST STAGE regression model
reg_fs = sm.OLS(data['avexpr'],
                    data[['const', 'logem4']],
                    missing='drop').fit()
print(reg_fs.summary())
###SECOND STAGE regression model

#we first need to retrive the residuals:
data['residual'] = reg_fs.resid

reg_ss = sm.OLS(data['avexpr'],
                    data[['const', 'residual']],
                    missing='drop').fit()
print(reg_ss.summary())