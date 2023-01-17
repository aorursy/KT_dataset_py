
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Any results you write to the current directory are saved as output.
USAhousing=pd.read_csv("../input/USA_Housing.csv")
USAhousing.head(5)
USAhousing.info()
USAhousing.describe()
USAhousing.columns
sns.heatmap(USAhousing.corr(),annot=True)

#We can see that the indepedent variables 'Number of Rooms' and 'Number of Bedrooms' are highly correlated

plt.figure(figsize=(20,12))
plt.scatter(USAhousing['Avg. Area Number of Rooms'],USAhousing['Price'])
plt.figure(figsize=(20,12))
plt.scatter(USAhousing['Avg. Area Number of Bedrooms'],USAhousing['Price'])

sns.pairplot(USAhousing)


#Now let us look at the distribution of the housing prices
#We can see that every parameter has data that is almost normally distributed except for the parameter Average area number of bedrooms.
#you can further confirm the distribution by calculating skewness and kurtosis,but I am not doing it here as I will be dropping the parameter.
sns.distplot(USAhousing['Price'])
X=USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Area Population']]
y=USAhousing['Price']
USAhousing.head()
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import zscore
y_stdrd = pd.Series(zscore(y),index=y.index)
X_stdrd = pd.DataFrame(data=zscore(X), index=X.index, columns=X.columns)

X_train,X_test,y_train,y_test=train_test_split(X_stdrd,y_stdrd,test_size=0.4,random_state=101)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)



from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.intercept_
coeff=pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
coeff
prediction=lm.predict(X_test)
plt.scatter(y_test,prediction)
from sklearn import metrics

print('MAE',metrics.mean_absolute_error(y_test,prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
import statsmodels.api as sm
from scipy import stats
X=USAhousing[['Avg. Area Income', 'Avg. Area House Age', 
        'Area Population','Avg. Area Number of Rooms']]
y=USAhousing['Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)


X2=sm.add_constant(X_train)
X2.head()
est=sm.OLS(y_train,X2)
est2=est.fit()
print(est2.summary())
#out of the three models we go with the last mpdel as it has high Adj R2 ,high Fstat.
#but the condition number seems to be very high .So we check for multi collinearity.

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
vifs=[vif(X2.values,i) for i in range(len(X2.columns))]
pd.Series(data=vifs,index=X2.columns)
#As Vif for all parameters is less than 5, we can say that the parameters are not correlated, and the condition number is due to not standardising the data.

from scipy.stats.mstats import zscore
y_stdrd = pd.Series(zscore(y),index=y.index)
X_stdrd = pd.DataFrame(data=zscore(X), index=X.index, columns=X.columns)

X_train,X_test,y_train,y_test=train_test_split(X_stdrd,y_stdrd,test_size=0.4,random_state=101)


X2=sm.add_constant(X_train)
X2.head()
est=sm.OLS(y_train,X2)
est2=est.fit()
print(est2.summary())
#we now see that condition number has decreased to 1.04, indicating the absence of multicollinearity.
#A condition number greater than 100 indicates the presence of multicollinearity.
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.intercept_
coeff=pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
coeff
prediction=lm.predict(X_test)
plt.scatter(y_test,prediction)
print('MAE',metrics.mean_absolute_error(y_test,prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
