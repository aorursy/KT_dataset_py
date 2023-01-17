import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
red = pd.read_csv('../input/winequality-red.csv')
red.shape
red.head()
sns.heatmap(red.isna(),cmap='coolwarm',cbar = False)
#there are no missing values
from sklearn.model_selection import train_test_split
X = red.drop(['quality'],axis=1)
y=red['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
X['constant']= 1
import statsmodels.formula.api as sm
X_opt = X
regressor_ols = sm.OLS(endog = y, exog=X_opt).fit()
regressor_ols.summary()
X_opt1 = X_opt.drop('fixed acidity',axis=1)
regressor_ols1 = sm.OLS(endog = y, exog=X_opt1).fit()
regressor_ols1.summary()
X_opt2 = X_opt1.drop('residual sugar',axis=1)
regressor_ols2 = sm.OLS(endog = y, exog=X_opt2).fit()
regressor_ols2.summary()
model = sm.OLS(y,X_opt2).fit()
model.predict(X_opt2)


