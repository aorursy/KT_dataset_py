# Import the necessary libraries 

import pandas as pd

import numpy as np

import numpy as np

import seaborn as sns
# import the dataset

df = pd.read_csv('../input/Advertising.csv')
# see the top five observation

df.head()
df= df[['TV','Radio','Newspaper','Sales']]
df.head()
def summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.mean(), x.median(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75),x.dropna().quantile(0.90),x.dropna().quantile(0.95),x.dropna().quantile(0.99), x.max()],

                    index =[ 'count', 'Null', 'mean', 'median', 'min', 'Q1', "Q5",'Q10', 'Q25', 'Q50', 'Q75', 'Q90','Q95', 'Q99', 'max' ])



df.apply(summary).T
#remove outliers



df['TV']= df['TV'].clip_lower(df.TV.quantile(0.01))

df['TV']= df["TV"].clip_upper(df.TV.quantile(0.99))



df['Radio']= df['Radio'].clip_lower(df.Radio.quantile(0.01))

df['Radio']= df['Radio'].clip_upper(df.Radio.quantile(0.99))



df['Newspaper']= df['Newspaper'].clip_lower(df.Newspaper.quantile(0.01))

df['Newspaper']= df['Newspaper'].clip_upper(df.Newspaper.quantile(0.99))



df['Sales']= df['Sales'].clip_lower(df.Sales.quantile(0.01))

df['Sales']= df['Sales'].clip_upper(df.Sales.quantile(0.99))
df.apply(summary).T
sns.distplot(df.TV)
sns.distplot(df.Newspaper)
sns.distplot(df.Radio)
#Is there a relationship between sales and spend various advertising channels?



#Sales Vs. Newspaper advertisement spends



sns.jointplot(df.Sales, df.TV)
sns.jointplot(df.Sales, df.Newspaper)
sns.jointplot(df.Sales, df.Radio)
# Assumptions of linear regression model



# Dependant variable should follows normal distribution



sns.distplot(df.Sales)

df['ln_Sales']= np.log(df['Sales']+1)

sns.distplot(df.ln_Sales)
#dropping the variable



df_model = df.drop(['Sales', 'ln_Sales'], axis = 1)
df_model.columns
corr= df.corr()

corr
corr.to_csv('marketing.csv')
#check the multicolliniarity

df_model.corr()
sns.heatmap(df.corr(), annot= True)
import statsmodels.formula.api as smf
df.columns
lm = smf.ols('Sales~TV', df).fit()
print(lm.summary())
round(float(lm.rsquared), 2)
### MAKING PREDICTIONS

lmpredict = lm.predict( {'TV': df.TV } )
lmpredict[0:10]
from sklearn import metrics
mse_lm = metrics.mean_squared_error(df.Sales, lmpredict)
rmse_lm = np.sqrt(mse_lm)

rmse_lm
#Get the residuals and plot them

lm.resid[1:10]
sns.distplot(lm.resid)
sns.jointplot(df.Sales, lm.resid)
sns.distplot(lm1.resid)
lm1 = smf.ols('Sales~TV+Radio+Newspaper', df).fit()
print(lm1.summary())


lmpredict1 = lm1.predict({'TV': df.TV, 'Radio': df.Radio, 'Newspaper': df.Newspaper} )

lmpredict1[0:10]
round(float(lm1.rsquared), 2)
mse_lm1 = metrics.mean_squared_error(df.Sales, lmpredict1)
rmse_lm1 = np.sqrt(mse_lm1)

rmse_lm1
lm1.params
lm1.pvalues
lm2 = smf.ols( 'Sales ~ TV + Radio', df ).fit()
lm2.params
lm2.pvalues
predicted2 = lm2.predict( {'TV': df.TV, 'Radio':df.Radio } )
#Get the residuals and plot them

lm2.resid[1:10]
sns.distplot(lm2.resid)
sns.jointplot(df.Sales, lm2.resid)
mse = metrics.mean_squared_error(df.Sales, predicted2)
rmse = np.sqrt(mse)

round(float(rmse), 2)
mae= metrics.mean_absolute_error(df.Sales, predicted2)

round(float(mae), 2)
from sklearn.linear_model import LinearRegression
# Splitting into Train and test data sets

# Typically the model should be built on a training dataset and validated against a test dataset

# Let's split the dataset into 70/30 ratio. 70% belongs to training and 30% belongs to test.



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    df[['TV', 'Radio', 'Newspaper']], df.Sales, test_size = 0.2, random_state = 34)
## Building the model with train set and make predictions on test set



linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)
y_pred
linreg.coef_
from sklearn import metrics

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

round(float(rmse), 3)
y_pred_t= linreg.predict(X_train)
rmse= np.sqrt(metrics.mean_squared_error(y_train, y_pred_t))

round(float(rmse), 3)
round(float(metrics.r2_score(y_test, y_pred)*100), 3)
round(float(metrics.r2_score(y_train, y_pred_t)*100), 3)
list(zip(['TV', 'Radio', 'Newspaper'], list(linreg.coef_)))
resid = y_test-y_pred

sns.distplot(resid)
# To ensure residues are random i.e. normally distributed a Q-Q plot can be used

# Q-Q plot shows if the residuals are plotted along the line.

from scipy import stats

import pylab



stats.probplot( resid, dist="norm", plot=pylab )

pylab.show()
from sklearn.model_selection import cross_val_score
linreg_k = LinearRegression()

cross_val_score(linreg_k, X_train, y_train, scoring = 'r2', cv = 10)

round(np.mean(cross_val_score(linreg_k, X_train, y_train, scoring = 'r2', cv = 10)*100), 2)
from sklearn.feature_selection import f_regression
F_values, P_values = f_regression(X_train, y_train)
F_values
P_values
['%.3f'% p for p in P_values]