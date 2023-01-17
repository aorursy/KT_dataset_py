import pandas as pd
import numpy as np
data = pd.read_csv("../input/insurance/insurance.csv")
data.describe()
data.info()
data.head()
data
data.region.value_counts()
#null values
data.isna().sum()
data.nunique()
data.region.unique()
#dependent,independent variables
y = data.charges
x= data.drop(['charges'], axis=1)
y.head()
x.head()
#encodeyes no to 1,0
x.smoker = x.smoker.eq('yes').mul(1)
x.head()
x.sex = x.sex.eq('female').mul(1)
x.head()
#one_hot encoding
cat_var = list(x.select_dtypes(include=["object"]))
cat_var
x = pd.get_dummies(x, columns=cat_var, drop_first = True)
x.head()
#check outliers
import seaborn as sns
columns = list(x)
columns
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

sns.set(rc={'figure.figsize':(20,10)})

for j in range(1, 9):
    plt.subplot(2, 4, j)
    sns.boxplot(x =  x[columns[j-1]])
   
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(x))
print(z)
threshold = 3
print(np.where(z > 3))

#first array is rows, second is column numbers
#OLS statsmodel regression
from statsmodels.stats import diagnostic

import statsmodels.api as sm
model = sm.OLS(y, x).fit()
#predictions = model.predict(x_test) 

print_model = model.summary()
print(print_model)
heter#inferences
'''
1. R square : model explains 87.4% variance of charges

2. Prob(F statistic) = 0.00, model is significant. F test is basically a test for the model significance. 
   Null hypothesis: No linear relationship between all the independent variales and the Dependent variable
   
3. AIC : basic idea of AIC is to penalize the inclusion of additional variables to a model. It adds a penalty that increases
   the error when including additional terms. The lower the AIC, the better the model.
   BIC:  Variant of AIC with a stronger penalty for including additional variables to the model.
   BIC tends to penalise the complex models more hence, A downside of BIC is that for smaller, less representative training 
   datasets, it is more likely to choose models that are too simple. In general. it doesnt choose the more complex models
   probability that BIC will select the true model increases with the size of the training dataset. This cannot be said for the 
   AIC score.
   
4. Degrees of Freedom of residuals: Degrees of freedom is the number of values in the final calculation of a statistic
   that are free to vary.
   df(residual) = n-k-1 (1 for the intercept)

5. Degress of Freedom of Model: No. of independent variables 

6. Std. error of variables: Individual Estimation error by variables.

7. P values of variables: all under 0.05 except Children but very close so can be avoided for now, for accuracy purposes 
   can be removed and checked later.
   T test is done for individual variable significance.
   Null hypothesis: No linear relationship between the Independent Variable and dependent variable.
   T statistic = Coefficient/Std. error
   
8. Omnibus: Test for Normality.
   Null hypothesis : Data is Normally Distributed
   P value can be used to understand if to accept or rejeheterct (Prob omnibus)

9. Skew: Measure for Noramlity. Amount and direction of skew (departure from horizontal symmetry)
   Skewness of Normal Distribution = 0 

10. Kurtosis: Measure for Normality. Height and sharpness of Central peak.
    Kurtosis of Normal Distribution = 3

11. Durbin Watson: Measure of Autocorrelation.
    Null Hypothesis: No first order correlation.
    
    The Durbin-Watson statistic will always have a value between 0 and 4.
    A value of 2.0 means that there is no autocorrelation detected in the sample.
    Values from 0 to less than 2 indicate positive autocorrelation 
    and values from 2 to 4 indicate negative autocorrelation.
    
    A rule of thumb is that test statistic values in the range of 1.5 to 2.5 are relatively normal. 
    
    Any value outside this range could be a cause for concern. The Durbin–Watson statistic, while displayed by many 
    regression analysis programs, is not applicable in certain situations. For instance, when lagged dependent variables 
    are included in the explanatory variables, then it is inappropriate to use this test.

12. Jarque Bera: Test for Normality.
    Null hypothesis : Data is Normally Distributed
    P value can be used to understand if to accept or reject (Prob Jargue Bera)

13. Standard Error of Regresion : The standard error of the regression model represents the average distance that the 
    observed values fall from the regression line. Conveniently, it tells you how wrong the regression model is on average 
    using the units of the response variable. Smaller values are better because it indicates that the observations are closer 
    to the fitted line.
    
    Unlike R-squared, you can use the standard error of the regression to assess the precision of the predictions. 
    Approximately 95% of the observations should fall within plus/minus 2*standard error of the regression from the 
    regression line, which is also a quick approximation of a 95% prediction interval. If want to use a regression model 
    to make predictions, assessing the standard error of the regression might be more important than assessing R-squared.

'''
#train_test splitting 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)

#Accuracy
from sklearn.metrics import r2_score
r2_test = r2_score(y_test, y_pred)  

from sklearn.metrics import mean_squared_error
from math import sqrt
mse_test = sqrt(mean_squared_error(y_test, y_pred))
print(r2_test)
print(mse_test)
y_predt = regressor.predict(x_train)
r2_train = r2_score(y_train, y_predt) 
mse_train = sqrt(mean_squared_error(y_train, y_predt))
print(r2_train)
print(mse_train)
#find the residuals
residual = y_test - y_pred
#why is sklearn and statsmodel R square different?
#why is test accuracy better than train accuracy?
#linearity
#plot predicted vs actual
%matplotlib inline
import matplotlib.pyplot as plt
plt.scatter(y_pred,y_test)

#sort of linear
#linearity with different independent variables to dependent variable  
sns.set(rc={'figure.figsize':(20,10)})

for j in range(1, 9):
    plt.subplot(2, 4, j)
    plt.scatter(x[columns[j-1]],y)

#Multicollinearity - VIF
#VIF>10, REMOVE. - high VIF indicates high multicollinearity
#VIF>10 indicates heavy multicollinearity so those factors should be removed, <5 is good

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
pd.DataFrame({'vif':vif[0:]}, index = x_train.columns).T

#based on results - BMI should be definitely removed
#Normality of Residual
#1 Distribution
#2 PP plot or a QQ plot. Whats the difference? How to interpret?
%matplotlib inline
import seaborn as sns
sns.distplot(residual)
#PP plot
import scipy as sp
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)

#QQ plot
import statsmodels.api as sm 
import pylab as py 
sm.qqplot(residual, fit=True, line = '45') 
py.show() 
np.mean(residual)
#should be close to zero or nearly zero in case of normal distribution
#homoskedasticity., there shouldnt be a pattern as such, increasing or decreasing or anything like that 
#what should be there? 
#what does it mean?

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,2.5))
ax.scatter(y_pred, residual)
#residuals vs predicted plot
# look for independence assumption. If the residuals are distributed uniformly randomly around the zero x-axes and 
#do not form specific clusters, then the assumption holds true.
plt.scatter(y_pred, residual)

#autocorrelation
#except the first line, all other lines should be inside the blue or the significance area
#outside the blue area would mean there is some significant autocorrelation

import statsmodels.api as sm
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(residual, lags=40 , alpha=0.05)
acf.show()
data.corr()['charges'].sort_values()
#correlation matrix
import seaborn as sns
x_cor = x.corr()

import matplotlib.pyplot as plt
plt.subplots(figsize=(10,10))
sns.set(font_scale=1)
sns.heatmap(x_cor, linewidths=3,fmt='.2f', annot=True)
'''
Assumption Inferences

Linearity: Sort of
Autocorrelation: No autocorrelation as such 
Heteroskedasticity: no certain pattern as such, but weird, idk
Normality: Nearly normal, but not completely
Multicollinearity: Should remove BMI

'''

#updated data after linear regression assumptions
x_asmp = x.drop(['bmi'], axis=1)
#reapply linear regression and see if any progress
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(x_asmp, y, test_size = 0.25, random_state = 0)

#Linear Regression Model
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_tr, y_tr)

# Predict
y_pr = regress.predict(x_te)

#Accuracy
from sklearn.metrics import r2_score
r2_te = r2_score(y_te, y_pr)  

from sklearn.metrics import mean_squared_error
from math import sqrt
mse_te = sqrt(mean_squared_error(y_te, y_pr))
print(r2_te)
print(mse_te)

#useless
#apply other models
#polynomial regression

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# Predicting a new result with Polynomial Regression
poly_pred = lin_reg_2.predict(poly_reg.fit_transform(x_test))

r2_poly = r2_score(y_test, poly_pred)  

print(r2_poly)

#Kaggle polynomial features
#why? and how?

X = x.drop(['region_northwest','region_southeast','region_southwest'], axis = 1)
Y = data.charges

from sklearn.preprocessing import PolynomialFeatures

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

r2_quad = r2_score(Y_test, Y_test_pred)  

print(r2_quad)
#decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)
                          
r2_dt = r2_score(y_test, y_pred)  
mse_dt = sqrt(mean_squared_error(y_test, y_pred))
                           
print(r2_dt)
print(mse_dt)
#random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10000, random_state = 0, max_depth=5)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)

r2_rf = r2_score(y_test, y_pred)  
mse_rf = sqrt(mean_squared_error(y_test, y_pred))
                           
print(r2_rf)
print(mse_rf)
#SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)

r2_svr = r2_score(y_test, y_pred)  
mse_svr = sqrt(mean_squared_error(y_test, y_pred))
                           
print(r2_svr)
print(mse_svr)
#bagging, boosting and its difference
#gradient boost
from sklearn import ensemble

reg = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 4,learning_rate = 0.01)
reg.fit(x_train, y_train)

y_pred_gbm = reg.predict(x_test)

r2_gbm = r2_score(y_test, y_pred_gbm)  
mse_gbm = sqrt(mean_squared_error(y_test, y_pred_gbm))

print(r2_gbm)
print(mse_gbm)
                           
#Feature Scaling
'''
Real world dataset contains features that highly vary in magnitudes, units, and range. Normalisation should be performed 
when the scale of a feature is irrelevant or misleading and not should Normalise when the scale is meaningful.

The algorithms which use Euclidean Distance measure are sensitive to Magnitudes. Here feature scaling helps to weigh all 
the features equally.

Formally, If a feature in the dataset is big in scale compared to others then in algorithms where Euclidean distance is 
measured this big scaled feature becomes dominating and needs to be normalized.

Examples of Algorithms where Feature Scaling matters
1. K-Means uses the Euclidean distance measure here feature scaling matters.
2. K-Nearest-Neighbours also require feature scaling.
3. Principal Component Analysis (PCA): Tries to get the feature with maximum variance, here too feature scaling is required.
4. Gradient Descent: Calculation speed increase as Theta calculation becomes faster after feature scaling.

Note: Naive Bayes, Linear Discriminant Analysis, and Tree-Based models are not affected by feature scaling.
In Short, any Algorithm which is Not Distance based is Not affected by Feature Scaling.


Something like millions and thousands should be taken care of. 


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

'''
#normalise all independent variables and try all models. Does result improve?
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_norm = scaler.fit_transform(x)
x_norm = pd.DataFrame(x_norm)
x_norm.columns = x.columns
x_norm.head()
#Apply linear regression
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x_norm, y, test_size = 0.25, random_state = 0)

#Linear Regression Model
from sklearn.linear_model import LinearRegression
regres = LinearRegression()
regres.fit(xtr, ytr)

# Predict
ypr = regres.predict(xte)

#Accuracy
from sklearn.metrics import r2_score
r2te = r2_score(yte, ypr)  

from sklearn.metrics import mean_squared_error
from math import sqrt
msete = sqrt(mean_squared_error(yte, ypr))
print(r2te)
print(msete)

#not much difference
#support vector on scaled data
regressor = SVR(kernel = 'rbf')
regressor.fit(xtr, ytr)

# Predicting a new result
y_pred = regressor.predict(xte)

r2_svr = r2_score(yte, ypr)  
mse_svr = sqrt(mean_squared_error(yte, ypr))
                           
print(r2_svr)
print(mse_svr)
