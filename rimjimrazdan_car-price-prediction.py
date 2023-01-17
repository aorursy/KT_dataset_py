import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
cars = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
cars.head()
cars.shape
cars.describe()
cars.info()
#Car_Name

cars.Car_Name.value_counts()
cars.drop(['Car_Name'],axis=1,inplace = True)
cars.head()
# Year

plt.figure(figsize = (15,5))
sns.boxplot(data=cars)
plt.show()
# from the boxplot we can see that kms_Driven has outliers
q1 = cars['Kms_Driven'].quantile(0.25)
q3 = cars['Kms_Driven'].quantile(0.75)
iqr = q3-q1

UL = q3 + (1.5 * iqr)
LL = q1 - (1.5 * iqr)
print(iqr,UL,LL)
cars[cars['Kms_Driven']>UL]
cars[cars['Kms_Driven']>UL].count()['Kms_Driven']
#outlier removal from Kms_Driven

df = cars[cars['Kms_Driven']<UL]
cars=df
cars
sns.distplot(df['Year'])
# The Years variable is left skewed
sns.distplot(df['Selling_Price'])
plt.show()
# the selling price is right skewed
sns.distplot(df['Present_Price'])
plt.show()
#the present_price is right skewed
sns.distplot(df['Kms_Driven'])
plt.show()
#The Kms_Driven are almost normally distributed after removing the outliers, the max values lie between 20000 to 50000
#kms
sns.countplot(cars['Fuel_Type'])
plt.show()
# From this bar plot we can see that there are three categories of Fuel_Type
#Petrol Fuel_type is the maximum in number and CNG cars are the least
sns.countplot(cars['Seller_Type'])
plt.show()
# There are two types of sellers : Individual and Dealer
# The seller_type dealer is greater than the individual seller_type
sns.countplot(cars['Transmission'])
plt.show()
# The Transmission feature has 2 categories
#Manual and Automatic
sns.countplot(cars['Owner'])
plt.show()
# The cars having 0 previous owners is more than the cars having one previous owner.
fig, (ax1, ax2,ax3) = plt.subplots(1,3,figsize = (15,5))

#scatter plot 1
ax1.scatter(x=cars['Year'],y= cars['Selling_Price'])
ax1.set_title('Years v/s Selling_Price')

#scatter plot 2
ax2.scatter(x=cars['Present_Price'], y=cars['Selling_Price']) 
ax2.set_title('Present_Price v/s Selling_Price')

#scatter plot 3
ax3.scatter(x=cars['Kms_Driven'],y=cars['Selling_Price'])
ax3.set_title('Kms_Driven v/s Selling_Price')

plt.draw()  
fig,axes = plt.subplots(2,2,figsize=(20,12))

sns.boxplot(x=cars.Fuel_Type,y=cars.Selling_Price,ax=axes[0][0])
axes[0][0].set_title('Fuel_Type v/s Selling_Price')

sns.boxplot(x=cars.Transmission,y=cars.Selling_Price,ax=axes[0][1])
axes[0][1].set_title('Transmission v/s Selling_Price')

sns.boxplot(x=cars.Owner,y=cars.Selling_Price,ax=axes[1][0])
axes[1][0].set_title('Owner v/s Selling_Price')

sns.boxplot(x=cars.Seller_Type,y=cars.Selling_Price,ax=axes[1][1])
axes[1][1].set_title('Seller_Type v/s Selling_Price')
sns.lmplot(x='Kms_Driven',y='Selling_Price',data=cars,fit_reg=False,col='Transmission',row='Seller_Type')   
plt.show()
sns.lmplot(x='Present_Price',y='Selling_Price',data=cars,fit_reg=False,col='Transmission',row='Seller_Type',hue='Fuel_Type')   
plt.show()
#Fuel_Type

cars.Fuel_Type.value_counts()
cars.Seller_Type.value_counts()
cars.Transmission.value_counts()
cars = pd.get_dummies(cars,columns=['Fuel_Type','Seller_Type','Transmission'],drop_first=True)
cars.info()
cars.shape
cars.head()
#Heatmap to show the correlation between various variables of the dataset

plt.figure(figsize=(10, 8))
cor = cars.corr()
ax = sns.heatmap(cor,annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
y = cars['Selling_Price']
X = cars.drop(['Selling_Price'],axis=1)
#Splitting the data into train and test

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.30 , random_state = 1)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
#standardization of the data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=X.columns)

X_test=sc.fit_transform(X_test)
X_test=pd.DataFrame(X_test,columns=X.columns)
#Building model using sklearn(Gradient Descent)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train) # training the algorithm

# Getting the coefficients and intercept

print('coefficients:\n', lin_reg.coef_)
print('\n intercept:', lin_reg.intercept_)
#coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])  
#print(coeff_df)

#Now predicting on the test data

y_pred = lin_reg.predict(X_test)
# compare the actual output values for X_test with the predicted values

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.reset_index(inplace=True,drop=True)
df
#Showing the difference between the actual and predicted value

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#Calculating the accuracy 

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('r2_score:', metrics.r2_score(y_test,y_pred))

#or
#print('rsquare_Train', lin_reg.score(X_train, y_train))
#print('rsquare_Test', lin_reg.score(X_test, y_test)) 
# Building a linear Regression model using statsmodels (OLS)
import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

y = cars['Selling_Price']
X = cars.drop(['Selling_Price'],axis=1)
X_constant = sm.add_constant(X)
model = sm.OLS(y, X_constant).fit()
predictions = model.predict(X_constant)
print(model.summary())
# 1. Durbin Watson Test

#Ho: Linear Regression Residuals are not correlated
#H1: Errors are correlated.

from statsmodels.stats.api import durbin_watson
durbin_watson(model.resid)
#2. time series analysis graph 

import statsmodels.tsa.api as smt #tsa time series anlaysis

acf = smt.graphics.plot_acf(model.resid, lags=40 , alpha=0.05) #model.resid comes from statsmodel 
acf.show()

# from this graph we dont see any pattern in the residuals so this shows no autocorrelation
#1. Jarque berua test

from scipy import stats
print(stats.jarque_bera(model.resid))

#ho : the data is normally distributed
#h1: the errors are not normally distributed
#2. Histogram

import seaborn as sns

sns.distplot(model.resid)
#3. QQ plot

import pylab

stats.probplot(model.resid, dist = 'norm', plot = pylab)
plt.show()
#4. shapiro wilk test

# Ho: The Data / Errors are Normal in Nature
# H1: The Data is not Normal

from scipy.stats import shapiro

teststats, pvalue = shapiro(model.resid)
print(pvalue)
print("reject the null ho")
#1. Visual representation

%matplotlib inline
%config InlineBackend.figure_format ='retina'
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
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')
    
    #LOWESS (Locally Weighted Scatterplot Smoothing) is a popular tool used in regression analysis that creates a smooth line 
    #through a timeplot or scatter plot to help you to see relationship between variables and foresee trends.

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    
linearity_test(model, y)
#2. Rainbow test

import statsmodels.api as sm
sm.stats.linear_rainbow(res=model, frac=0.5)
# frac : we are not checking the whole data we are just checking the fraction of it
from statsmodels.stats.api import het_goldfeldquandt
from statsmodels.compat import lzip

#1. Goldfeld Quandt Test:

# Ho: The residuals are not heteroscedastic / same variance / homoscedastic
# H1: The residuals are Heteroscedastic / unequal variance

name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(model.resid, model.model.exog)
lzip(name, test)

#exog - x varibles and endog - y variables
#2. Visual representation

fitted_vals = model.predict()
resids = model.resid
resids_standardized = model.get_influence().resid_studentized_internal
fig, ax = plt.subplots(1,2,figsize=(20,12))

sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Residuals vs Fitted', fontsize=16)
ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Scale-Location', fontsize=16)
ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]
df = pd.DataFrame({'vif': vif[1:]}, index=X.columns)
df
df[df.vif > 5].index
## After removing multicollinear feature 'Fuel_Type_Diesel'....cars1
cars1 = cars
cars1.drop(['Fuel_Type_Diesel'],axis=1,inplace=True)

X_vif = cars1.drop(['Selling_Price'],axis=1)
y_vif = cars1['Selling_Price']
from sklearn.linear_model import LinearRegression

lin_reg_vif = LinearRegression()
lin_reg_vif.fit(X, y)

print(f'Coefficients: {lin_reg_vif.coef_}')
print(f'Intercept: {lin_reg_vif.intercept_}')
print(f'R^2 score: {lin_reg_vif.score(X, y)}')
## After removing multicollinear feature 'Fuel_Type_Diesel'

import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

X = cars1.drop(['Selling_Price'],axis=1)
y = cars1['Selling_Price']

X_constant = sm.add_constant(X)
model = sm.OLS(y,X_constant).fit()
predictions = model.predict(X_constant)
model.summary()
vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=X.columns).T
#After checking the assumptions found that Normality criteria not met

# we will apply transformation on the data to make the data meet the assumption
# Residual plot

sns.set(style = 'whitegrid')

cars1['predictions'] = model.predict(X_constant)
residuals = model.resid

ax = sns.residplot(cars1.predictions, residuals, lowess = True, color = 'g')
ax.set(xlabel = 'Fitted value', ylabel = 'Residuals', title = 'Residual vs Fitted Plot \n')
plt.show()
## for sqrt(X)

final_df = cars1.transform(lambda x: x**0.5)
final_df.head()
X_final = final_df.drop(['Selling_Price','predictions'],axis=1)
y_final = final_df.Selling_Price
X_constant_final = sm.add_constant(X_final)
model_final = sm.OLS(y_final, X_constant_final).fit()
predictions_final = model_final.predict(X_constant_final)
model_final.summary()
#After transformating the data the accuracy/R2 score for the model improved.

#We can look further into the different regularization techniques with different values of alpha and build models

#The best R2 score that this model is giving is using these parameters
from sklearn.linear_model import RidgeCV,Ridge

alphas = 10**np.linspace(10,-2,100)*0.5

ridgecv = RidgeCV(alphas = alphas,normalize = True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_
rr = Ridge(alpha = ridgecv.alpha_, normalize = True)
rr.fit(X_train, y_train)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rr.predict(X_test))))

print('r2_score:', metrics.r2_score(y_test, rr.predict(X_test)))
from sklearn.linear_model import LassoCV,Lasso

lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lasso.predict(X_test))))

print('r2_score:', metrics.r2_score(y_test, lasso.predict(X_test)))
# Plot the coefficients
plt.figure(figsize=(8, 5))

colnames = X_train.columns

plt.plot(range(len(colnames)), lasso.coef_, linestyle='none',marker='*',markersize=5,color='red')
plt.xticks(range(len(colnames)), colnames.values, rotation=60) 
plt.margins(0.02)
plt.show()
# Let's perform a cross-validation to find the best combination of alpha and l1_ratio
from sklearn.linear_model import ElasticNetCV, ElasticNet

# how much importance should be given to l1 reguralization
cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True, 
                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, 
                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
cv_model.fit(X_train, y_train)
print('Optimal alpha: %.8f'%cv_model.alpha_)
#The amount of penalization chosen by cross validation

print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
#The compromise between l1 and l2 penalization chosen by cross validation

print('Number of iterations %d'%cv_model.n_iter_)
#number of iterations run by the coordinate descent solver to reach the specified tolerance for the optimal alpha.
# train model with best parameters from CV
elastic = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha = cv_model.alpha_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)
elastic.fit(X_train, y_train)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, elastic.predict(X_test))))

print('r2_score:', metrics.r2_score(y_test, elastic.predict(X_test)))
