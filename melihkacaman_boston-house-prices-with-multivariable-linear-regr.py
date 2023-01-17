from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



%matplotlib inline
boston_dataset = load_boston() 
type(boston_dataset)
dir(boston_dataset)
print(boston_dataset.DESCR)
type(boston_dataset.data)
boston_dataset.data.shape
boston_dataset.feature_names
# Actual prices in thousands

boston_dataset.target[:5]
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)

data['PRICE'] = boston_dataset.target



data.head()
data.tail()
data.info()
pd.isnull(data).any()
plt.figure(figsize=(10,6))

plt.hist(data['PRICE'], bins=50, ec='black', color='#2196f3')

plt.xlabel('Price in 000s')

plt.ylabel('Nr. of Houses')

plt.show()
plt.figure(figsize=(10,6))

plt.hist(data['RM'], ec='black', color='green', bins=50)

plt.xlabel('Average Number of Rooms')

plt.ylabel('Nr. of Houses')

plt.show()
data['RM'].mean()

data['RM'].median()
plt.figure(figsize=(10,6))

plt.hist(data['RAD'], bins=50, ec='white', color='purple')

plt.xlabel('Accessibility to Highways')

plt.ylabel('Nr. of Houses')

plt.show()



print(data['RAD'].value_counts())

frequency = data['RAD'].value_counts()

plt.figure(figsize=(10,6))

plt.xlabel('Accessibility to Highways')

plt.ylabel('Nr. of Houses')

plt.bar(frequency.index, height=frequency)

plt.show()
data['CHAS'].value_counts() # dummy variable
print('Min price:', data['PRICE'].min())

print('Min price:', data['PRICE'].max())
data.min()
data.max()
data.describe()
data.corr()
"""

- CRIM     per capita crime rate by town

        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.

        - INDUS    proportion of non-retail business acres per town

        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

        - NOX      nitric oxides concentration (parts per 10 million)

        - RM       average number of rooms per dwelling

        - AGE      proportion of owner-occupied units built prior to 1940

        - DIS      weighted distances to five Boston employment centres

        - RAD      index of accessibility to radial highways

        - TAX      full-value property-tax rate per $10,000

        - PTRATIO  pupil-teacher ratio by town

        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

        - LSTAT    % lower status of the population

        - MEDV     Median value of owner-occupied homes in $1000's



"""

print()
mask = np.zeros_like(data.corr())

triangle_indices = np.triu_indices_from(mask)

mask[triangle_indices] = True

plt.figure(figsize=(16,10))

sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size": 10})

sns.set_style('white')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
nox_dis_corr = round(data['NOX'].corr(data['DIS']), 3)



plt.figure(figsize=(9,6))

plt.scatter(data['DIS'], data['NOX'], alpha=0.6, color='purple')

plt.title(f'DIS vs NOX (Correlation {nox_dis_corr})', fontsize=14)

plt.xlabel('DIS - Distance from employment', fontsize=14)

plt.ylabel('NOX - Nitric Oxide Pollution', fontsize=14)

plt.show()
sns.set()

sns.set_context('talk')

sns.set_style('whitegrid')

sns.jointplot(x=data['DIS'], y=data['NOX'], kind='hex', size=7)

plt.show()
sns.set()

sns.set_context('talk')

sns.set_style('whitegrid')

sns.jointplot(x=data['TAX'], y=data['RAD'], size=7, color='darkred', joint_kws={'alpha':0.5})

plt.show()

print(data['TAX'].corr(data['RAD']))
sns.lmplot(x='TAX', y='RAD', data=data, size=7)

plt.show()
sns.lmplot(x='RM', y='PRICE', data=data, size=7)

plt.plot()

print('Corr between the RM and PRICE:',data['RM'].corr(data['PRICE']))
# let's take a loak at the corr amongs all features 

sns.pairplot(data)

plt.show()
%%time



sns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color': 'cyan'}})

plt.show()
prices = data['PRICE']

features = data.drop('PRICE', axis=1)



# split and shuffle our dataset 

X_train, X_test, y_train, y_test = train_test_split(features, prices,test_size= 0.2, random_state=10)

        
regr = LinearRegression().fit(X_train, y_train)



print('Training data r-squared:', regr.score(X_train, y_train))

print('Test data r-squared:', regr.score(X_test, y_test))

print('Intercept', regr.intercept_)

pd.DataFrame(data= regr.coef_, index=X_train.columns, columns=['coef'])
data['PRICE'].skew()
y_log = np.log(data['PRICE'])

y_log.tail()
y_log.skew()
sns.distplot(y_log)

plt.title(f'Log price with skew {y_log.skew()}')

plt.show()



skeww = data['PRICE'].skew()



sns.distplot(data['PRICE'])

plt.title(f'price with skew {skeww}')

plt.show()
sns.lmplot(x='LSTAT', y='PRICE', data=data, size=7, 

           scatter_kws={'alpha': 0.6}, 

           line_kws={'color':'darkred'})

plt.show()
transformed_data = features

transformed_data['LOG_PRICE'] = y_log



sns.lmplot(x='LSTAT', y='LOG_PRICE', data=transformed_data, size=7, 

           scatter_kws={'alpha': 0.6}, line_kws={'color':'cyan'})

plt.show()
prices = np.log(data['PRICE']) # Use log prices

features = data.drop('PRICE', axis=1)



X_train, X_test, y_train, y_test = train_test_split(features, prices, 

                                                    test_size=0.2, random_state=10)



regr = LinearRegression()

regr.fit(X_train, y_train)



print('Training data r-squared:', regr.score(X_train, y_train))

print('Test data r-squared:', regr.score(X_test, y_test))



print('Intercept', regr.intercept_)

pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])
X_incl_const = sm.add_constant(X_train)



model = sm.OLS(y_train, X_incl_const)

results = model.fit() 



pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)})
variance_inflation_factor(exog=X_incl_const.values, exog_idx=1)

#type(X_incl_const)
vifs = []

for i in range(X_incl_const.shape[1]):

    vifs.append(round(variance_inflation_factor(exog=X_incl_const.values, exog_idx=i),2))

pd.DataFrame({'coef' : results.params, 'vif': vifs })
X_incl_const = sm.add_constant(X_train)



model = sm.OLS(y_train, X_incl_const)

results = model.fit() 



org_coef= pd.DataFrame({'coef' : results.params, 'p-value' : round(results.pvalues,2)})



print('BIC is ', results.bic)

print('r-squared is ', results.rsquared)
X_incl_const = sm.add_constant(X_train)

X_incl_const = X_incl_const.drop(['INDUS'], axis=1)



model = sm.OLS(y_train, X_incl_const)

results = model.fit()



coef_minus_indus = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)})



print('BIC is', results.bic)

print('r-squared is', results.rsquared)
# Reduced model #2 excluding INDUS and AGE

X_incl_const = sm.add_constant(X_train)

X_incl_const = X_incl_const.drop(['INDUS', 'AGE'], axis=1)



model = sm.OLS(y_train, X_incl_const)

results = model.fit()



reduced_coef = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)})



print('BIC is', results.bic)

print('r-squared is', results.rsquared)
frames = [org_coef, coef_minus_indus, reduced_coef] 

pd.concat(frames, axis=1, sort=False)
prices = np.log(data['PRICE']) # Use log prices

features = data.drop(['PRICE', 'INDUS', 'AGE'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(features, prices, 

                                                    test_size=0.2, random_state=10)

X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_incl_const)

results = model.fit()



# Residuals

# residuals = y_train - results.fittedvalues

# results.resid



corr = round(y_train.corr(results.fittedvalues), 2)

plt.figure(figsize=(10,6))

plt.scatter(x=y_train, y=results.fittedvalues, c='navy', alpha=0.4)

plt.plot(y_train,y_train, color='cyan')



plt.xlabel('Actual log prices $y _i$', fontsize=14)

plt.ylabel('Prediced log prices $\hat y _i$', fontsize=14)

plt.title(f'Actual vs Predicted log prices: $y _i$ vs $\hat y_i$ (Corr {corr})', fontsize=17)



plt.show()



plt.figure(figsize=(10,6))

plt.scatter(x=np.e**y_train, y=np.e**results.fittedvalues, c='blue', alpha=0.6)

plt.plot(np.e**y_train, np.e**y_train, color='cyan')



plt.xlabel('Actual prices 000s $y _i$', fontsize=14)

plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)

plt.title(f'Actual vs Predicted prices: $y _i$ vs $\hat y_i$ (Corr {corr})', fontsize=17)



plt.show()



# Residuals vs Predicted values

plt.figure(figsize=(10,6))

plt.scatter(x=results.fittedvalues, y=results.resid, c='navy', alpha=0.6)



plt.xlabel('Predicted log prices $\hat y _i$', fontsize=14)

plt.ylabel('Residuals', fontsize=14)

plt.title('Residuals vs Fitted Values', fontsize=17)



plt.show()



# Mean Squared Error & R-Squared

reduced_log_mse = round(results.mse_resid, 3)

reduced_log_rsquared = round(results.rsquared, 3)



print('MSE', reduced_log_mse)

print('RSQUARED',reduced_log_rsquared)
resid_mean = round(results.resid.mean(), 3)

resid_skew = round(results.resid.skew(), 3)

plt.figure(figsize=(10,6))

sns.distplot(results.resid, color='navy')

plt.title(f'Log price model: residuals Skew ({resid_skew}) Mean ({resid_mean})')

plt.show()
# Original model: normal prices & all features

prices = data['PRICE']

features = data.drop(['PRICE'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(features, prices, 

                                                    test_size=0.2, random_state=10)



X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_incl_const)

results = model.fit()



# Graph of Actual vs. Predicted Prices

corr = round(y_train.corr(results.fittedvalues), 2)

plt.scatter(x=y_train, y=results.fittedvalues, c='indigo', alpha=0.6)

plt.plot(y_train, y_train, color='cyan')



plt.xlabel('Actual prices 000s $y _i$', fontsize=14)

plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)

plt.title(f'Actual vs Predicted prices: $y _i$ vs $\hat y_i$ (Corr {corr})', fontsize=17)



plt.show()



# Residuals vs Predicted values

plt.scatter(x=results.fittedvalues, y=results.resid, c='indigo', alpha=0.6)



plt.xlabel('Predicted prices $\hat y _i$', fontsize=14)

plt.ylabel('Residuals', fontsize=14)

plt.title('Residuals vs Fitted Values', fontsize=17)



plt.show()



# Residual Distribution Chart

resid_mean = round(results.resid.mean(), 3)

resid_skew = round(results.resid.skew(), 3)



sns.distplot(results.resid, color='indigo')

plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')

plt.show()



# Mean Squared Error & R-Squared

full_normal_mse = round(results.mse_resid, 3)

full_normal_rsquared = round(results.rsquared, 3)
# Model Omitting Key Features using log prices

prices = np.log(data['PRICE'])

features = data.drop(['PRICE', 'INDUS', 'AGE', 'LSTAT', 'RM', 'NOX', 'CRIM'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(features, prices, 

                                                    test_size=0.2, random_state=10)



X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_incl_const)

results = model.fit()



# Graph of Actual vs. Predicted Prices

corr = round(y_train.corr(results.fittedvalues), 2)

plt.scatter(x=y_train, y=results.fittedvalues, c='#e74c3c', alpha=0.6)

plt.plot(y_train, y_train, color='cyan')



plt.xlabel('Actual log prices $y _i$', fontsize=14)

plt.ylabel('Predicted log prices $\hat y _i$', fontsize=14)

plt.title(f'Actual vs Predicted prices with omitted variables: $y _i$ vs $\hat y_i$ (Corr {corr})', fontsize=17)



plt.show()



# Residuals vs Predicted values

plt.scatter(x=results.fittedvalues, y=results.resid, c='#e74c3c', alpha=0.6)



plt.xlabel('Predicted prices $\hat y _i$', fontsize=14)

plt.ylabel('Residuals', fontsize=14)

plt.title('Residuals vs Fitted Values', fontsize=17)



plt.show()



# Mean Squared Error & R-Squared

omitted_var_mse = round(results.mse_resid, 3)

omitted_var_rsquared = round(results.rsquared, 3)
pd.DataFrame({'R-Squared': [reduced_log_rsquared, full_normal_rsquared, omitted_var_rsquared],

             'MSE': [reduced_log_mse, full_normal_mse, omitted_var_mse], 

             'RMSE': np.sqrt([reduced_log_mse, full_normal_mse, omitted_var_mse])}, 

            index=['Reduced Log Model', 'Full Normal Price Model', 'Omitted Var Model'])
# The best model in our models is Reduced Log Models 







log_prices = np.log(boston_dataset.target)

target = pd.DataFrame(log_prices, columns=['PRICE'])



regr = LinearRegression().fit(features, target) 

fitted_vals = regr.predict(features)

fitted_vals[0:5]