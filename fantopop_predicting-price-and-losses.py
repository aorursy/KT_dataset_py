import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/Automobile_data.csv', na_values=['?'])



# Replace '-' in column names with '_'

names = []

for name in data.columns:

    names.append(name.replace('-', '_'))



data.columns = names

data.head()
na_cols = {}

for col in data.columns:

    missed = data.shape[0] - data[col].dropna().shape[0]

    if missed > 0:

        na_cols[col] = missed



na_cols
data[np.any(data[data.columns[2:]].isnull(), axis=1)]
data.dropna(subset=data.columns[2:-1], inplace=True)



# Save indicies for missed values in losses and price.

na_losses = data['normalized_losses'].isnull()

na_price = data['price'].isnull()
plt.figure(figsize=(15, 5))



mean_loss = data.normalized_losses.mean()

dev_loss = np.sqrt(data.normalized_losses.var())



plt.plot(data.index, data['symboling'] * dev_loss + mean_loss)

plt.plot(data.index, data['normalized_losses'])

plt.show()
sns.regplot(x='symboling', y='normalized_losses', data=data)
from scipy.stats import pearsonr



pr, p_value = pearsonr(data.dropna()['symboling'], data.dropna()['normalized_losses'])

print('{:50}{:f}'.format('Correlation', pr))

print('{:50}{:f}'.format('p-value for non-correlation:', p_value))
plt.figure(figsize=(15, 6))



plt.subplot(121)

plt.hist(data['price'].dropna(), bins=12, ec='black', alpha=0.8)

plt.xlabel('price')

plt.ylabel('frequency')

plt.title('Price histogram')



plt.subplot(122)

plt.hist(data['normalized_losses'].dropna(), bins=12, ec='black', alpha=0.8)

plt.xlabel('normalized_losses')

plt.ylabel('frequency')

plt.title('Normalized losses histogram')



plt.show()
plt.figure(figsize=(15, 15))

ax = sns.heatmap(data.corr(), vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)

plt.title('Cross correlation between numerical')

plt.show()
data['mpg'] = (data['city_mpg'] + data['highway_mpg']) / 2
plt.figure(figsize=(10, 5))

sns.countplot(x='make', data=data)

plt.xticks(rotation='vertical')

plt.title('Manufacturers distribution in dataset')

plt.show()
categorical = ['make', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style', 'engine_location',

               'drive_wheels', 'engine_type', 'num_of_cylinders', 'fuel_system']
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

for col, ax in zip(categorical[1:], axs.ravel()):

    sns.countplot(x=col, data=data, ax=ax)
def add_balanced(column, data, values, other_value='other', exclude_values=False):

    if exclude_values:

        res = data[column].apply(lambda x: other_value if x in values else x)

    else:

        res = data[column].apply(lambda x: x if x in values else other_value)

    data[column + '_balanced'] = res
add_balanced(column='fuel_system', data=data, values=['mpfi', '2bbl'])

add_balanced(column='num_of_cylinders', data=data, values=['four'])

add_balanced(column='engine_type', data=data, values=['ohc'])

add_balanced(column='body_style', data=data, values=['sedan', 'hatchback'])

add_balanced(column='drive_wheels', data=data, values=['fwd'])



rare_make = data.groupby(by='make').size().sort_values().index[:3]

add_balanced(column='make', data=data, values=rare_make, exclude_values=True, other_value='rare')
categorical_balanced = ['make_balanced', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style_balanced', 'engine_location',

               'drive_wheels_balanced', 'engine_type_balanced', 'num_of_cylinders_balanced', 'fuel_system_balanced']



fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

for col, ax in zip(categorical_balanced[1:], axs.ravel()):

    sns.countplot(x=col, data=data, ax=ax)
plt.figure(figsize=(10, 5))

sns.countplot(x='make_balanced', data=data)

plt.xticks(rotation='vertical')

plt.title('Manufacturers distribution in dataset')

plt.show()
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

for col, ax in zip(categorical_balanced[1:], axs.ravel()):

    sns.barplot(x=col, y='price', data=data, ax=ax)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

for col, ax in zip(categorical[1:], axs.ravel()):

    sns.barplot(x=col, y='normalized_losses', data=data, ax=ax)
# Numerical columns

numerical = [column for column in data.columns if data[column].dtype != 'O']



# Categorical columns

categorical = [column for column in data.columns if data[column].dtype == 'O']
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, make_scorer
# Numerical features for losses prediction.

xgb_num = ['symboling', 'height', 'horsepower', 'price']



# Target values for losses prediction.

xgb_target = ['normalized_losses']



# Split data to train and prediction sets.

xgb_train = np.logical_and(np.invert(na_losses), np.invert(na_price))

xgb_to_predict = np.logical_and(na_losses, np.invert(na_price))



xgb_scaler = StandardScaler()



X = xgb_scaler.fit_transform(data.loc[xgb_train, xgb_num].values)

y = data.loc[xgb_train, xgb_target].values

P = xgb_scaler.transform(data.loc[xgb_to_predict, xgb_num].values)
xgb = XGBRegressor()

cv = 3

scores = np.sqrt(cross_val_score(xgb, X, y, scoring=make_scorer(mean_squared_error), cv=cv))

print('Average MSE for %d CV-folds: %.2f' % (cv, scores.mean()))
param_grid = {

    'learning_rate': [0.005, 0.01, 0.05, 0.1],

    'n_estimators': [100, 500, 1000],

    'max_depth': [3, 5, 7],

    'min_child_weight': [1, 3, 5]

}



cv = 3

optimizer = GridSearchCV(xgb, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')

optimizer.fit(X, y)



print(optimizer.best_params_)

print('Deviation: %.2f' % np.sqrt(-optimizer.best_score_))
xgb_predictions = pd.DataFrame(index=data.index[xgb_train])

xgb_predictions['normalized_losses'] = data['normalized_losses']

xgb_predictions['predicted_losses'] = optimizer.best_estimator_.predict(X)

xgb_predictions['symboling'] = data['symboling']

sns.lmplot(x='normalized_losses', y='predicted_losses', hue='symboling', data=xgb_predictions, fit_reg=False)
xgb_predictions['residuals'] = xgb_predictions['normalized_losses'] - xgb_predictions['predicted_losses']

plt.scatter(xgb_predictions.index, xgb_predictions['residuals'])

plt.title('Residuals distribution')

plt.ylabel('residuals')
from scipy.stats import ttest_1samp

t, p = ttest_1samp(xgb_predictions['residuals'], 0)

print('Student\'s T-test p-value: %.3f' % p)
data.loc[xgb_to_predict, 'normalized_losses'] = optimizer.best_estimator_.predict(P)

data.head()
from statsmodels.formula.api import ols

from scipy.stats import probplot

from scipy.stats import shapiro

from scipy.stats import ttest_1samp

from statsmodels.stats.api import het_breuschpagan

from sklearn.preprocessing import StandardScaler
numerical = ['height', 'curb_weight', 'engine_size', 'bore', 'stroke', 

             'compression_ratio', 'horsepower', 'peak_rpm', 'mpg']



target = ['price']



formula = target[0] + ' ~ ' + ' + '.join(numerical + categorical_balanced[:5] + categorical_balanced[6:])

print('Baseline formula:\n\n' + formula)
train = np.invert(na_price)



scaler = StandardScaler()

data_scaled = data[target+numerical+categorical].copy()

data_scaled.loc[train, numerical] = scaler.fit_transform(data_scaled.loc[train, numerical])

data_scaled.loc[na_price, numerical] = scaler.transform(data_scaled.loc[na_price, numerical])



model = ols(formula, data_scaled[train])

fitted = model.fit()



print(fitted.summary2())
def residuals_zero_means_test(fitted_model, alpha=0.05):

    t, p = ttest_1samp(fitted_model.resid, 0)

    print('{:<20}{}'.format('Zero means test', 'rejected' if p < alpha else 'not rejected'))

    print('{:<20}{:.8f}'.format('p-value', p))



def residuals_normality_test(fitted_model, alpha=0.05, test=True, plot=True):

    if test:

        W, p = shapiro(fitted_model.resid)

        print('{:<20}{}'.format('Shapiro-Wilk test', 'rejected' if p < alpha else 'not rejected'))

        print('{:<20}{:.8f}'.format('p-value', p))

    

    if plot:

        plt.figure(figsize=(15, 6))

        plt.subplot(121)

        probplot(fitted_model.resid, dist='norm', plot=plt)

        plt.title('Residuals Q-Q Plot')

        

        plt.subplot(122)

        plt.hist(fitted_model.resid, bins=20, ec='black', alpha=0.8)

        plt.title('Residuals distribution')

        plt.show()



def residuals_homoscedasticity_test(fitted_model, alpha=0.05):

    p = het_breuschpagan(fitted_model.resid, fitted_model.model.exog)[1]

    print('{:<20}{}'.format('Breusch-Pagan test', 'rejected' if p < alpha else 'not rejected'))

    print('{:<20}{:.8f}'.format('p-value', p))
residuals_zero_means_test(fitted)

print()



residuals_homoscedasticity_test(fitted)

print()



residuals_normality_test(fitted)
formula_log = 'np.log(price) ~ ' + ' + '.join(numerical + categorical_balanced[:5] + categorical_balanced[6:])

model_log = ols(formula_log, data_scaled[train])

fitted_log = model_log.fit()

# print(fitted_log.summary2())
residuals_zero_means_test(fitted_log)

print()



residuals_homoscedasticity_test(fitted_log)

print()



residuals_normality_test(fitted_log)
model_log_het = ols(formula_log, data_scaled[train])

fitted_log_het = model_log_het.fit(cov_type='HC1')



print(fitted_log_het.summary2())
formula_short = 'np.log(price) ~ height + curb_weight + compression_ratio + horsepower + make_balanced + fuel_type'

model_short = ols(formula_short, data_scaled[train])

fitted_short = model_short.fit(cov_type='HC1')



print(fitted_short.summary2())
print('F=%f, p=%f, k1=%f' % model_log_het.fit().compare_f_test(model_short.fit()))
print('RMSE of the model: %.0f' % np.sqrt(mean_squared_error(np.exp(fitted_short.fittedvalues), data.loc[train, 'price'])))
data.loc[na_price, 'price'] = np.exp(fitted_short.predict(data_scaled[na_price]))

data.loc[na_price, data.columns[2:26]]