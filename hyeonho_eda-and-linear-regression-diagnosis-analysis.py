import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.linear_model import LinearRegression

import statsmodels



%matplotlib inline

import cloudpickle
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.columns, test.columns
print(set(train.columns)-set(test.columns))

print(set(test.columns)-set(train.columns))
train.head()
test.head()
import sklearn.ensemble
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    types = [str(data[i].dtype) for i in data.columns]

    

    df = pd.DataFrame({'Total':total, 'Precent':percent, 'Types':types})

    

    return(sp.transpose(df))
%%time

missing_data(train)
%%time

missing_data(test)
%%time

train.describe()
%%time

test.describe()
sns.distplot(train['price']).set_title('price probability density function')

plt.xticks(rotation=45)

plt.show()
train['price'].skew(), train['price'].kurt()
sns.distplot(sp.log1p(train['price'])).set_title('price pdf with using log scale')

plt.show()
sp.log1p(train['price']).skew(), sp.log1p(train['price']).kurt()
f, axes = plt.subplots(1, 2, figsize=(12, 6))

sp.stats.probplot(train['price'], plot=axes[0])

sp.stats.probplot(sp.log1p(train['price']), plot=axes[1])

plt.show()
train['price'] = sp.log1p(train['price'])
features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

nrow=2; ncol=4

f, axes = plt.subplots(nrow, ncol, figsize=(25,15))

for idx, feature in enumerate(features):

    plt.subplot(nrow, ncol,idx+1)

    sns.distplot(train[feature]).set_title(str(feature) + ' probability density function')

    plt.xlabel(feature, fontsize=15)

plt.show()
features = ['date', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode']

nrow=4; ncol=3

f, axes = plt.subplots(nrow, ncol, figsize=(25,20))

for idx, feature in enumerate(features):

    plt.subplot(nrow, ncol,idx+1)

    sns.countplot(data=train, y=feature)

    plt.ylabel(feature, fontsize=15)

    plt.xticks(rotation=90)

plt.show()
features = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

pp = sns.pairplot(train[features], 

                  diag_kws=dict(shade=True), # "diag" adjusts/tunes the diagonal plots

                  diag_kind="kde", # use "kde" for diagonal plots

                  kind="reg") # <== 😀 linear regression to the scatter plots



fig = pp.fig 

fig.subplots_adjust(top=0.93, wspace=0.3)

fig.suptitle('Continuous Variable Pairwise Plots', fontsize=15, fontweight='bold')

plt.show()
corr = train[features].corr()

fig, (ax) = plt.subplots(1, 1, figsize=(10,6))



hm = sns.heatmap(corr, 

                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.

                 cmap="coolwarm", # Color Map.

                 square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.

                 annot=True, 

                 fmt='.2f',       # String formatting code to use when adding annotations.

                 annot_kws={"size": 12},

                 linewidths=.05)



fig.subplots_adjust(top=0.93)

fig.suptitle('Continuous Variable Correlation Heatmap', 

              fontsize=14, 

              fontweight='bold')

plt.show()
train['date'] = train['date'].apply(lambda x: str(x)[:6])

train['yr_built'] = train['yr_built'].apply(lambda x: str(x)[:3])

train['yr_renovated'] = train['yr_renovated'].apply(lambda x: str(x)[:3])

train['zipcode'] = train['zipcode'].apply(lambda x: str(x)[:4])
features = ['date', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode']

nrow=4; ncol=3

f, axes = plt.subplots(nrow, ncol, figsize=(25, 20))

for idx, feature in enumerate(features):

    plt.subplot(nrow, ncol,idx+1)

    sns.boxplot(data=train, x=feature, y='price') # violinplot과 비슷

    ax.set_xlabel(feature, size=12, alpha=0.8)

    ax.set_ylabel('Price', size=12, alpha=0.8)

    plt.xticks(rotation=45)

plt.show()
corr = train.drop(columns='id').corr(method='spearman')

fig, axes = plt.subplots(1, 1, figsize=(20,12))



hm = sns.heatmap(corr, 

                 ax=axes,           # Axes in which to draw the plot, otherwise use the currently-active Axes.

                 cmap="coolwarm", # Color Map.

                 square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.

                 annot=True, 

                 fmt='.2f',       # String formatting code to use when adding annotations.

                 annot_kws={"size": 12},

                 linewidths=.05)



fig.subplots_adjust(top=0.93)

fig.suptitle('Spearman Correlation Heatmap', 

              fontsize=14, 

              fontweight='bold')

plt.show()
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train['price'] = sp.log1p(train['price'])
%%time

features = train.columns

unique_max_train = []

unique_max_test = []

for feature in features:

    if feature == 'price':

        values = train[feature].value_counts()

        unique_max_train.append([feature, values.max(), values.idxmax()])

    else:

        values = train[feature].value_counts()

        unique_max_train.append([feature, values.max(), values.idxmax()])

        values = test[feature].value_counts()

        unique_max_test.append([feature, values.max(), values.idxmax()])
sp.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).sort_values(by = 'Max duplicates', ascending=False))
sp.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value'])).sort_values(by = 'Max duplicates', ascending=False))
for df in [train, test]:

    df['date'] = df['date'].apply(lambda x: str(x)[:6])

    df['yr_built'] = df['yr_built'].apply(lambda x: str(x)[:3])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: str(x)[:3])

    df['zipcode'] = df['zipcode'].apply(lambda x: str(x)[:4])
features = ['date', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated', 'zipcode']

for feature in features:

    train_dummy = pd.get_dummies(train[feature], columns=features)

    train_dummy.columns = [feature +'_' + str(i) for i in train_dummy.columns]

    train = pd.concat([train, train_dummy], axis=1)

    train = train.drop(columns=feature)

    test_dummy = pd.get_dummies(test[feature], columns=features)

    test_dummy.columns = [feature +'_' + str(i) for i in test_dummy.columns]

    test = pd.concat([test, test_dummy], axis=1)

    test = test.drop(columns=feature)

train = train.drop(columns='id')

test = test.drop(columns='id')
set(train.columns) - set(test.columns), set(test.columns) - set(train.columns)
train = train.drop(columns=['bathrooms_7.5', 'bathrooms_7.75', 'bathrooms_8.0', 'grade_1'])

test = test.drop(columns=['bathrooms_6.5', 'bedrooms_11', 'bedrooms_33', 'yr_renovated_193'])

set(train.columns) - set(test.columns), set(test.columns) - set(train.columns)
price = train['price']

train = train.drop(columns='price')
import statsmodels.api as sm
X = sm.add_constant(train)

reg = sm.OLS(price, X.astype(float)).fit()
reg.summary()
result_pvalue = train.columns[reg.pvalues[1:]<0.00001]
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(train[result_pvalue].values, i) for i in range(train[result_pvalue].shape[1])]

vif["features"] = result_pvalue

vif = vif.sort_values("VIF Factor").reset_index(drop=True)
vif
result_pvalue = result_pvalue[~sp.isinf(vif['VIF Factor'])]
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(train[result_pvalue].values, i) for i in range(train[result_pvalue].shape[1])]

vif["features"] = result_pvalue

vif = vif.sort_values("VIF Factor").reset_index(drop=True)
vif
X = sm.add_constant(train[result_pvalue])

reg2 = sm.OLS(price, X.astype(float)).fit()
reg2.summary()
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']

test = sm.stats.jarque_bera(reg2.resid)

statsmodels.compat.lzip(name, test)
fig, axes = plt.subplots(figsize=(8,6))

statsmodels.graphics.gofplots.qqplot(reg2.resid, ax=axes, line='r')

plt.show()
fig, ax = plt.subplots(figsize=(8,6))

statsmodels.graphics.regressionplots.plot_leverage_resid2(reg2, ax = ax)

plt.show()
name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']

test = sm.stats.het_breuschpagan(reg2.resid, reg2.model.exog)

statsmodels.compat.lzip(name, test)
X = sm.add_constant(train[['sqft_living','long']])

reg3 = sm.OLS(price, X.astype(float)).fit()
name = ['t value', 'p value']

test = sm.stats.linear_harvey_collier(reg3)

statsmodels.compat.lzip(name, test)
# fitted values (need a constant term for intercept)

model_fitted_y = reg2.fittedvalues



# model residuals

model_residuals = reg2.resid



# normalized residuals

model_norm_residuals = reg2.get_influence().resid_studentized_internal



# absolute squared normalized residuals

model_norm_residuals_abs_sqrt = sp.sqrt(sp.absolute(model_norm_residuals))



# absolute residuals

model_abs_resid = sp.absolute(model_residuals)



# leverage, from statsmodels internals

model_leverage = reg2.get_influence().hat_matrix_diag



# cook's distance, from statsmodels internals

model_cooks = reg2.get_influence().cooks_distance[0]
plot_lm_1 = plt.figure(1)

plot_lm_1.set_figheight(8)

plot_lm_1.set_figwidth(12)



plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'price', pd.concat([train, price], axis=1),

                          lowess=True, 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_1.axes[0].set_title('Residuals vs Fitted')

plot_lm_1.axes[0].set_xlabel('Fitted values')

plot_lm_1.axes[0].set_ylabel('Residuals')



# annotations

abs_resid = model_abs_resid.sort_values(ascending=False)

abs_resid_top_3 = abs_resid[:3]



for i in abs_resid_top_3.index:

    plot_lm_1.axes[0].annotate(i, 

                               xy=(model_fitted_y[i], 

                                   model_residuals[i]));
QQ = sm.ProbPlot(model_norm_residuals)

plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)



plot_lm_2.set_figheight(8)

plot_lm_2.set_figwidth(12)



plot_lm_2.axes[0].set_title('Normal Q-Q')

plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')

plot_lm_2.axes[0].set_ylabel('Standardized Residuals');



# annotations

abs_norm_resid = sp.flip(sp.argsort(sp.absolute(model_norm_residuals)), 0)

abs_norm_resid_top_3 = abs_norm_resid[:3]



for r, i in enumerate(abs_norm_resid_top_3):

    plot_lm_2.axes[0].annotate(i, 

                               xy=(sp.flip(QQ.theoretical_quantiles, 0)[r],

                                   model_norm_residuals[i]));
plot_lm_3 = plt.figure(3)

plot_lm_3.set_figheight(8)

plot_lm_3.set_figwidth(12)



plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)

sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 

            scatter=False, 

            ci=False, 

            lowess=True,

            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_3.axes[0].set_title('Scale-Location')

plot_lm_3.axes[0].set_xlabel('Fitted values')

plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');



# annotations

abs_sq_norm_resid = sp.flip(sp.argsort(model_norm_residuals_abs_sqrt), 0)

abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]



for i in abs_norm_resid_top_3:

    plot_lm_3.axes[0].annotate(i, 

                               xy=(model_fitted_y[i], 

                                   model_norm_residuals_abs_sqrt[i]));


plot_lm_4 = plt.figure(4)

plot_lm_4.set_figheight(8)

plot_lm_4.set_figwidth(12)



plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)

sns.regplot(model_leverage, model_norm_residuals, 

            scatter=False, 

            ci=False, 

            lowess=True,

            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_4.axes[0].set_xlim(0, 0.125)

plot_lm_4.axes[0].set_ylim(-3, 5)

plot_lm_4.axes[0].set_title('Residuals vs Leverage')

plot_lm_4.axes[0].set_xlabel('Leverage')

plot_lm_4.axes[0].set_ylabel('Standardized Residuals')



# annotations

leverage_top_3 = sp.flip(sp.argsort(model_cooks), 0)[:3]



for i in leverage_top_3:

    plot_lm_4.axes[0].annotate(i, 

                               xy=(model_leverage[i], 

                                   model_norm_residuals[i]))

    

# shenanigans for cook's distance contours

def graph(formula, x_range, label=None):

    x = x_range

    y = formula(x)

    plt.plot(x, y, label=label, lw=1, ls='--', color='red')



p = len(reg2.params) # number of model parameters



graph(lambda x: sp.sqrt((0.5 * p * (1 - x)) / x), 

      sp.linspace(0.001, 0.125, 50), 

      'Cook\'s distance') # 0.5 line

graph(lambda x: sp.sqrt((1 * p * (1 - x)) / x), 

      sp.linspace(0.001, 0.125, 50)) # 1 line

plt.legend(loc='upper right');
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.model_selection import KFold



from sklearn.metrics import mean_squared_error
n_splits = 5

rkf = KFold(n_splits=n_splits, shuffle=True, random_state=1028)
reg_mean_rmse = 0

reg2_mean_rmse = 0

ridge_reg_mean_rmse = 0

lasso_reg_mean_rmse = 0

elastic_reg_mean_rmse = 0





for trn_idx, val_idx in rkf.split(train):

    trn_X, trn_y = train.loc[trn_idx], price[trn_idx]

    val_X, val_y = train.loc[val_idx], price[val_idx]

    

    regr_pred = LinearRegression().fit(trn_X, trn_y).predict(val_X)

    regr2_pred = LinearRegression().fit(trn_X[result_pvalue], trn_y).predict(val_X[result_pvalue])

    regr3_pred = Ridge().fit(trn_X, trn_y).predict(val_X)

    regr4_pred = Lasso().fit(trn_X, trn_y).predict(val_X)

    regr5_pred = ElasticNet().fit(trn_X, trn_y).predict(val_X)

    

    reg_mean_rmse += sp.sqrt(mean_squared_error(sp.expm1(val_y), sp.expm1(regr_pred)))/n_splits

    reg2_mean_rmse += sp.sqrt(mean_squared_error(sp.expm1(val_y), sp.expm1(regr2_pred)))/n_splits

    ridge_reg_mean_rmse += sp.sqrt(mean_squared_error(sp.expm1(val_y), sp.expm1(regr3_pred)))/n_splits

    lasso_reg_mean_rmse += sp.sqrt(mean_squared_error(sp.expm1(val_y), sp.expm1(regr4_pred)))/n_splits

    elastic_reg_mean_rmse += sp.sqrt(mean_squared_error(sp.expm1(val_y), sp.expm1(regr5_pred)))/n_splits



print('Full Model RMSE : {0:0.2f}'.format(reg_mean_rmse))

print('Variable remove Model RMSE : {0:0.2f}'.format(reg2_mean_rmse))

print('Ridge Model RMSE : {0:0.2f}'.format(ridge_reg_mean_rmse))

print('Lasso Model RMSE : {0:0.2f}'.format(lasso_reg_mean_rmse))

print('Elastice Net Model RMSE : {0:0.2f}'.format(elastic_reg_mean_rmse))