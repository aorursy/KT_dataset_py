import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



import statsmodels.api as sm

from statsmodels.stats.diagnostic import normal_ad

from statsmodels.stats.diagnostic import normal_ad

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from sklearn.pipeline import Pipeline

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.pipeline import make_pipeline

from patsy import dmatrices



from sklearn.preprocessing import MinMaxScaler



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



plt.style.use('ggplot')



def get_x():

    return data[list(filter(lambda x: x != 'MEDV', data.columns.values))].copy()



def get_y():

    return data['MEDV'].copy()



def get_norm_x():

    return MinMaxScaler().fit_transform(get_x())



def get_norm_y():

    return MinMaxScaler().fit_transform(get_y().values.reshape(-1, 1))
data = pd.read_csv('../input/data.csv')

data.head()
data.describe().drop(['count'])
data.info()
sns.pairplot(data=data, y_vars=['MEDV'], x_vars=list(filter(lambda x: x != 'MEDV' and x != 'CHAS', data.columns)), kind='reg')

plt.draw()
v = MinMaxScaler()

d = v.fit_transform(data)

fig, ax = plt.subplots(1,1, figsize=(20, 10))

sns.boxplot(data=d, palette="Set2",dodge=False, ax=ax)

ax.set_xticklabels(data.columns)

ax.set_title('Features distributions')

plt.show()
fig, axes = plt.subplots(2, 7, figsize=(20, 10))

data_without_chas = data.drop(['CHAS'], axis=1)



for i, ax in zip(data_without_chas.columns, axes.flatten()):

    sns.kdeplot(data_without_chas[i], ax = ax)

    ax.set_title(i)

    ax.get_legend().remove()
normal_ad(data)
get_x()['CHAS'].value_counts().plot(kind='bar')
data.corr()
independent_values = filter(lambda x: x != 'MEDV', data.columns.values) 

cols = filter(lambda x: x != i ,independent_values)

y, X = dmatrices(formula_like= 'MEDV' + " ~" + " + ".join(cols), data=data, return_type="dataframe")

vif_values = [vif(X.values, i) for i in range(X.shape[1])]

v = dict(zip(X.columns.values, vif_values))

pd.DataFrame({k: [v] for k,v in v.items() }).drop('Intercept', axis=1).T.plot(kind='bar')

X = get_x()

y = get_y()

linearModelFit = sm.OLS(y, X).fit()

linearModelFit.summary()
X = get_x()[['RM', 'DIS']]

X = MinMaxScaler().fit_transform(X)

linearModelFit = sm.OLS(y, X).fit()

linearModelFit.summary()
X = get_x()[['RM', 'LSTAT']]

X.LSTAT = stats.boxcox(X.LSTAT)[0]

X = MinMaxScaler().fit_transform(X)

linearModelFit = sm.OLS(y, X).fit()

linearModelFit.summary()
res = linearModelFit.resid

fig, axes = plt.subplots(1, 2, figsize=(20,5))

sm.qqplot(res, stats.t, fit=True, line='45', ax=axes[0])

sns.kdeplot(res)
X_train, X_test, y_train, y_test = train_test_split(get_x()[['RM', 'LSTAT']], y)

lm_pipe =  Pipeline([('scaler', MinMaxScaler()), ('lm', LinearRegression())])

fit = lm_pipe.fit(X_train,y_train)

v = fit.predict(X_test)

display(r2_score(v, y_test))

%%time

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from xgboost import plot_tree



def get_xgboost_best_model(X_train, y_train):

    

    params =[ \

            {'xgb__n_estimators': [5, 50, 100], \

             'xgb__max_depth': [5, 10, 20], \

             'xgb__learning_rate': [0.01, 0.1, 1], \

             'xgb__reg_alpha': [0.01, 0.1, 1, 10], \

             'xgb__reg_lambda': [0.01, 0.1, 1, 10],

             'xgb__gamma': [0.01, 0.1, 1, 10]}]

    

    pipe = Pipeline([('scaler', MinMaxScaler()), ('xgb', xgb.XGBRegressor()) ]) 

    grid = GridSearchCV(pipe, param_grid=params)

    fit = grid.fit(X_train, y_train)

    return fit.best_estimator_



X_train, X_test, y_train, y_test = train_test_split(get_x(), y)







fit = get_xgboost_best_model(X_train, y_train)

display(fit)

display("R2: " + str(r2_score(fit.predict(X_test), y_test)))
from sklearn.model_selection import cross_val_score

from xgboost import plot_tree



best_model = fit.steps[1][1]

fitted_model = best_model.fit(X_train, y_train)

xgb.plot_importance(fitted_model, importance_type='weight')
xgb.plot_importance(fitted_model, importance_type='cover')
xgb.plot_importance(fitted_model, importance_type='gain')
plot_tree(model.fit(X_train, y_train))

fig = plt.gcf()

fig.set_size_inches(10, 5)
X_feature_names = ['RM', 'LSTAT', 'NOX', 'DIS']

X = pd.DataFrame(MinMaxScaler().fit_transform(get_x()[X_feature_names]))

X.columns = X_feature_names

y = get_y()



X_train, X_test, y_train, y_test = train_test_split(X, y)

fitted_model = best_model.fit(X_train, y_train)

display("R2: " + str(r2_score(fitted_model.predict(X_test), y_test)))

plot_tree(fitted_model)

fig = plt.gcf()

fig.set_size_inches(25, 10)
fig, ax = plt.subplots(1,1,figsize=(10,5))

xgb.plot_importance(fitted_model, importance_type='gain', ax=ax)