import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv', encoding = "ISO-8859-1")

train.head(10)
train.info()
obj_cols = train.select_dtypes('object')

obj_cols.head()
obj_cols.drop('Born',axis=1,inplace=True)

train['Born'] = pd.to_datetime(train.Born)
train.Born.head()
for c in obj_cols.columns:

    print('Obj Col: ', c, '   Number of Unqiue Values ->', len(obj_cols[c].value_counts()))
373+37+18+16+2+573+308+18+68
fig, ax=plt.subplots(1,2,figsize=(18,10))

obj_cols['Cntry'].value_counts().sort_values().plot(kind='barh',ax=ax[0]) 

ax[0].set_title("Counts of Hockey Players by Country");

obj_cols['Cntry'].value_counts().plot(kind='pie', autopct='%.2f', shadow=True,ax=ax[1]);

ax[1].set_title("Distribution of Hockey Players by Country");

fig, ax=plt.subplots(1,1,figsize=(12,8))

obj_cols['Team'].value_counts().plot(kind='bar',ax=ax);

plt.title('Counts of Team Values');
train.Salary.head(10)
fig, ax=plt.subplots(1,1,figsize=(12,8))

train.Salary.plot(kind='hist',ax=ax, bins=20);

plt.title("Distribution of Salaries");

plt.xlabel('Dollars');
sal_gtmil = train[train.Salary >= 1e7]
sal_gtmil.head(10)
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
def data_clean(x):

    ## Were going to change Born to date time

    x['Born'] = pd.to_datetime(x.Born, yearfirst=True)

    x['dowBorn'] = x.Born.dt.dayofweek

    x["doyBorn"] = x.Born.dt.dayofyear

    x["monBorn"] = x.Born.dt.month

    x['yrBorn'] = x.Born.dt.year

    ## Drop Pr/St due to NaNs from other countries and First Name

    x.drop(['Pr/St','First Name'], axis=1, inplace=True)

    ocols = ['City', 'Cntry', 'Nat', 'Last Name', 'Position', 'Team']

    for oc in ocols:

        temp = pd.get_dummies(x[oc])

        x = x.join(temp, rsuffix=str('_'+oc))

    x['Hand'] = pd.factorize(x.Hand)[0]

    x.drop(ocols, axis=1, inplace=True)

    x.drop(['Born'],axis=1,inplace=True)

    return x

    

    
try:

    del train, x0, xc, test

except:

    pass
train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

train.head()
test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

test.head()
full = train.merge(test, how='outer')

print(train.shape, test.shape, full.shape)
y = np.log(full.Salary.dropna())

full0 = full.drop(['Salary'],axis=1)
fig, ax=plt.subplots(1,1,figsize=(10,6))

y.plot(ax=ax);

plt.title("Ln Salary");
obj_cols.columns
full_c = data_clean(full0)
print(full0.shape, full_c.shape)
full_c.head()
ss = StandardScaler()
full_cs = ss.fit_transform(full_c)
train_c = full_cs[:612]

test_c = full_cs[612:]
print(train_c.shape, y.shape, test_c.shape)
type(y)
folds = 3

lgbm_params = {

    "max_depth": -1,

    "num_leaves": 1000,

    "learning_rate": 0.01,

    "n_estimators": 1000,

    "objective":'regression',

    'min_data_in_leaf':64,

    'feature_fraction': 0.8,

    'colsample_bytree':0.8,

    "metric":['mae','mse'],

    "boosting_type": "gbdt",

    "n_jobs": -1,

    "reg_lambda": 0.9,

    "random_state": 123

}

preds = 0

for f in range(folds):

    xt, xv, yt, yv = train_test_split(train_c, y.values, test_size=0.2, random_state=((f+1)*123))

    

    xtd = lgb.Dataset(xt, label=yt)

    xvd = lgb.Dataset(xv, label=yv)

    mod = lgb.train(params=lgbm_params, train_set=xtd, 

                    num_boost_round=1000, valid_sets=xvd, valid_names=['valset'],

                    early_stopping_rounds=20, verbose_eval=20)

    

    preds += mod.predict(test_c)

    

preds = preds/folds

    

    
acts = pd.read_csv('../input/test_salaries.csv', encoding="ISO-8859-1")

acts['preds'] = np.exp(preds)

acts.head()
import matplotlib

from sklearn.metrics import mean_absolute_error, mean_squared_error
fig, ax=plt.subplots(1,1,figsize=(12,8))

acts.plot(ax=ax, style=['b-','r-']);

plt.title("Comparison of Preds and Actuals");

plt.ylabel('$');

ax.get_yaxis().set_major_formatter(

    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
mse = mean_squared_error(np.log(acts.Salary), np.log(acts.preds))

mae = mean_absolute_error(np.log(acts.Salary), np.log(acts.preds))
print("Ln Level Mean Squared Error :", mse)

print("Ln Level Mean Absolute Error :", mae)
fi_df = pd.DataFrame( 100*mod.feature_importance()/mod.feature_importance().max(), 

                      index=full_c.columns, #mod.feature_name(),

                      columns =['importance'])
fig, ax=plt.subplots(1,1,figsize=(12,8))

fi_df.sort_values(by='importance',ascending=True).iloc[-20:].plot(kind='barh', color='C0', ax=ax)

plt.title("Normalized Feature Importances");
import statsmodels.api as sma
top10 = fi_df.sort_values(by='importance',ascending=True).iloc[-10:].index

top10



exog = pd.DataFrame(test_c, columns=full_c.columns)[list(top10)].fillna(0)
ols = sma.OLS(exog=exog, endog=acts.Salary)

ols_fit = ols.fit()

print(ols_fit.summary())
ols_preds = ols_fit.predict()
fig, ax=plt.subplots(1,1,figsize=(12,8))

acts.Salary.plot(ax=ax, color='C1');

ax.plot(ols_preds, color='C0');

plt.title("Comparison of StatsModels Preds and Actuals");

plt.ylabel('$');

plt.legend(['salary actual','ols preds']);

ax.get_yaxis().set_major_formatter(

    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()