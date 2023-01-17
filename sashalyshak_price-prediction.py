import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import lightgbm as lgbm

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression
train_data = pd.read_csv('../input/mobile-price-classification/train.csv')

test_data = pd.read_csv('../input/mobile-price-classification/test.csv')



train_data.head()
train_data.describe()
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

features = train_data.columns.values.tolist()

for col in features:

    if train_data[col].dtype in numerics: continue

    categorical_columns.append(col)

indexer = {}

for col in categorical_columns:

    if train_data[col].dtype in numerics: continue

    _, indexer[col] = pd.factorize(train_data[col])

    

for col in categorical_columns:

    if train_data[col].dtype in numerics: continue

    train_data[col] = indexer[col].get_indexer(train_data[col])
corr = train_data.corr()

corr
fig = plt.figure(figsize=(15,12))

sns.heatmap(corr)
corr.sort_values(by=["price_range"],ascending=False).iloc[0].sort_values(ascending=False)
plt.hist(train_data['battery_power'])

plt.show()
plt.figure(figsize=(8,8))

sns.FacetGrid(train_data, hue="price_range", size=8).map(sns.kdeplot, "battery_power").add_legend()

plt.ioff() 

plt.show()
plt.hist(train_data['ram'])

plt.show()
plt.figure(figsize=(8,8))

sns.FacetGrid(train_data, hue="price_range", size=8).map(sns.kdeplot, "ram").add_legend()

plt.ioff() 

plt.show()
sns.countplot(train_data['dual_sim'])

plt.show()
plt.figure(figsize=(8,8))

sns.FacetGrid(train_data, hue="price_range", size=8).map(sns.kdeplot, "dual_sim").add_legend()

plt.ioff() 

plt.show()
plt.figure(figsize=(8,8))

sns.FacetGrid(train_data, hue="price_range", size=8).map(sns.kdeplot, "four_g").add_legend()

plt.ioff() 

plt.show()
plt.figure(figsize=(10,6))

train_data['fc'].hist(alpha=0.5,color='blue',label='Front camera')

train_data['pc'].hist(alpha=0.5,color='red',label='Primary camera')

plt.legend()

plt.xlabel('MegaPixels')
y = train_data['price_range']



del train_data['price_range']



X = train_data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
train_set = lgbm.Dataset(X_train, y_train, silent=False)

valid_set = lgbm.Dataset(X_valid, y_valid, silent=False)



params = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.8,

        'bagging_fraction' : 1,

        'max_bin' : 5000 ,

        'bagging_freq': 20,

        'colsample_bytree': 0.6,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'zero_as_missing': True,

        'seed':0,        

    }



modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,

                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgbm.plot_importance(modelL,ax = axes,height = 0.5)

plt.show();plt.close()
feature_score = pd.DataFrame(X.columns, columns = ['feature']) 

feature_score['score_lgb'] = modelL.feature_importance()
# Standardization for regression models

train = pd.DataFrame(

    preprocessing.MinMaxScaler().fit_transform(train_data),

    columns=train_data.columns,

    index=train_data.index

)
logreg = LogisticRegression()

logreg.fit(train, y)

coeff_logreg = pd.DataFrame(train.columns.delete(0))

coeff_logreg.columns = ['feature']

coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])

coeff_logreg.sort_values(by='score_logreg', ascending=False)
coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs()

feature_score = pd.merge(feature_score, coeff_logreg, on='feature')
linreg = LinearRegression()

linreg.fit(train, y)

coeff_linreg = pd.DataFrame(train.columns.delete(0))

coeff_linreg.columns = ['feature']

coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)

coeff_linreg.sort_values(by='score_linreg', ascending=False)
coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()
feature_score = pd.merge(feature_score, coeff_linreg, on='feature')

feature_score = feature_score.fillna(0)

feature_score = feature_score.set_index('feature')

feature_score
feature_score['mean'] = feature_score.mean(axis=1)
feature_score['total'] = 0.7*feature_score['score_lgb'] + 0.15*feature_score['score_logreg'] + 0.15*feature_score['score_linreg']
feature_score.sort_values('total', ascending=False)
feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 10))
feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 10))
feature_columns = ['ram', 'px_height', 'px_width', 'mobile_wt', 'pc', 'sc_w']

X = X[feature_columns];



X.head()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

accur = dtree.score(X_valid,y_valid) * 100

accur
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

accur = rfc.score(X_valid, y_valid) * 100

accur
gbc = GradientBoostingClassifier()

gbc.fit(X_train,y_train)

accur = gbc.score(X_valid,y_valid) * 100

accur
X_new = test_data[feature_columns]



prediction = GradientBoostingClassifier()

prediction.fit(X, y)

predicted_price=prediction.predict(X_new)
predicted_price