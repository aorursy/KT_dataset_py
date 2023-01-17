import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import lightgbm as lgbm

import eli5

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')

test_data = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')
train_data.head()
mapping = {'Yes': 1, 'No': 0}



train_data = train_data.replace({'RainToday': mapping})

train_data = train_data.replace({'RainTomorrow': mapping})



test_data = test_data.replace({'RainToday': mapping})
# Drop useless coluns

cols_to_remove = ['Date', 'Location', 'RISK_MM'] 

train_data.drop(cols_to_remove, axis=1, inplace=True)



test_data.drop(['RainTomorrow'], axis=1, inplace=True)
train_data = train_data.dropna(how='any')

test_data = test_data.dropna(how='any')
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
train_data.head()
train_data.info()
corr = train_data.corr()

fig = plt.figure(figsize=(15,10))

sns.heatmap(corr)
corr.sort_values(by=["RainTomorrow"],ascending=False).iloc[0].sort_values(ascending=False)
plt.figure(figsize=(8,8))

sns.FacetGrid(train_data, hue="RainTomorrow", size=8).map(sns.kdeplot, "Humidity3pm").add_legend()

plt.ioff() 

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(data=train_data,x='RainToday')
plt.figure(figsize=(8,8))

sns.FacetGrid(train_data, hue="RainTomorrow", size=8).map(sns.kdeplot, "MinTemp").add_legend()

plt.ioff() 

plt.show()
plt.figure(figsize=(8,8))

sns.FacetGrid(train_data, hue="RainTomorrow", size=8).map(sns.kdeplot, "MaxTemp").add_legend()

plt.ioff() 

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(data=train_data,x='WindGustDir')
plt.figure(figsize=(8,8))

sns.countplot(data=train_data,x='WindDir9am')
y = train_data['RainTomorrow']

del train_data['RainTomorrow']



X = train_data;
# data split

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
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X, y)

coeff_logreg = pd.DataFrame(X.columns.delete(0))

coeff_logreg.columns = ['feature']

coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])

coeff_logreg.sort_values(by='score_logreg', ascending=False)
coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs()

feature_score = pd.merge(feature_score, coeff_logreg, on='feature')
eli5.show_weights(logreg)
# Linear Regression



linreg = LinearRegression()

linreg.fit(X, y)

coeff_linreg = pd.DataFrame(X.columns.delete(0))

coeff_linreg.columns = ['feature']

coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)

coeff_linreg.sort_values(by='score_linreg', ascending=False)
eli5.show_weights(linreg)
coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()
feature_score = pd.merge(feature_score, coeff_linreg, on='feature')

feature_score = feature_score.fillna(0)

feature_score = feature_score.set_index('feature')

feature_score
feature_score = pd.DataFrame(

    preprocessing.MinMaxScaler().fit_transform(feature_score),

    columns=feature_score.columns,

    index=feature_score.index

)



# Create mean column

feature_score['mean'] = feature_score.mean(axis=1)



# Plot the feature importances

feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 10))
feature_score['total'] = 0.7*feature_score['score_lgb'] + 0.15*feature_score['score_logreg'] + 0.15*feature_score['score_linreg']



# Plot the feature importances

feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 10))
feature_score.sort_values('total', ascending=False)
feature_columns = ['Humidity3pm', 'Pressure3pm', 'WindGustSpeed', 'Pressure9am', 'Sunshine', 'Temp9am']

X = X[feature_columns];



X.head()
# data split for train

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# Decision Tree Classifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

pred = decision_tree.predict(X_valid)



results = []

result = accuracy_score(y_valid, pred) * 100

results.append(result)



print(result)
# Extra Trees Classifier

etr = ExtraTreesClassifier(n_estimators=100)

etr.fit(X_train, y_train)

pred = etr.predict(X_valid)



result = accuracy_score(y_valid, pred) * 100

results.append(result)

print(result)
#random forrest classifier

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

pred = clf.predict(X_valid)



result = accuracy_score(y_valid, pred) * 100

results.append(result)

print(result)
x = np.arange(3)



fig, ax = plt.subplots()

plt.bar(x, results)

ax.set_ylim(bottom=78)

plt.xticks(x, ('DecisionTree', 'ExtraTrees', 'RandomForest'))

plt.show()
clss = ExtraTreesClassifier(n_estimators=100)

clss.fit(X, y)
condition = [test_data[feature_columns].mean().values.tolist()]
condition
pred = clss.predict(condition)



print(pred)