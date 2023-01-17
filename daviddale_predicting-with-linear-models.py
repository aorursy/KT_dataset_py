import pandas as pd
import numpy as np
data = pd.read_csv('../input/credit_card_clients_split.csv')
print(data.shape)
data.head()
data['target'] = np.log(1 + data.avg_monthly_turnover)
data['log_income']= np.log(1 + data.income)
data['positive_target'] = (data.avg_monthly_turnover > 0).astype(int)
train_data = data[data.avg_monthly_turnover.notnull()].copy()
from sklearn.model_selection import train_test_split
train, test = train_test_split(train_data, test_size=0.5, random_state=1, shuffle=True)
train_pos = train[train.avg_monthly_turnover > 0]
from sklearn.linear_model import LinearRegression
linr = LinearRegression().fit(train_pos[['log_income']], train_pos['target'])
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
channel_dummy = pd.get_dummies(train.sales_channel_id)
logr = LogisticRegression().fit(channel_dummy, train.positive_target)
def predict(dataset):
    p_positive = logr.predict_proba(pd.get_dummies(dataset.sales_channel_id))[:, 1]
    log_positive = linr.predict(dataset[['log_income']])
    prediction = np.exp(p_positive * log_positive)
    return prediction
from sklearn.metrics import mean_squared_log_error
print(mean_squared_log_error(train.avg_monthly_turnover, predict(train)) ** 0.5)
print(mean_squared_log_error(test.avg_monthly_turnover, predict(test)) ** 0.5)
numeric_predictors = ['log_income', 'age']
categorical_predictors = ['education', 'sales_channel_id', 'wrk_rgn_code']
x_train = pd.get_dummies(train[categorical_predictors + numeric_predictors], columns=categorical_predictors)
x_train_columns = x_train.columns
print(x_train_columns)
def preprocess(dataset):
    x = pd.get_dummies(dataset[categorical_predictors + numeric_predictors], columns=categorical_predictors)
    x = x.reindex(x_train_columns, axis=1).fillna(0)
    return x
x_train = preprocess(train)
print(x_train.shape)
from sklearn.linear_model import Ridge
level1 = Ridge(alpha=1)
level1.fit(x_train[train.positive_target == 1], train.target[train.positive_target==1])
level0 = LogisticRegression(C=1, solver='liblinear')
level0.fit(x_train, train.positive_target)
def predict(dataset):
    x = preprocess(dataset)
    p_positive = level0.predict_proba(x)[:, 1]
    log_positive = level1.predict(x)
    prediction = np.exp(p_positive * log_positive)
    return prediction
print(mean_squared_log_error(train.avg_monthly_turnover, predict(train)) ** 0.5)
print(mean_squared_log_error(test.avg_monthly_turnover, predict(test)) ** 0.5)
def preprocess(dataset):
    x = pd.get_dummies(dataset[categorical_predictors + numeric_predictors], columns=categorical_predictors)
    x = x.reindex(x_train_columns, axis=1).fillna(0)
    # adding cross_products of features
    for i, c1 in enumerate(x_train_columns):
        for j, c2 in enumerate(x_train_columns):
            if j > i and c1[:10] != c2[:10]:
                x[c1 + '_' + c2] = x[c1] * x[c2]
    return x

x_train = preprocess(train)
print(x_train.shape)
level1 = Ridge(alpha=1)
level1.fit(x_train[train.positive_target == 1], train.target[train.positive_target==1])
level0 = LogisticRegression(C=1, solver='liblinear')
level0.fit(x_train, train.positive_target)
def predict(dataset):
    x = preprocess(dataset)
    p_positive = level0.predict_proba(x)[:, 1]
    log_positive = level1.predict(x)
    prediction = np.exp(p_positive * log_positive)
    return prediction
print(mean_squared_log_error(train.avg_monthly_turnover, predict(train)) ** 0.5)
print(mean_squared_log_error(test.avg_monthly_turnover, predict(test)) ** 0.5)
