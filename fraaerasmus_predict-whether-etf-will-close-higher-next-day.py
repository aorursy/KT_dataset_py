import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

PATH = '../input/Data/ETFs'
etfs = dict()
for filename in os.listdir(PATH):
    if not filename.endswith('.txt'):
        print('Skipping', filename)
        continue
    etfs[filename[:-4]] = pd.read_csv(PATH + '/' + filename)
print('Loaded', len(etfs), 'ETFs')
def prepare_samples(etf):
    # Delete rows where opening or closing prices is 0,
    # since we can't compute pct_change for those.
    etf = etf[(etf['Open'] != 0) & (etf['Close'] != 0)]
    
    samples = pd.DataFrame()
    # Target
    samples['target'] = (etf['Close'].pct_change() > 0).shift(-1)
    
    # Features
    samples['Open_chg'] = etf['Open'].pct_change()
    for days in range(1, 6):
        samples['Open_chg_' + str(days) + 'days_ago'] = samples['Open_chg'].shift(days)
    
    samples['Open_return'] = 1.0 + samples['Open_chg']

    samples['Open_above_30day_avg'] = (etf['Open'] > etf['Open'].rolling(30).mean()).astype(np.float)

    samples['Open_5day_return'] = samples['Open_return'].rolling(5).apply(np.prod)
    samples['Open_10day_return'] = samples['Open_return'].rolling(10).apply(np.prod)

    chg_functions = {'std': np.std, 'mean': np.mean, 'max': np.max}
    for name, function in chg_functions.items():
        samples['Open_chg_5day_' + name] = samples['Open_chg'].rolling(5).apply(function)
        samples['Open_chg_10day_' + name] = samples['Open_chg'].rolling(10).apply(function)

    del samples['Open_return']
    
    return samples.iloc[30:-1]

samples = pd.concat([prepare_samples(etfs[symbol]) for symbol in etfs.keys()])
print('Prepared', len(samples), 'samples.')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.utils import shuffle

shuffled_samples = shuffle(samples, random_state=2610)
y = shuffled_samples['target']
X = shuffled_samples.drop(['target'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
train_X.shape
import xgboost

model = xgboost.XGBClassifier(n_estimators=130, learning_rate=1.0)
model.fit(train_X, train_y)
model.best_score
print('Accuracy', (model.predict(test_X) == test_y).mean())
print('Random model accuracy', ((np.random.rand(len(test_y)) < 0.5) == test_y).mean())
print('Constant True model accuracy', (True == test_y).mean())
print('Constant False model accuracy', (False == test_y).mean())
import matplotlib.pyplot as plt
importances = sorted(zip(X.columns, model.feature_importances_), key=lambda i: -i[1])
plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), [i[1] for i in importances])
plt.xticks(range(len(importances)), [i[0] for i in importances], rotation=90);
