!pip install -U xgboost
import pandas as pd
import os
import numpy as np
import seaborn as sns
import xgboost as xgb
import datetime
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))
def baseline(y_test):
    # step = len(y_test.columns)
    y_actual = np.array(y_test)[1:,0]
    y_pred = np.array(y_test)[:-1,0]
    # y_pred = np.repeat(np.array(y_test['PM2.5(t)'][:-1]), step, axis=0).reshape(-1, step)
    score = smape(y_actual, y_pred)
    print('baseline sampe: ', score)
# 生成单个站点的数据集

def generate_station_dataset(data, labels, station, n_in=1, n_out=1, dropnan=True):
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    label_idx = list()
    for idx, label in enumerate(data.columns):
        if label in labels:
            label_idx.append(idx)
    for i in range(0, n_out):
        if i == 0:
            cols.append(df[labels].shift(-i))
            names += [('%s(t)' % (df.columns[j])) for j in label_idx]
        else:
            cols.append(df[labels].shift(-i))
            names += [('%s(t+%d)' % (df.columns[j], i)) for j in label_idx]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    print(agg.isnull().any())
    if dropnan:
        agg.dropna(inplace=True)
    X = agg[agg.columns[:-n_out*len(labels)]]
    y = agg[agg.columns[-n_out*len(labels):]]

    return X, y
# 生成带id的单个站点的数据集

def generate_station_dataset_with_id(data, labels, station_id, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    label_idx = list()
    for idx, label in enumerate(data.columns):
        if label in labels:
            label_idx.append(idx)
    for i in range(0, n_out):
        if i == 0:
            cols.append(df[labels].shift(-i))
            names += [('%s(t)' % (df.columns[j])) for j in label_idx]
        else:
            cols.append(df[labels].shift(-i))
            names += [('%s(t+%d)' % (df.columns[j], i)) for j in label_idx]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    X, y = agg[agg.columns[:-n_out*len(labels)]].copy(), agg[agg.columns[-n_out*len(labels):]].copy()
    X['station'] = [station_id]*len(X)
    
    agg = pd.concat([X,y], axis=1)

    return agg

# 逐站点生成数据集后拼接

def generate_dataset(data, labels, n_in=1, n_out=1, dropnan=True):
    stations = data.loc[:,'station']
#     station_ids = encoder.transform(stations)
    dataset = None
    for idx, station in enumerate(np.unique(stations)):
#         print(station)
        data_block = data[data['station']==station].drop('station', axis=1)
        dataset_block = generate_station_dataset_with_id(data_block, labels, idx, n_in, n_out, dropnan)
        
        if dataset is None:
            dataset = dataset_block
        else:
            dataset = dataset.append(dataset_block)
    X, y = dataset[dataset.columns[:-n_out*len(labels)]], dataset[dataset.columns[-n_out*len(labels):]]
    
    return X, y
# 生成带所有站点特征的数据集

def generate_full_station_dataset_with_id(data, labels, station_id, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    label_idx = list()
    for idx, label in enumerate(data.columns):
        if label in labels:
            label_idx.append(idx)
    for i in range(0, n_out):
        if i == 0:
            cols.append(df[labels].shift(-i))
            names += [('%s(t)' % (df.columns[j])) for j in label_idx]
        else:
            cols.append(df[labels].shift(-i))
            names += [('d%s(t+%d)' % (df.columns[j], i)) for j in label_idx]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    X, y = agg[agg.columns[:-n_out*len(labels)]].copy(), agg[agg.columns[-n_out*len(labels):]].copy()
    X['station'] = [station_id]*len(X)
    
    agg = pd.concat([X,y], axis=1)

    return agg

def generate_full_dataset(data, labels, n_in=1, n_out=1, dropnan=True):
    stations = data.loc[:,'station']
#     station_ids = encoder.transform(stations)
    dataset = None
    for idx, station in enumerate(np.unique(stations)):
#         print(station)
        data_block = data[data['station']==station].drop('station', axis=1)
        dataset_block = generate_full_station_dataset_with_id(data_block, labels, idx, n_in, n_out, dropnan)
        
        if dataset is None:
            dataset = dataset_block
        else:
            dataset = dataset.append(dataset_block)
    X, y = dataset[dataset.columns[:-n_out*len(labels)]], dataset[dataset.columns[-n_out*len(labels):]]
    
    return X, y
dir = '../input/beijing-multisite-airquality-data-data-set/PRSA_Data_20130301-20170228/'
files = os.listdir(dir)
data_all = pd.DataFrame()
for idx, f in enumerate(files):
    df = pd.read_csv(dir+f, index_col=0)
    df['station'] = f.strip('.csv').split('_')[2]
    if idx == 0:
        data_all = df
    else:
        data_all = pd.concat([data_all, df])
data_all.drop('wd', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

data_all['wd'].fillna(value='NAN', inplace = True)
encoder.fit(data_all['wd'])
data_all['wd'] = encoder.transform(data_all['wd'])
encoder.fit(data_all['station'])
data_all['station'] = encoder.transform(data_all['station'])
data_all.interpolate(inplace=True, limit_direction='both')
data_all.isnull().any()
data_all.reset_index(inplace=True,drop=True)
data_all
grouped = data_all.groupby(["year","month","day","hour"])

for g in grouped:
    subdf = g[1]
    subdf.sort_values('station')
    row,names = list(),list()
    for idx, r in subdf.iterrows():
        row.append(r)
        names += [('%s_%s_'%(r.station,r.index[i])) for i in range(len(r.index))]
        s = pd.Series(index=names, data=row)
    print(names)
    break
#     print(subdf)
features = ['station', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']

data_train = data_all.loc[data_all.year <= 2016][features].copy()
data_test = data_all.loc[data_all.year > 2016][features].copy()
print(data_train.shape)
print(data_test.shape)
# 生成单个站点的数据集

def generate_station_dataset_t(data, labels, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    label_idx = list()
    for idx, label in enumerate(data.columns):
        if label in labels:
            label_idx.append(idx)
    for i in range(0, n_out):
        if i == 0:
            cols.append(df[labels].shift(-i))
            names += [('%s(t)' % (df.columns[j])) for j in label_idx]
        else:
            cols.append(df[labels].shift(-i))
            names += [('d%s(t+%d)' % (df.columns[j], i)) for j in label_idx]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    X, y = agg[agg.columns[:-n_out*len(labels)]].copy(), agg[agg.columns[-n_out*len(labels):]].copy()
#     X['station'] = [station_id]*len(X)

    return X, y
data_aoti_train = data_train[data_train['station']=='Aotizhongxin'].drop('station', axis=1)
data_aoti_test = data_test[data_test['station']=='Aotizhongxin'].drop('station', axis=1)
X_train, y_train = generate_station_dataset_t(data_aoti_train, n_in=24, n_out=12, labels=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP'])
X_test, y_test = generate_station_dataset_t(data_aoti_test, n_in=24, n_out=12, labels=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP'])
X_test, y_test = generate_dataset(data_test,  n_in=24, n_out=12, labels=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP'])
X_train, y_train = generate_dataset(data_train, n_in=24, n_out=12, labels=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP'])
X_train
y_test
X_test.drop('station', inplace=True, axis=1)
X_train.drop('station', inplace=True, axis=1)
model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=666, max_depth=4, tree_method='gpu_hist')).fit(X_train, y_train[y_train.columns[:9]], verbose=True)
from random import randint

def iter_predict(model, X_test, y_test, n_features, if_station=False):
    gt = list()
    pred = list()
    
    if if_station:
        station = X_test['station'].copy()
    for i in range(12):
        y_pred = model.predict(X_test)
        # drop late data
        if if_station:
            X_test.drop('station', inplace=True, axis=1)
        X_test = X_test.shift(-n_features, axis=1)
        # add predicted data
        X_test[X_test.columns[-n_features:]] = y_pred
        if if_station:
            X_test['station'] = station
        sample_gt = list()
        sample_pred = list()
        for j in range(20):
            idx = randint(0,len(y_test)-1)
            sample_gt.append(y_test[y_test.columns[i*n_features]].tolist()[idx])
            sample_pred.append(y_pred[idx,0])
        gt.append(sample_gt)
        pred.append(sample_pred)
        print('iter %d: smape=%f'%(i, smape(y_test[y_test.columns[i*n_features]], y_pred[:,0])))
        
    return np.array(gt), np.array(pred)
gt, pred = iter_predict(model, X_test, y_test, 9)
plt.plot(range(12), gt[:,0])
plt.plot(range(12), pred[:,0])
plt.plot(range(12), gt[:,1])
plt.plot(range(12), pred[:,1])
plt.plot(range(12), gt[:,5])
plt.plot(range(12), pred[:,5])
print(gt[:5])
print(pred[:5])
iter_predict(model, X_test, y_test, 9)
!pip install joblib
import joblib

joblib.dump(model, 'Xgboost.model')
y_pred = model.predict(X_test)
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
p = figure(title="Prediction", x_axis_label='x', y_axis_label='y', width=1100, height=400)
p.line(range(len(y_pred)), y_pred[:,0], legend_label="pred(t)", line_width=2, line_color='yellow')
# p.line(range(1,len(y_pred)), y_pred[:-1,1], legend_label="pred(t+1)", line_width=2, line_color='blue')
p.line(range(11,len(y_pred)), y_pred[:-11,11], legend_label="pred(t+11)", line_width=2, line_color='blue')
p.line(range(4,len(y_pred)), y_pred[:-4,4], legend_label="pred(t+4)", line_width=2, line_color='green')
p.line(range(len(y_test)), np.array(y_test)[:,0], legend_label="ground truth", line_width=2, line_color='red')
output_notebook()
show(p)
from catboost import CatBoostRegressor
cat_model = CatBoostRegressor(iterations=40, depth=5,learning_rate=0.1, loss_function='MultiRMSE',logging_level='Verbose')
cat_model.fit(X_train,y_train[y_train.columns[:9]],eval_set=(X_test, y_test[y_test.columns[:9]]),plot=True)
iter_predict(cat_model, X_test, y_test, 9)