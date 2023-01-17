import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from tqdm import tqdm_notebook as tqdm
from sklearn.linear_model import LinearRegression
df = pd.read_csv('sputnik/train.csv', sep =',')
df.head(2)
df.epoch = pd.to_datetime(df.epoch, format='%Y-%m-%d %H:%M:%S')
df.index  = df.epoch
df = df.drop('epoch', axis = 1)
df.head(2)
train = df[df.type == "train"]
test =  df[df.type == "test"]
train['error']  = np.linalg.norm(train[['x', 'y', 'z']].values - \
                                 train[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
test_groups = {i: test[test.sat_id == i] for i in test.sat_id.unique()}
train_groups = {i: train[train.sat_id == i] for i in train.sat_id.unique()}
plt.figure(figsize=(16,2))
plt.plot(train_groups[0].z)
plt.show()
result = sm.tsa.seasonal_decompose(train_groups[0].z,freq= 24)
result.plot()
plt.show()
def predict_set(train, test_set, ndays, T):
    def learn_model(features, variable):
        model = LinearRegression()
        test_df = df[df.mark == 'test'][features]
        train_df = df[df.mark == 'train'][features + ['{}_target'.format(variable)]].dropna()
        model.fit(train_df.drop('{}_target'.format(variable), axis = 1) ,train_df['{}_target'.format(variable)])
        return model.predict(test_df)
    
    pred = pd.DataFrame()
    x_features, y_features, z_features = [], [], []
    
    train_data = train[['x', 'y', 'z']]
    test_data = test_set[['x', 'y', 'z']]
    train_data['mark'] = 'train'
    test_data['mark'] = 'test'
    df = pd.concat((train_data, test_data), axis = 0)
    df[['x_target', 'y_target', 'z_target']] = df[['x', 'y', 'z']]
    
    for period_number in range(1, ndays):
        df[["x_period_lag_{}".format(period_number), \
           "y_period_lag_{}".format(period_number), \
           "z_period_lag_{}".format(period_number)]] = \
        df[['x_target', 'y_target', 'z_target']].shift(period_number * T)
        x_features.append("x_period_lag_{}".format(period_number))
        y_features.append("y_period_lag_{}".format(period_number))
        z_features.append("z_period_lag_{}".format(period_number))

    df['x_mean_lag'] = df[x_features].mean(axis = 1)
    df['y_mean_lag'] = df[y_features].mean(axis = 1)
    df['z_mean_lag'] = df[z_features].mean(axis = 1)
    x_features.append('x_mean_lag')
    y_features.append('y_mean_lag')
    z_features.append('z_mean_lag')

    pred['x'] = learn_model(x_features, 'x')
    pred['y'] = learn_model(y_features, 'y')
    pred['z'] = learn_model(z_features, 'z')
    pred.index = test_set.index
    
    return pred
T = 24
max_start_ndays = 3
max_growth_ndays = 6
predict = pd.DataFrame()
for sat_id in tqdm(train_groups):
    ndays = min(train_groups[sat_id].shape[0] // T, max_start_ndays)
    train, test = pd.DataFrame(), pd.DataFrame()
    train[['x', 'y', 'z']] = train_groups[sat_id][['x', 'y', 'z']]
    test[['x_sim', 'y_sim', 'z_sim', 'id']] = test_groups[sat_id][['x_sim', 'y_sim', 'z_sim', 'id']]
    for test_set in [test_groups[sat_id].iloc[i * T : (i + 1) * T] for \
                   i in range(test_groups[sat_id].shape[0] // T + 1)]:
        if test_set.empty:
            continue
        train = pd.concat((train, predict_set(train, test_set, ndays, T)), axis = 0)
        if ndays < max_growth_ndays:
            ndays += 1

    test[['x', 'y', 'z']] = train[train_groups[sat_id].shape[0]:]
    test['error']  = np.linalg.norm(test[['x', 'y', 'z']].values - \
                                      test[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
    predict = pd.concat((predict, test[['id', 'error']]), axis = 0)
predict.to_csv('prediction.csv', index = False)