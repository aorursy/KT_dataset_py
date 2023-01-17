import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px 

sns.set_style('darkgrid')
data = pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')

print(data.shape)

data.head()
data.describe()
fig, ax = plt.subplots(figsize=(20, 6))

sns.heatmap(data.isnull(), cbar=False, yticklabels=False)
data['Date'] = pd.to_datetime(data['Data']).dt.date.astype(str)

data['TimeSunRise'] = data['Date'] + ' ' + data['TimeSunRise']

data['TimeSunSet'] = data['Date'] + ' ' + data['TimeSunSet']

data['Date'] = data['Date'] + ' ' + data['Time']



data = data.sort_values('Date').reset_index(drop=True)

data.set_index('Date', inplace=True)

data.drop(['Data', 'Time', 'UNIXTime'], axis=1, inplace=True)

data.index = pd.to_datetime(data.index)

data.head()
data.rename({

    'Radiation': 'Radiation(W/m2)', 'Temperature': 'Temperature(F)', 'Pressure': 'Pressure(mm Hg)', 'Humidity': 'Humidity(%)',

    'Speed': 'Speed(mph)'

}, axis=1, inplace=True)

data.head()
fig, ax = plt.subplots(figsize=(20, 6))

data['Radiation(W/m2)'].plot(ax=ax, style=['--'], color='red')

ax.set_title('Radiation as a Time Series', fontsize=18)

ax.set_ylabel('W/m2')

plt.show()
fig, ax = plt.subplots(figsize=(20, 6))

data.groupby(pd.Grouper(freq="D"))['Radiation(W/m2)'].mean().plot(ax=ax, style=['--'], color='red')

ax.set_title('Radiation as a Time Series (Daily)', fontsize=18)

ax.set_ylabel('W/m2')

plt.show()
for col in ['Radiation(W/m2)','Temperature(F)', 'Pressure(mm Hg)', 'Humidity(%)', 'WindDirection(Degrees)', 'Speed(mph)']:

    fig, ax = plt.subplots(figsize=(20, 3))

    data[col].plot.box(ax=ax, vert=False, color='red')

    ax.set_title(f'{col} Distrubution', fontsize=18)

    plt.show()
fig = plt.figure()

fig.suptitle('Feature Correlation', fontsize=18)

sns.heatmap(data.corr(), annot=True, cmap='RdBu', center=0)
def total_seconds(series):

    return series.hour*60*60 + series.minute*60 + series.second
data['MonthOfYear'] = data.index.strftime('%m').astype(int)

data['DayOfYear'] = data.index.strftime('%j').astype(int)

data['WeekOfYear'] = data.index.strftime('%U').astype(int)

data['TimeOfDay(h)'] = data.index.hour

data['TimeOfDay(m)'] = data.index.hour*60 + data.index.minute

data['TimeOfDay(s)'] = total_seconds(data.index)

data['TimeSunRise'] = pd.to_datetime(data['TimeSunRise'])

data['TimeSunSet'] = pd.to_datetime(data['TimeSunSet'])

data['DayLength(s)'] = total_seconds(data['TimeSunSet'].dt) - total_seconds(data['TimeSunRise'].dt)

data['TimeAfterSunRise(s)'] = total_seconds(data.index) - total_seconds(data['TimeSunRise'].dt)

data['TimeBeforeSunSet(s)'] = total_seconds(data['TimeSunSet'].dt) - total_seconds(data.index)

data['RelativeTOD'] = data['TimeAfterSunRise(s)'] / data['DayLength(s)']

data.drop(['TimeSunRise','TimeSunSet'], inplace=True, axis=1)

data.head()
fig, ax = plt.subplots(4, 2, figsize=(20, 20))

for j, timeunit in enumerate(['MonthOfYear', 'TimeOfDay(h)']):

    grouped_data=data.groupby(timeunit).mean().reset_index()

    palette = sns.color_palette("YlOrRd", len(grouped_data))

    for i, col in enumerate(['Radiation(W/m2)', 'Temperature(F)', 'Pressure(mm Hg)', 'Humidity(%)']):

        sns.barplot(data=grouped_data, x=timeunit, y=col, ax=ax[i][j], palette=palette)

        ax[i][j].set_title(f'Mean {col} by {timeunit}', fontsize=12)

        range_values = grouped_data[col].max() - grouped_data[col].min()

        ax[i][j].set_ylim(max(grouped_data[col].min() - range_values, 0), grouped_data[col].max() + 0.25*range_values)
fig = plt.figure(figsize=(20, 12))

fig.suptitle('Feature Correlation', fontsize=18)

sns.heatmap(data.corr(), annot=True, cmap='RdBu', center=0)
feats = [

    'Temperature(F)', 'Pressure(mm Hg)', 'Humidity(%)', 'WindDirection(Degrees)', 'Speed(mph)', 

    'MonthOfYear','DayOfYear', 'RelativeTOD',

]

X = data[feats].values

y = data['Radiation(W/m2)'].values



print(X.shape)
from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.dummy import DummyRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error



kf = KFold(shuffle=True, random_state=19)
scores = []

rmse = []

mae = []



for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = DummyRegressor(strategy='mean').fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

    rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    mae.append(mean_absolute_error(y_test, model.predict(X_test)))

    

print('Mean R2 Score:', round(np.mean(scores), 5))

print('Mean RMSE:', round(np.mean(rmse), 5))

print('Mean MAE:', round(np.mean(mae), 5))
%%time



scores = []

rmse = []

mae = []



for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    dtmodel = DecisionTreeRegressor(random_state=19).fit(X_train, y_train)

    scores.append(dtmodel.score(X_test, y_test))

    rmse.append(np.sqrt(mean_squared_error(y_test, dtmodel.predict(X_test))))

    mae.append(mean_absolute_error(y_test, dtmodel.predict(X_test)))

    

print('Mean R2 Score:', round(np.mean(scores), 5))

print('Mean RMSE:', round(np.mean(rmse), 5))

print('Mean MAE:', round(np.mean(mae), 5))
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor, XGBRFRegressor

from catboost import CatBoostRegressor



trees = {

    'RandomForest': RandomForestRegressor(random_state=19), 'ExtraTrees': ExtraTreesRegressor(random_state=19),

    'GradientBoosting': GradientBoostingRegressor(random_state=19), 'LightGBM': LGBMRegressor(random_state=19),

    'XGBoost': XGBRegressor(random_state=19), 'XGBoostRF': XGBRFRegressor(random_state=19), 

    'CatBoost': CatBoostRegressor(random_state=19, silent=True)

}
%%time



performance = {'rmse':[], '100* r2':[], 'mae':[]}

for name, model in trees.items():

    scores = []

    rmse = []

    mae = []



    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        model = model.fit(X_train, y_train)

        scores.append(100*model.score(X_test, y_test))

        rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

        mae.append(mean_absolute_error(y_test, model.predict(X_test)))

    performance['100* r2'].append(np.mean(scores))

    performance['rmse'].append(np.mean(rmse))

    performance['mae'].append(np.mean(mae))
fig = px.bar(pd.DataFrame(performance, index=trees.keys()), barmode='group', title='Model Comparison')

fig.show()
feat_imp = {

    k: trees[k].feature_importances_ for k, v in trees.items()

}

feat_imp['DecisionTree'] = dtmodel.feature_importances_

feat_imp = pd.DataFrame(feat_imp)



feat_imp /= feat_imp.sum()

feat_imp.index = feats



fig, ax= plt.subplots(figsize=(20, 6))

fig.suptitle('Feature Importance', fontsize=18)

pd.DataFrame(feat_imp).plot.bar(ax=ax, color=sns.color_palette("summer", 8))