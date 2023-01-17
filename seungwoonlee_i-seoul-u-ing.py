import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_summary = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

df_item = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')

df_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')

df_station = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
df_summary.head()
df_summary['Measurement date'] = pd.to_datetime(df_summary['Measurement date'], format='%Y-%m-%d %H:%M:%S')
df_summary['hour'] = df_summary['Measurement date'].dt.hour

df_summary['Measurement date'] = df_summary['Measurement date'].dt.date
df_summary = df_summary[['Measurement date', 'hour', 'Station code', 'SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']]
df_summary.head()
df_station[df_station['Station name(district)'] == 'Gangnam-gu']
df_summary_123 = df_summary[df_summary['Station code'] == 123]
df_aws = pd.read_csv('../input/aws-test/Gangnam.csv', encoding='CP949')
df_aws.head()
df_aws['Station code'] = 123
df_aws.head()
df_aws['Measurement date'] = pd.to_datetime(df_aws['Measurement date'], format='%Y-%m-%d %H:%M:%S')
df_aws['hour'] = df_aws['Measurement date'].dt.hour

df_aws['Measurement date'] = df_aws['Measurement date'].dt.date
gangnam = pd.merge(df_aws, df_summary_123, on=['Station code','Measurement date','hour'])
gangnam = gangnam[['Station code','Measurement date', 'hour', '기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '습도(%)',

                   'SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']]
gangnam.head()
gangnam[['기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '습도(%)','SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']].describe()
gangnam.isnull().sum()
gangnam['강수량(mm)'] = gangnam['강수량(mm)'].fillna(0)

gangnam = gangnam.fillna(gangnam.mean())
gangnam[['기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '습도(%)','SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']].corr()
from windrose import WindroseAxes

from matplotlib import pyplot as plt

import matplotlib.cm as cm



wd = gangnam['풍향(deg)']

ws = gangnam['풍속(m/s)']



ax = WindroseAxes.from_ax()

ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', cmap = cm.viridis)

ax.set_legend()
X = gangnam[['기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '습도(%)','SO2', 'NO2', 'O3', 'CO', 'PM2.5']]

y = gangnam['PM10']
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



estimators = [DecisionTreeRegressor(random_state=42),

         RandomForestRegressor(random_state=42),

         GradientBoostingRegressor(random_state=42)

        ]

estimators
max_depth = np.random.randint(5, 30, 10)

max_depth
max_features = np.random.uniform(0.3, 1.0, 10)

max_features
results = []

for estimator in estimators:

    result = []

    result.append(estimator.__class__.__name__)

    results.append(result)



pd.DataFrame(results)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import RandomizedSearchCV



results = []

for estimator in estimators:

    result = []



    max_depth = np.random.randint(5, 30, 100)

    max_features = np.random.uniform(0.3, 1.0, 100)



    param_distributions = {"max_depth": max_depth, "max_features": max_features }



    regressor = RandomizedSearchCV(estimator,

        param_distributions,

        n_iter=100,

        scoring=None,

        n_jobs=-1,

        cv=5,

        verbose=2, 

        random_state=42)



    regressor.fit(X_train, y_train)

    

    result.append(estimator.__class__.__name__)

    result.append(regressor.best_params_)

    result.append(regressor.best_estimator_)

    result.append(regressor.best_score_)

    result.append(regressor.cv_results_)

    results.append(result)
df_cv = pd.DataFrame(results)

df_cv.columns = ["model", "params", "estimator", "score", "cv_result"]

df_cv
best_estimator = df_cv.loc[1, "estimator"]

best_estimator
best_estimator.fit(X_train, y_train)
from sklearn.model_selection import cross_val_predict



y_predict = cross_val_predict(best_estimator, X_train, y_train, cv=5, verbose=2, n_jobs=-1)

y_predict[:5]
sns.regplot(y_train, y_predict)
sns.distplot(y_train, hist=False, label="train")

sns.distplot(y_predict, hist=False, label="predict")
np.sqrt(((y_train - y_predict) ** 2).mean())
best_estimator.feature_importances_
plt.figure(figsize = (10,5))

sns.barplot(x=best_estimator.feature_importances_, y=X.columns)
print("훈련 세트 정확도: {:.3f}".format(best_estimator.score(X_train, y_train)))

print("테스트 세트 정확도: {:.3f}".format(best_estimator.score(X_test, y_test)))