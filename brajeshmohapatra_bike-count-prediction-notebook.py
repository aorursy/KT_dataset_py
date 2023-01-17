import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
%matplotlib inline
pd.pandas.set_option('display.max_columns', None)
df_train = pd.read_csv('../input/bike-count-prediction-data-set/train.csv')
df_test = pd.read_csv('../input/bike-count-prediction-data-set/test.csv')
df_train.head()
df_test.head()
df_train.shape, df_test.shape
data_types_train = pd.DataFrame(df_train.dtypes, columns = ['Train'])
data_types_test = pd.DataFrame(df_test.dtypes, columns = ['Test'])
data_types = pd.concat([data_types_train, data_types_test], axis = 1)
data_types
missing_values_train = pd.DataFrame(df_train.isna().sum(), columns = ['Train'])
missing_values_test = pd.DataFrame(df_test.isna().sum(), columns = ['Test'])
missing_values = pd.concat([missing_values_train, missing_values_test], axis = 1)
missing_values
df_train['yyyymmdd'] = df_train['datetime'].apply(lambda x : x.split()[0])
df_train['year'] = df_train['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').year)
df_train['month'] = df_train['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').month)
df_train['date'] = df_train['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').day)
df_train['hour'] = df_train['datetime'].apply(lambda x : x.split()[1].split(":")[0])
df_train = df_train.drop(['datetime', 'yyyymmdd'], axis = 1)
df_test['yyyymmdd'] = df_test['datetime'].apply(lambda x : x.split()[0])
df_test['year'] = df_test['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').year)
df_test['month'] = df_test['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').month)
df_test['date'] = df_test['yyyymmdd'].apply(lambda dateString : datetime.strptime(dateString,'%Y-%m-%d').day)
df_test['hour'] = df_test['datetime'].apply(lambda x : x.split()[1].split(":")[0])
df_test = df_test.drop(['datetime', 'yyyymmdd'], axis = 1)
week = []
for i in df_train['date']:
    if i < 8:
        week.append(1)
    elif i >= 8 and i < 16:
        week.append(2)
    elif i >=16 and i < 22:
        week.append(3)
    else:
        week.append(4)
df_train['week'] = week
week = []
for i in df_test['date']:
    if i < 8:
        week.append(1)
    elif i >= 8 and i < 16:
        week.append(2)
    elif i >=16 and i < 22:
        week.append(3)
    else:
        week.append(4)
df_test['week'] = week
df_train['hour'] = df_train['hour'].astype('object').astype(int)
df_test['hour'] = df_test['hour'].astype('object').astype(int)
df_train.columns
df_test.columns
data_types_train = pd.DataFrame(df_train.dtypes, columns = ['Train'])
data_types_test = pd.DataFrame(df_test.dtypes, columns = ['Test'])
data_types = pd.concat([data_types_train, data_types_test], axis = 1)
data_types
df_train.head()
df_test.head()
df_train[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']].describe()
df_test[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']].describe()
fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(df_train.corr(), vmax = 1, vmin = -1, square = False, annot = True)
df_train.groupby('season')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Season', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train.groupby('holiday')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Holiday', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train.groupby('workingday')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Working Day', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train.groupby('weather')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Weather', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.regplot(x = df_train['atemp'], y = df_train['count'], line_kws = {'color': 'red'})
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Actual Temperature', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.regplot(x = df_train['humidity'], y = df_train['count'], line_kws = {'color': 'red'})
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Humidity', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.regplot(x = df_train['windspeed'], y = df_train['count'], line_kws = {'color': 'red'})
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Wind Speed', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train.groupby('year')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
new_df = df_train[df_train['month'] < 7]
new_df.groupby('year')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train.groupby('month')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train.groupby('week')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train.groupby('hour')['count'].sum().plot.bar()
plt.xticks(rotation = 0, fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Hour', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.boxplot(y = df_train['atemp'])
plt.title('Train Actual Temperature')
plt.show()
sns.boxplot(y = df_test['atemp'])
plt.title('Test Actual Temperature')
plt.show()
sns.distplot(df_train['atemp'])
plt.title('Train Actual Temperature')
plt.show()
sns.distplot(df_test['atemp'])
plt.title('Test Actual Temperature')
plt.show()
sns.boxplot(y = df_train['humidity'])
plt.title('Train Humidity')
plt.show()
sns.boxplot(y = df_test['humidity'])
plt.title('Test Humidity')
plt.show()
sns.distplot(df_train['humidity'])
plt.title('Train Humidity')
plt.show()
sns.distplot(df_test['humidity'])
plt.title('Test Humidity')
plt.show()
sns.boxplot(y = df_train['windspeed'])
plt.title('Train Wind Speed')
plt.show()
sns.boxplot(y = df_test['windspeed'])
plt.title('Test Wind Speed')
plt.show()
sns.distplot(df_train['windspeed'])
plt.title('Train Wind Speed')
plt.show()
sns.distplot(df_test['windspeed'])
plt.title('Test Wind Speed')
plt.show()
df_train['windspeed'] = df_train['windspeed'] ** (1/2)
df_test['windspeed'] = df_test['windspeed'] ** (1/2)
sns.distplot(df_train['windspeed'])
plt.title('Train Wind Speed')
plt.show()
sns.distplot(df_test['windspeed'])
plt.title('Test Wind Speed')
plt.show()
wind_speed_train = []
for i in df_train['windspeed']:
    if i < (df_train['windspeed'].mean() - (2 * df_train['windspeed'].std())):
        wind_speed_train.append(df_train['windspeed'].mean() - (2 * df_train['windspeed'].std()))
    elif i > (df_train['windspeed'].mean() + (2 * df_train['windspeed'].std())):
        wind_speed_train.append(df_train['windspeed'].mean() + (2 * df_train['windspeed'].std()))
    else:
        wind_speed_train.append(i)
df_train['windspeed'] = wind_speed_train
wind_speed_test = []
for j in df_test['windspeed']:
    if j < (df_test['windspeed'].mean() - (2 * df_test['windspeed'].std())):
        wind_speed_test.append(df_test['windspeed'].mean() - (2 * df_test['windspeed'].std()))
    elif j > (df_test['windspeed'].mean() + (2 * df_test['windspeed'].std())):
        wind_speed_test.append(df_test['windspeed'].mean() + (2 * df_test['windspeed'].std()))
    else:
        wind_speed_test.append(j)
df_test['windspeed'] = wind_speed_test
sns.boxplot(y = df_train['count'])
plt.title('Train Count')
plt.show()
sns.distplot(df_train['count'])
plt.title('Train Count')
plt.show()
df_train['count'] = df_train['count'] ** (1/3)
sns.distplot(df_train['count'])
plt.title('Train Count')
plt.show()
count_train = []
for i in df_train['count']:
    if i < (df_train['windspeed'].mean() - (2 * df_train['windspeed'].std())):
        count_train.append(df_train['windspeed'].mean() - (2 * df_train['windspeed'].std()))
    elif i > (df_train['windspeed'].mean() + (2 * df_train['windspeed'].std())):
        count_train.append(df_train['windspeed'].mean() + (2 * df_train['windspeed'].std()))
    else:
        count_train.append(i)
df_train['count'] = count_train
df_train_scaled = pd.DataFrame(StandardScaler().fit_transform(df_train.drop('count', axis = 1)), columns = df_test.columns)
df_test_scaled = pd.DataFrame(StandardScaler().fit_transform(df_test), columns = df_test.columns)
df_train_scaled.head()
df_test_scaled.head()
pca_columns = []
for i in range(df_train_scaled.shape[1]):
    pca_columns.append('PC' + str(i+1))
pca_model = PCA()
pca_model.fit(df_train_scaled)
df_pca_train = pd.DataFrame(pca_model.transform(df_train_scaled), columns = pca_columns)
explained_info_train = pd.DataFrame(pca_model.explained_variance_ratio_, columns=['Explained Info']).sort_values(by = 'Explained Info', ascending = False)
imp = []
for i in range(explained_info_train.shape[0]):
    imp.append(explained_info_train.head(i).sum())
explained_info_train_sum = pd.DataFrame()
explained_info_train_sum['Variable'] = pca_columns
explained_info_train_sum['Importance'] = imp
explained_info_train_sum
pca_columns = []
for i in range(8):
    pca_columns.append('PC' + str(i+1))
pca_model = PCA(n_components = 8)
pca_model.fit(df_train_scaled)
df_pca_train = pd.DataFrame(pca_model.transform(df_train_scaled), columns = pca_columns)
df_pca_train.head()
pca_model = PCA(n_components = 8)
pca_model.fit(df_test_scaled)
df_pca_test = pd.DataFrame(pca_model.transform(df_test_scaled), columns = pca_columns)
X = df_pca_train
y = df_train['count']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 17)
X_test = df_pca_test
X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm)
lr = lr.fit()
print(lr.summary())
X_val_sm = sm.add_constant(X_val)
y_pred_val = lr.predict(X_val_sm)
r2_score(y_val, y_pred_val)
models = [LinearRegression(), Lasso(), Ridge(), DecisionTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), KNeighborsRegressor(), SVR(), XGBRegressor()]
model_names = ['LinearRegression', 'Lasso', 'Ridge', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'KNeighborsRegressor', 'SVR', 'XGBRegressor']
r2_train = []
r2_val = []
for model in models:
    mod = model
    mod.fit(X_train, y_train)
    y_pred_train = mod.predict(X_train)
    y_pred_train = y_pred_train.clip(0)
    y_pred_val = mod.predict(X_val)
    y_pred_val = y_pred_val.clip(0)
    r2_train.append(r2_score(y_train, y_pred_train))
    r2_val.append(r2_score(y_val, y_pred_val))
data = {'Modelling Algorithm' : model_names, 'Train R2' : r2_train, 'Validation R2' : r2_val}
data = pd.DataFrame(data)
data['Difference'] = ((np.abs(data['Train R2'] - data['Validation R2'])) * 100)/(data['Train R2'])
data.sort_values(by = 'Validation R2', ascending = False)
svr = SVR()
possible_parameter_values = {'gamma' : [float(x)/10000 for x in range(100001)],
                             'C' : [float(x)/10 for x in range(1001)]}
svr_rs_cv = RandomizedSearchCV(estimator = svr, param_distributions = possible_parameter_values, cv = 10, scoring = 'r2')
svr_rs_cv.fit(X_train, y_train)
svr_rs_cv.best_params_
svr_rs_cv.best_score_
svr = SVR(gamma = 0.0516, C = 84.2)
svr.fit(X_train, y_train)
y_pred_val = svr.predict(X_val)
y_pred_val = y_pred_val.clip(0)
np.sqrt(mean_squared_log_error(y_val, y_pred_val))
svr = SVR(gamma = 0.0516, C = 84.2)
svr.fit(X_train, y_train)
y_pred_test = svr.predict(X_test)
y_pred_test = y_pred_test.clip(0)
y_pred_test = (y_pred_test) ** 3
y_pred_test = pd.DataFrame(y_pred_test, columns = ['Predicted Counts'])
y_pred_test = y_pred_test.round(decimals = 0)
y_pred_test['Predicted Counts'] = y_pred_test['Predicted Counts'].astype('float').astype(int)
y_pred_test.head()
y_pred_test.to_csv('Prediction.csv')
pca_columns = []
for i in range(8):
    pca_columns.append('PC' + str(i+1))
org_var = pd.DataFrame(pca_model.components_, index = pca_columns, columns = df_train_scaled.columns)
values = []
for i in org_var.columns:
    values.append(org_var[i].sum())
dep_var = pd.DataFrame()
dep_var['Variables'] = df_train_scaled.columns
dep_var['Values'] = values
dep_var.sort_values(by = 'Values', ascending = False)