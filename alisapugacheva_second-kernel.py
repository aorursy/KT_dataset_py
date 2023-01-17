import warnings
import gc
warnings.filterwarnings("ignore")
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
#rcParams[]
import seaborn as sns
import pickle
rcParams['figure.figsize'] = 15, 10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
import statsmodels.stats.api as ssm
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, log_loss, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
features = pd.read_csv('../input/Features data set.csv', parse_dates=['Date'], dayfirst=True)

sales = pd.read_csv('../input/sales data-set.csv', parse_dates=['Date'], dayfirst=True)

stores = pd.read_csv('../input/stores data-set.csv') #("../input")
features.shape, sales.shape, stores.shape
print('Количество уникальных записей по Store и Date: {}'.format(len(sales.groupby(['Date', 'Store']).groups)))

sales.head()

print('Количество уникальных записей по Store и Date: {}'.format(len(features.groupby(['Date', 'Store']).groups)))

features.head()
features.tail()
stores.head()
arr =[]
for d in sales.Date.unique():
    for dept in sales.Dept.unique():
        for s in sales.Store.unique():
            arr.append([d, dept, s])
all_values = pd.DataFrame.from_records(arr, columns=['Date', 'Dept', 'Store'])
whole_sales = pd.merge(all_values, sales.drop('IsHoliday', axis=1), on=['Date', 'Dept', 'Store'], how = 'left')
whole_sales.Weekly_Sales.fillna(0, inplace=True)
whole_sales.info()
whole_sales.shape
def create_new_features(whole_sales, features, stores):
    
    datas = pd.to_datetime(pd.unique(sales.Date), format="%d/%m/%Y")
    
    date_dict = {k:v for k,v in zip(datas, datas.week)}

    whole_sales['week'] = whole_sales.Date.apply(lambda x: date_dict[x])

    whole_sales['year'] = whole_sales.Date.apply(lambda x: x.year)

    whole_sales['month'] = whole_sales.Date.apply(lambda x: x.month)

    whole_sales['day'] = whole_sales.Date.apply(lambda x: x.day)
    
    # Флаги пред- и постпраздничных недель
    df_of_holidays = sales.groupby(['Date', 'IsHoliday']).Store.nunique().reset_index()[['Date', 'IsHoliday']]

    df_of_holidays['post_holiday'] = df_of_holidays.IsHoliday.shift(1)
    df_of_holidays['pred_holiday'] = df_of_holidays.IsHoliday.shift(-1)
    df_of_holidays = df_of_holidays.fillna(False)

    whole_sales = pd.merge(whole_sales, df_of_holidays.drop('IsHoliday', axis=1), on='Date')
    whole_data_with_markdown = whole_sales.merge(features, on=['Date', 'Store'])
    whole_data_with_markdown = whole_data_with_markdown.merge(stores, on=['Store'])
    
    #закодируем булевы переменные
    encoder = LabelEncoder()
    
    whole_data_with_markdown['Type'] = encoder.fit_transform(whole_data_with_markdown['Type'])

    whole_data_with_markdown['IsHoliday'] = encoder.fit_transform(whole_data_with_markdown['IsHoliday'])
    whole_data_with_markdown['post_holiday'] = encoder.fit_transform(whole_data_with_markdown['IsHoliday'])
    whole_data_with_markdown['pred_holiday'] = encoder.fit_transform(whole_data_with_markdown['IsHoliday'])
    
    return whole_data_with_markdown
whole_data_with_markdown = create_new_features(whole_sales, features, stores)
whole_data_with_markdown.head()
whole_data_with_markdown.info()
fig, ax = plt.subplots()
whole_data_with_markdown.groupby('Date').Weekly_Sales.mean().plot(x='Date', y='Weekly_Sales', ax=ax);
ax.xticks = whole_data_with_markdown.groupby('Date')
plt.show()
df_stores_depts = whole_data_with_markdown.groupby(['Store', 'Dept']).sum().reset_index()

df_stores_depts = df_stores_depts.pivot(index='Store', columns='Dept', values='Weekly_Sales')


ax = sns.heatmap(df_stores_depts.apply(lambda col: (col-min(col))/(max(col)-min(col)), axis=0), cbar_kws={'label': 'Normalized Sale'})
whole_data_with_markdown_new = whole_data_with_markdown[(whole_data_with_markdown.Date >= pd.datetime(2011,11,5))]
markdown_cols = [x for x in features.columns if 'MarkDown' in x]
whole_data_with_markdown_new['markdown_sum'] = whole_data_with_markdown[markdown_cols].sum(axis=1)
whole_data_with_markdown_new.head()
whole_data_with_markdown_new.groupby(['Date', 'Store']).markdown_sum.nunique().reset_index().groupby('markdown_sum').Store.count()
average_promo_sales = whole_data_with_markdown_new.groupby(['Date'])['Weekly_Sales','markdown_sum'].mean().reset_index()
fig, (axis1, axis2) = plt.subplots(2,1)
ax1 = average_promo_sales.plot(y='Weekly_Sales', x='Date', legend=True,ax=axis1,marker='o',title="Average Sales")
ax2 = average_promo_sales.plot(y='markdown_sum', x='Date', legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Average Promo")
fig, axis = plt.subplots(1,2)

whole_data_with_markdown.Weekly_Sales.hist(bins=20, ax=axis[0])

whole_data_with_markdown.Weekly_Sales.apply(lambda x: 0 if x<=0 else np.log(x)).hist(bins=20, ax=axis[1])
whole_data_with_markdown.Weekly_Sales = whole_data_with_markdown.Weekly_Sales.apply(lambda x: 0 if x==0 else np.log(x))
whole_data_with_markdown.info()
corrmat = whole_data_with_markdown.corr()
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
whole_data = whole_data_with_markdown.drop(markdown_cols, axis=1)
whole_data.head()
whole_data.info()
with open('whole_data.pkl', 'wb') as f:
    pickle.dump(whole_data, f)
def train_test_spl(whole_data, test_len=16):
    whole_data = whole_data.fillna(0)
    whole_data = whole_data.sort_values(by='Date')
    
    unique_date = whole_data.Date.unique()[-test_len:]
    
    train = whole_data[~whole_data.Date.isin(unique_date)]
    test = whole_data[whole_data.Date.isin(unique_date)]
    
    return [train, test]
train, test = train_test_spl(whole_data, 16)
train_m, test_m = train_test_spl(whole_data_with_markdown_new, 16)
cv_splits = TimeSeriesSplit(n_splits=10)
gc.collect()
estimators =[RandomForestRegressor(n_estimators=100, max_features ='sqrt'),
             KNeighborsRegressor(n_neighbors=6),
             ExtraTreesRegressor(n_estimators=20, criterion='mse', bootstrap=True, n_jobs=-1, random_state=17)
            ]
def plot_scores(test, group_cols = 'Date'):
    
    for col in group_cols:
        if col not in test.columns:
            return 'group columns not exist in test dataframe'
    
    ttest = test.groupby(by=group_cols)[['Weekly_Sales', 'predict_y']].mean()

    fig, ax = plt.subplots()
    ttest.plot(y='Weekly_Sales', ax=ax)
    ttest.plot(y='predict_y', ax=ax)
    ax.set_title('Mean squared error: {}. r2-score : {}'.format(mean_squared_error(test.Weekly_Sales, test.predict_y), r2_score(test.Weekly_Sales, test.predict_y)))
    plt.show()
def scale_set(x):
    scaler = StandardScaler()
    return scaler.fit_transform(x)
def x_y_split(x_y):
    X = x_y.drop(['Weekly_Sales', 'Date'], axis=1)
    y = x_y['Weekly_Sales']
    return [X,y]
scores = pd.DataFrame()
tmp = {}
for m,est in zip(['RandomForestRegressor', 'KNeighborsRegressor', 'ExtraTreesRegressor'], estimators):
    tmp['Model'] = m
    for j,i in enumerate([train, train_m]):
        X_train, y_train = x_y_split(i)
        cv_scores = cross_val_score(est, scale_set(X_train), y_train, cv=cv_splits, scoring='r2')
        tmp['R2_Y%s'%str(j+1)] = np.mean(cv_scores)
    scores = scores.append([tmp])
    scores.set_index('Model', inplace=True)
fig, axes = plt.subplots(ncols=2, figsize=(10,4))
scores.R2_Y1.plot(ax=axes[0], kind='bar', title='R2_Y1')
scores.R2_Y2.plot(ax=axes[1], kind='bar', color='green', title='R2_Y2')
etr = ExtraTreesRegressor(n_estimators=20, criterion='mse', bootstrap=True, n_jobs=-1, random_state=17)
X_train, y_train = x_y_split(train)
X_test, y_test = x_y_split(test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
etr.fit(X_train, y_train)
test['Weekly_Sales'] = test.Weekly_Sales.apply(lambda x: np.e**x)
pred_y = list(map(lambda x: np.e**x, etr.predict(X_test)))
plot_scores(test.assign(predict_y=pred_y), ['Date', 'Store'])
plot_scores(test.assign(predict_y = pred_y), ['Date', 'Dept'])

plot_scores(test.assign(predict_y=pred_y), ['Date'])
plot_scores(test.assign(predict_y=pred_y)[test.Store == 20], ['Date', 'Store'])