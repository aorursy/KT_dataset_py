import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime, date

import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.kernel_ridge import KernelRidge



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('train data shape: ', train.shape)

print('test data shape: ', test.shape)

print('feature columns not in test data: ', set(train.columns[:-1]).difference(set(test.columns)))
train.isnull().values.any(), test.isnull().values.any()
train.dtypes
train.describe()
def transform_time(d):

    d['hour'] = d['datetime'].apply(lambda x: int(x[11:13]))

    d['date'] = d['datetime'].apply(lambda x: x[:10])

    d['weekday'] = d['date'].apply(lambda s: date(*(int(i) for i in s.split('-'))).weekday() + 1)

    d['month'] = d['date'].apply(lambda s: int(s[5:7]))

    d['day'] = d['date'].apply(lambda s: int(s[8:10]))

    d['year'] = d['date'].apply(lambda s: int(s[:4]))

    # 注意monday是0， sunday是6，所以最后加1



transform_time(train)

train.head()
train.drop(['datetime', 'date'], axis=1, inplace=True)
from sklearn.feature_selection import mutual_info_regression



df_corr = pd.DataFrame(train.corr()['count'])

df_corr.columns = ['corr_coef']

mut_reg = mutual_info_regression(train, train['count'])

df_corr['mut_reg'] = mut_reg.tolist()

df_corr.sort_values(by='corr_coef', ascending=False)
from scipy.stats import norm, skew

plt.subplots(1, 2, figsize=(10,4))

plt.subplot(1, 2, 1)

sns.distplot(train['count'], fit=norm);

plt.subplot(1, 2, 2)

sns.distplot(np.log2(train['count']), fit=norm);

print('skewness before transformation', np.abs(skew(train['count'])))

# transform count

train['count'] = np.log2(train['count'] + 1)

print('skewness after transformation', np.abs(skew(train['count'])))
sns.distplot(train.loc[train['season']==1]['count'], label='1')

sns.distplot(train.loc[train['season']==2]['count'], label='2')

sns.distplot(train.loc[train['season']==3]['count'], label='3')

plt.legend()
sns.boxplot('season', 'count', data=train);
sns.boxplot('holiday', 'count', data=train);
plt.subplots(1, 4, figsize=(20,4))

plt.subplot(1, 4, 1)

sns.boxplot('holiday', 'count', hue='season',  data=train.loc[train['season']==1]);



plt.subplot(1, 4, 2)

sns.boxplot('holiday', 'count', hue='season',  data=train.loc[train['season']==2]);



plt.subplot(1, 4, 3)

sns.boxplot('holiday', 'count', hue='season',  data=train.loc[train['season']==3]);



plt.subplot(1, 4, 4)

sns.boxplot('holiday', 'count', hue='season',  data=train.loc[train['season']==4]);



train.groupby(['holiday','season'])['count'].median().unstack()
sns.lineplot('holiday', 'count', hue='season', data=train, estimator=np.median, markers=True,palette="ch:2.5,.25");
sns.boxplot('workingday', 'count', data=train);
sns.distplot(train.loc[train['workingday']==0]['count'], label='0')

sns.distplot(train.loc[train['workingday']==1]['count'], label='1')



plt.legend()
train.groupby(['workingday', 'holiday'])['count'].median().unstack()
sns.boxplot('workingday', 'count', data=train, hue='holiday');
sns.lineplot('workingday', 'count', data=train, hue='season', estimator=np.median);
sns.boxplot('weather','count', data=train)
sns.boxplot('season','count', data=train, hue='weather')
train.groupby(['weather', 'season'])['count'].median().unstack()
sns.lineplot('season', 'count', data=train, hue='weather');
tempbin = pd.qcut(train['temp'], 5)

sns.boxplot(tempbin, 'count', data=train)
tempbin = pd.qcut(train['temp'], 5)

sns.boxplot(tempbin, 'count', data=train, hue='season');
train.groupby(['season', pd.qcut(train['temp'], 5)])['count'].size().unstack()
plt.subplot(1,2,1)

sns.scatterplot('temp', 'atemp', data=train);

plt.subplot(1,2,2)

sns.scatterplot('temp', 'atemp', data=test);
df_unusual = test.loc[(test['atemp']<15) & (test['temp']>20)]

plt.subplot(1,2,1)

plt.plot(df_unusual.atemp)

plt.subplot(1,2,2)

plt.plot(df_unusual.temp)

df_unusual
humidbin = pd.qcut(train['humidity'], 5)

sns.boxplot(humidbin, 'count', data=train)
humidbin = pd.qcut(train['humidity'], 5)

sns.boxplot(tempbin, 'count', data=train, hue=humidbin)
windbin = pd.qcut(train['windspeed'], 8)

sns.boxplot(windbin, 'count', data=train)

plt.xticks(rotation=90);
train['bi_humid'] = 0

train.loc[train['humidity'] <=15, 'bi_humid'] = 1
sns.boxplot('hour', 'count', data=train)
sns.lineplot('weekday', 'count', data=train, hue='holiday')
train.loc[(train.holiday==0) & (train.weekday<=5)]['workingday'].unique(), train.loc[ (train.weekday>=6) ]['workingday'].unique()
sns.boxplot('month', 'count', data=train, hue='workingday')


train['month_working'] = 0

train.loc[(train['month']>4) & (train['workingday']==1), 'month_working'] = 0

train.loc[(train['month']>4) & (train['workingday']==0), 'month_working'] = 1

train.loc[(train['month']<=4) & (train['workingday']==1), 'month_working'] = -1

train.loc[(train['month']<=4) & (train['workingday']==0), 'month_working'] = -2
sns.boxplot('day', 'count', data=train)
sns.boxplot('year', 'count', data=train)
df_data = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train = df_data.iloc[:, :-3]

df_label = df_data['count']

train_ohc = pd.DataFrame()

test_ohc = pd.DataFrame()



X = pd.concat([df_train, df_test]).reset_index(drop=True)

# log label

df_label = np.log2(df_data['count'] + 1)

# transform tim

transform_time(X)

X.drop(['datetime', 'date', 'atemp', 'day'], axis=1, inplace=True)



X['bi_humid'] = 0



X.loc[X['humidity'] <=15, 'bi_humid'] = 1



X['month_working'] = 0

X.loc[(X['month']>4) & (X['workingday']==1), 'month_working'] = 0

X.loc[(X['month']>4) & (X['workingday']==0), 'month_working'] = 1

X.loc[(X['month']<=4) & (X['workingday']==1), 'month_working'] = -1

X.loc[(X['month']<=4) & (X['workingday']==0), 'month_working'] = -2



X['weekday_holiday'] = 0

X.loc[(X['holiday']==0), 'weekday_holiday'] = 1

X.loc[(X['weekday']<=3) & (X['holiday']==1), 'weekday_holiday'] = 2

X.loc[(X['weekday']>3) & (X['month']<=5) & (X['workingday']==1), 'weekday_holiday'] = 3

X.loc[(X['weekday']>5) & (X['holiday']==1), 'weekday_holiday'] = 4



#X.drop(['weekday'], axis=1, inplace=True)



class_features = [

    'season', 'weather', 'month', 'year', 'hour', 'weekday', 'weekday_holiday', 'month_working'

]



# data_ohc = pd.get_dummies(X, columns=class_features, drop_first=True)

def ohc(data, columns):

    for col in columns:

        temp = pd.get_dummies(X[col], prefix=col+'_', drop_first=True)

        data = pd.concat([data, temp], axis=1)

        data.drop(col, axis=1, inplace=True)

    return data



data_ohc = ohc(X, class_features)

data_ohc.head()
# shuffle and split train and test

# data_ohc = data_ohc.sample(frac=1).reset_index(drop=True)

train_ohc = data_ohc[:df_train.shape[0]]

test_ohc = data_ohc[df_train.shape[0]:]





from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()

Xtrain_ohc = scaler.fit_transform(train_ohc)

Xtest_ohc = scaler.transform(test_ohc)



# to dataframe with column names

train_ohc = pd.DataFrame(Xtrain_ohc, columns=train_ohc.columns)

test_ohc = pd.DataFrame(Xtest_ohc, columns=test_ohc.columns)



train_ohc['count'] = df_label

all_features = test_ohc.columns
# try linear regression

model_base = LinearRegression()

model_base.fit(train_ohc[all_features], train_ohc['count'])

kf = KFold(n_splits=3, shuffle=True, random_state=42)

cv = np.sqrt(-cross_val_score(model_base, train_ohc[all_features], train_ohc['count'], cv=kf, scoring='neg_mean_squared_error').mean())

cv
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(train_ohc[all_features], train_ohc['count'])

kf = KFold(n_splits=3, shuffle=True, random_state=42)

cv = np.sqrt(-cross_val_score(dt, train_ohc[all_features], train_ohc['count'], cv=kf, scoring='neg_mean_squared_error').mean())

cv
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)

X = poly.fit_transform(train_ohc[all_features])

y = train_ohc['count']
all_features_after_poly = poly.get_feature_names(all_features)
from sklearn.linear_model import Ridge, Lasso

def try_model1(features, data=train, target='count', alpha=1, degree=3):

    X = data

    y = target

    

    # model_lin = KernelRidge(alpha=alpha, kernel='polynomial', degree=degree)

    model_lin = Lasso(alpha=0.1)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    cv1 = np.sqrt(-cross_val_score(model_lin, X, y, cv=kf, scoring='neg_mean_squared_error').mean())

#     cv2 = -cross_val_score(model_ker, X, y, cv=kf, scoring='neg_mean_squared_error').mean()

    print('average cross validation score', cv1)

    return model_lin

#     print('Kernel Ridge', cv2)





model = try_model1(data=X, features=all_features, target=y)
model = Lasso(alpha=0.1)

model.fit(X, y)
feature_imp = pd.DataFrame({'feature':all_features_after_poly, 'weight':model.coef_}).sort_values(by='weight', ascending=False)
feature_imp[np.abs(feature_imp['weight']) > 0.03]
# from sklearn.model_selection import GridSearchCV



# gs = GridSearchCV(KernelRidge(kernel='polynomial'), scoring="neg_mean_squared_error", cv=3, verbose=3,

#                   param_grid={"degree": [i for i in range(1, 6)], 

#                               "alpha":[i*0.1 for i in range(1, 11)]}, )

# gs.fit(train_ohc[test_ohc.columns], train_ohc['count'])
# gs.best_params_, gs.best_score_
# model = KernelRidge(alpha=1., kernel='polynomial', degree=3)

# model.fit(train_ohc[test_ohc.columns], train_ohc['count'])



pred = model.predict(poly.transform(test_ohc))

# transform pred back np.log2(df_data['count'] + 1)

pred = 2 ** pred - 1

# save result

pd.DataFrame({'datetime': df_test.datetime, 'count': pred}).to_csv('submission.csv', index=False, header=True)