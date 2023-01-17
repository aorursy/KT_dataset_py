import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR



%matplotlib inline



plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12,5)
filepath = '../input/train.csv'

df = pd.read_csv(filepath, sep=',')

df.head()
df = df.replace({'StateHoliday': {'0': 0, 'a': 1, 'b': 2, 'c': 3}})
filepath = '../input/store.csv'

df_store = pd.read_csv(filepath, sep=',')

df_store.head()
df_store = df_store.replace({'StoreType': {'a': 0, 'b': 1, 'c': 2, 'd': 3}})

df_store = df_store.replace({'Assortment': {'a': 0, 'b': 1, 'c': 2}})
df_full = df.merge(df_store, on='Store', how='left')

df_full.head()
filepath = '../input/test.csv'

df_test = pd.read_csv(filepath, sep=',').set_index('Id')

df_test.head()
df_test = df_test.replace({'StateHoliday': {'0': 0, 'a': 1, 'b': 2, 'c': 3}})
df_test.isnull().loc[:,'Open'].value_counts()
df_test.Open.fillna(value=1, inplace=True)

df_test['Open'] = df_test['Open'].astype(int)

df_test['Open'].unique()
df_test_full = df_test.merge(df_store, on='Store', how='left')

df_test_full.head()
del df

del df_store

del df_test
filepath = '../input/sample_submission.csv'

df_out = pd.read_csv(filepath, sep=',')

print(len(df_out))

df_out.head()
#train

df_full['Year'] = df_full['Date'].apply(lambda x: int(x[:4]))

df_full['Month'] = df_full['Date'].apply(lambda x: int(x[5:7]))

df_full['Day'] = df_full['Date'].apply(lambda x: int(x[8:10]))



#test

df_test_full['Year'] = df_test_full['Date'].apply(lambda x: int(x[:4]))

df_test_full['Month'] = df_test_full['Date'].apply(lambda x: int(x[5:7]))

df_test_full['Day'] = df_test_full['Date'].apply(lambda x: int(x[8:10]))
df_full['CompetitionOpen'] = ((df_full['CompetitionOpenSinceYear'] < df_full['Year']) |\

                             ((df_full['CompetitionOpenSinceYear'] == df_full['Year']) &\

                              (df_full['CompetitionOpenSinceMonth'] <= df_full['Month']))).astype(int)



df_test_full['CompetitionOpen'] = ((df_test_full['CompetitionOpenSinceYear'] < df_test_full['Year']) |\

                             ((df_test_full['CompetitionOpenSinceYear'] == df_test_full['Year']) &\

                              (df_test_full['CompetitionOpenSinceMonth'] <= df_test_full['Month']))).astype(int)
df_full['FullDate'] = pd.to_datetime(df_full.Date, format='%Y-%m-%d')

df_test_full['FullDate'] = pd.to_datetime(df_test_full.Date, format='%Y-%m-%d')



import datetime

df_full['WeekNum'] = df_full['FullDate'].apply(func=lambda x: x.isocalendar()[1])

df_test_full['WeekNum'] = df_test_full['FullDate'].apply(func=lambda x: x.isocalendar()[1])
sum_sales = df_full.groupby('Store')['Sales'].sum()

sum_customers = df_full.groupby('Store')['Customers'].sum()

mean_check = sum_sales/sum_customers

df_meancheck = pd.DataFrame(mean_check, columns=['MeanCheck'])

df_meancheck = df_meancheck.reset_index()

df_full = df_full.merge(df_meancheck, on='Store', how='left')

df_test_full = df_test_full.merge(df_meancheck, on='Store', how='left')

del df_meancheck, sum_sales, sum_customers, mean_check
df_full['PromoInterval'].unique()
intervs_list = {'Jan,Apr,Jul,Oct': (1,4,7,10), 'Feb,May,Aug,Nov': (2,5,8,11), 'Mar,Jun,Sept,Dec': (3,6,9,12), np.nan: ()}



#train

df_full['Promo2Now'] = map(lambda week, month, year, promo2, interv, since_week, since_year:

                          int((promo2) == 1 & ((year > since_year) | ((year == since_year) & (week >= since_week))) &\

                           month in intervs_list[interv]), 

                           df_full['WeekNum'], df_full['Month'], df_full['Year'], df_full['Promo2'],

                           df_full['PromoInterval'], df_full['Promo2SinceWeek'], df_full['Promo2SinceYear'])



#test

df_test_full['Promo2Now'] = map(lambda week, month, year, promo2, interv, since_week, since_year:

                          int((promo2) == 1 & ((year > since_year) | ((year == since_year) & (week >= since_week))) &\

                           month in intervs_list[interv]), 

                           df_test_full['WeekNum'], df_test_full['Month'], df_test_full['Year'], 

                           df_test_full['Promo2'], df_test_full['PromoInterval'], df_test_full['Promo2SinceWeek'], 

                           df_test_full['Promo2SinceYear'])
df_full.head().T
columns = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

           'CompetitionDistance', 'CompetitionOpenSinceYear', 'Year', 'Month', 'Day',

           'WeekNum', 'MeanCheck']

X = np.array(df_full.loc[:,columns]).astype(int)

y = np.array(df_full.loc[:,'Sales']).astype(int)

X_test = np.array(df_test_full.loc[:,columns]).astype(int)
#попробуем предсказывать число покупателей, а прибыль получать из среднего чека

columns = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

           'CompetitionDistance', 'CompetitionOpenSinceYear', 'Year', 'Month', 'Day',

           'WeekNum', 'MeanCheck']

X = np.array(df_full.loc[:,columns]).astype(int)

y = np.array(df_full.loc[:,'Customers']).astype(int)

X_test = np.array(df_test_full.loc[:,columns]).astype(int)
cls = RandomForestRegressor(n_estimators=30, criterion='mse').fit(X, y)
score = cls.predict(X_test)
score_final = score*df_test_full.MeanCheck
features = {}

for i in range(len(columns)):

    features[columns[i]] = cls.feature_importances_[i]
import operator

sorted_features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)

sorted_features
#%%timeit

#boost = GradientBoostingClassifier(n_estimators=50).fit(X[:100], y[:100])
df_full_linear = df_full.copy()

day_of_week = OneHotEncoder().fit_transform(df_full_linear['DayOfWeek'].values.reshape(-1, 1)).toarray().astype(int)

store_type = OneHotEncoder().fit_transform(df_full_linear['StoreType'].values.reshape(-1, 1)).toarray().astype(int)

assortment = OneHotEncoder().fit_transform(df_full_linear['Assortment'].values.reshape(-1, 1)).toarray().astype(int)



for i in range(len(day_of_week[0])):

    df_full_linear['day_of_week_{}'.format(i)] = day_of_week[:,i]

    

for i in range(len(store_type[0])):

    df_full_linear['store_type_{}'.format(i)] = store_type[:,i]

    

for i in range(len(assortment[0])):

    df_full_linear['assortment_{}'.format(i)] = assortment[:,i]

    

df_full_linear.drop(df_full_linear[['DayOfWeek', 'StoreType', 'Assortment']], axis=1, inplace=True)
df_full_linear.head().T
df_full_linear.info()
columns = ['Store', 'Promo', 'StateHoliday', 'SchoolHoliday', 'CompetitionDistance', 'Promo2',

          'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5',

          'day_of_week_6', 'store_type_0', 'store_type_1', 'store_type_2', 'store_type_3',

          'assortment_0', 'assortment_1', 'assortment_2', 'MeanCheck']

X_linear = np.array(df_full_linear.loc[df_full_linear.Open == 1,columns])

y_linear = np.array(df_full_linear.loc[df_full_linear.Open == 1,'Sales'])
X_linear.shape
np.unique(np.isnan(X_linear[:,4]))
X_linear[np.isnan(X_linear[:,4]),4] = 0
np.unique(np.isnan(X_linear[:,4]))
X_linear = StandardScaler().fit_transform(X_linear)

#y_linear = (y_linear - y_linear.mean())/y_linear.std()
y_linear[0]
df_linear_test = df_test_full.copy()

day_of_week = OneHotEncoder().fit_transform(df_linear_test['DayOfWeek'].values.reshape(-1, 1)).toarray().astype(int)

store_type = OneHotEncoder().fit_transform(df_linear_test['StoreType'].values.reshape(-1, 1)).toarray().astype(int)

assortment = OneHotEncoder().fit_transform(df_linear_test['Assortment'].values.reshape(-1, 1)).toarray().astype(int)



for i in range(len(day_of_week[0])):

    df_linear_test['day_of_week_{}'.format(i)] = day_of_week[:,i]

    

for i in range(len(store_type[0])):

    df_linear_test['store_type_{}'.format(i)] = store_type[:,i]

    

for i in range(len(assortment[0])):

    df_linear_test['assortment_{}'.format(i)] = assortment[:,i]

    

df_linear_test.drop(df_linear_test[['DayOfWeek', 'StoreType', 'Assortment']], axis=1, inplace=True)



df_linear_test.loc[:, 'CompetitionDistance'] = (df_linear_test.loc[:, 'CompetitionDistance'] \

                                                - df_linear_test.loc[:, 'CompetitionDistance'].mean()).div(\

                                                 df_linear_test.loc[:, 'CompetitionDistance'].std())



columns = ['Store', 'Promo', 'StateHoliday', 'SchoolHoliday', 'CompetitionDistance', 'Promo2',

          'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5',

          'day_of_week_6', 'store_type_0', 'store_type_1', 'store_type_2', 'store_type_3',

          'assortment_0', 'assortment_1', 'assortment_2', 'MeanCheck']

X_linear_test = np.array(df_linear_test.loc[:,columns])



X_linear_test[np.isnan(X_linear_test[:,4]),4] = 0

X_linear_test = StandardScaler().fit_transform(X_linear_test)
lin_reg = LinearRegression().fit(X_linear, y_linear)
ridge = Ridge(alpha=1.0).fit(X_linear, y_linear)
lin_reg.score(X_linear, y_linear)
ridge.score(X_linear, y_linear)
score = ridge.predict(X_linear_test)
ridge.coef_
score[X_linear_test[:,1] == 0] = 0
#%%timeit

#svr = SVR().fit(X_linear[:10000], y_linear[:10000])

#svr = SVR().fit(X_linear[750000:], y_linear[750000:])

#svr.score(X_linear[750000:], y_linear[750000:])

#score = svr.predict(X_linear_test)

#score[df_test.Open == 0] = 0
df_out = pd.DataFrame(list(score_final), columns=['Sales'])

df_out = df_out.reset_index().rename(index=str, columns={'index': 'Id'})

df_out.loc[:,'Id'] += 1

df_out.head()
df_out['Sales'].isnull().unique()
filepath = 'rossman.csv'

df_out.to_csv(filepath, index=False)
filepath = 'rossman.csv'

df_out = pd.read_csv(filepath)

df_out.head()
len(df_out)