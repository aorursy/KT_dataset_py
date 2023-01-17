# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



from sklearn import model_selection, preprocessing

import xgboost as xgb



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",

"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]



#train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'], index_col='id')

train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

#test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'], index_col='id')

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')

test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')



#fx = pd.read_excel('../input/BAD_ADDRESS_FIX.xlsx').drop_duplicates('id').set_index('id')



#train_df.update(fx, overwrite=True)

#test_df.update(fx, overwrite=True)



train_df.head()
fx = pd.read_excel('../input/BAD_ADDRESS_FIX.xlsx').drop_duplicates('id')

fx.head()
fx.shape
train_df.update(fx, overwrite=True)

train_df.head()
train_df.shape
print('Fix in train: ', train_df.index.intersection(fx.index).shape[0])
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



import random

random.seed(1556)



from sklearn import model_selection, preprocessing

import xgboost as xgb







pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",

"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]



train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')

test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')

print(train_df.shape, test_df.shape)



# truncate the extreme values in price_doc #

ulimit = np.percentile(train_df.price_doc.values, 99)

llimit = np.percentile(train_df.price_doc.values, 1)

train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit

train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit



for f in train_df.columns:

    if train_df[f].dtype=='object':

        print(f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))

        train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))

        test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))

        

print("label encoder...")



# year and month #

train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month

test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month



# year and week #

train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear

test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear



# year #

train_df["year"] = train_df["timestamp"].dt.year

test_df["year"] = test_df["timestamp"].dt.year



# month of year #

train_df["month_of_year"] = train_df["timestamp"].dt.month

test_df["month_of_year"] = test_df["timestamp"].dt.month



# week of year #

train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear

test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear



# day of week #

train_df["day_of_week"] = train_df["timestamp"].dt.weekday

test_df["day_of_week"] = test_df["timestamp"].dt.weekday



# ratio of living area to full area #

train_df["ratio_life_sq_full_sq"] = train_df["life_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)

test_df["ratio_life_sq_full_sq"] = test_df["life_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)

train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]<0] = 0

train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]>1] = 1

test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]<0] = 0

test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]>1] = 1



# ratio of kitchen area to living area #

train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"),1)

test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"),1)

train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]<0] = 0

train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]>1] = 1

test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]<0] = 0

test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]>1] = 1



# ratio of kitchen area to full area #

train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)

test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)

train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]<0] = 0

train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]>1] = 1

test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]<0] = 0

test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]>1] = 1



# floor of the house to the total number of floors in the house #

train_df["ratio_floor_max_floor"] = train_df["floor"] / train_df["max_floor"].astype("float")

test_df["ratio_floor_max_floor"] = test_df["floor"] / test_df["max_floor"].astype("float")



# num of floor from top #

train_df["floor_from_top"] = train_df["max_floor"] - train_df["floor"]

test_df["floor_from_top"] = test_df["max_floor"] - test_df["floor"]



train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]

test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]



train_df["age_of_building"] = train_df["build_year"] - train_df["year"]

test_df["age_of_building"] = test_df["build_year"] - test_df["year"]



def add_count(df, group_col):

    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()

    grouped_df.columns = [group_col, "count_"+group_col]

    df = pd.merge(df, grouped_df, on=group_col, how="left")

    return df



train_df = add_count(train_df, "yearmonth")

test_df = add_count(test_df, "yearmonth")



train_df = add_count(train_df, "yearweek")

test_df = add_count(test_df, "yearweek")



train_df["ratio_preschool"] = train_df["children_preschool"] / train_df["preschool_quota"].astype("float")

test_df["ratio_preschool"] = test_df["children_preschool"] / test_df["preschool_quota"].astype("float")



train_df["ratio_school"] = train_df["children_school"] / train_df["school_quota"].astype("float")

test_df["ratio_school"] = test_df["children_school"] / test_df["school_quota"].astype("float")



y_train = train_df["price_doc"]

x_train = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test_df.drop(["id", "timestamp"], axis=1)



test_id=test_df.id



xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)



'''

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)

'''

    

print(" ")

print("training...")

    

#num_boost_rounds = len(cv_output)

num_boost_rounds= 363

print("num_boost_rounds:", num_boost_rounds)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)



y_predict = model.predict(dtest)

output = pd.DataFrame({'id': test_id, 'price_doc': y_predict})



#output.to_csv('xgbSub.csv', index=False)



fig, ax = plt.subplots(1, 1, figsize=(8, 16))



xgb.plot_importance(partial_model, max_num_features=50, height=0.5, ax=ax)

#train_df = pd.read_csv("../input/train.csv")

train_df.shape
train_df.head()
train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

train_df['price_doc_log'] = np.log1p(train_df['price_doc'])
train_df.isnull().sum()
train_df.index
train_na = (train_df.isnull().sum() / len(train_df)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
f, ax = plt.subplots(figsize=(12, 8))

plt.xticks(rotation='90')

sns.barplot(x=train_na.index, y=train_na)

ax.set(title='Percent missing data by feature', ylabel='% missing')
#train_df['state'][:50]

#train_df['state'].unique()

train_df['state'].value_counts()
train_df.loc[train_df['state']==33, 'state'] = train_df['state'].mode().iloc[0]
train_df['state'].value_counts()
train_df['build_year'].value_counts()
train_df.loc[train_df['build_year'] == 20052009, 'build_year'] = 2007

#build_year has an erronus value 20052009. Since its unclear which it should be, let's replace with 2007
train_1900= train_df[train_df['build_year'] < 1900]

train_1900.shape
internal_chars = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'kitch_sq', 'state', 'price_doc']

corrmat = train_df[internal_chars].corr()
f, ax = plt.subplots(figsize=(10, 7))

plt.xticks(rotation='90')

sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
#x= full_sq, y=price_doc

f, ax = plt.subplots(figsize=(10, 7))

plt.scatter(x=train_df['full_sq'], y=train_df['price_doc'], c='r')
f, ax = plt.subplots(figsize=(10, 7))

ind = train_df[train_df['full_sq'] > 2000].index

plt.scatter(x=train_df.drop(ind)['full_sq'], y=train_df.drop(ind)['price_doc'], c='r', alpha=0.5)

ax.set(title='Price by area in sq meters', xlabel='Area', ylabel='Price')
(train_df['life_sq'] > train_df['full_sq']).sum()
f, ax = plt.subplots(figsize=(10, 7))

sns.countplot(x=train_df['num_room'])

ax.set(title='Distribution of room count', xlabel='num_room')
train_df.groupby('product_type')['price_doc'].median()
f, ax = plt.subplots(figsize=(12, 8))

plt.xticks(rotation='90')

ind = train_df[(train_df['build_year'] <= 1691) | (train_df['build_year'] >= 2018)].index

by_df = train_df.drop(ind).sort_values(by=['build_year'])

sns.countplot(x=by_df['build_year'])

ax.set(title='Distribution of build year')
f, ax = plt.subplots(figsize=(12, 6))

by_price = by_df.groupby('build_year')[['build_year', 'price_doc']].mean()

sns.regplot(x="build_year", y="price_doc", data=by_price, scatter=False, order=3, truncate=True)

plt.plot(by_price['build_year'], by_price['price_doc'], color='r')

ax.set(title='Mean price by year of build')
f, ax = plt.subplots(figsize=(12, 6))

ts_df = train_df.groupby('timestamp')[['price_doc']].mean()

#sns.regplot(x="timestamp", y="price_doc", data=ts_df, scatter=False, truncate=True)

plt.plot(ts_df.index, ts_df['price_doc'], color='r', )

ax.set(title='Daily median price over time')
import datetime

import matplotlib.dates as mdates



years = mdates.YearLocator()   # every year

yearsFmt = mdates.DateFormatter('%Y')

ts_vc = train_df['timestamp'].value_counts()



f, ax = plt.subplots(figsize=(12, 6))

plt.bar(left=ts_vc.index, height=ts_vc)

ax.xaxis.set_major_locator(years)

ax.xaxis.set_major_formatter(yearsFmt)

ax.set(title='Sales volume over time', ylabel='Number of transactions')
f, ax = plt.subplots(figsize=(12, 8))

ts_df = train_df.groupby(by=[train_df.timestamp.dt.month])[['price_doc']].median()

plt.plot(ts_df.index, ts_df, color='r')

ax.set(title='Price by month of year')
f, ax = plt.subplots(figsize=(12, 8))

ind = train_df[train_df['state'].isnull()].index

train_df['price_doc_log10'] = np.log10(train_df['price_doc'])

sns.violinplot(x="state", y="price_doc_log10", data=train_df.drop(ind), inner="box")

# sns.swarmplot(x="state", y="price_doc_log10", data=train_df.dropna(), color="w", alpha=.2);

ax.set(title='Log10 of median price by state of home', xlabel='state', ylabel='log10(price)')
train_df.drop(ind).groupby('state')['price_doc'].mean()
f, ax = plt.subplots(figsize=(12, 8))

ind = train_df[train_df['material'].isnull()].index

sns.violinplot(x="material", y="price_doc_log", data=train_df.drop(ind), inner="box")

# sns.swarmplot(x="state", y="price_doc_log10", data=train_df.dropna(), color="w", alpha=.2);

ax.set(title='Distribution of price by build material', xlabel='material', ylabel='log(price)')
f, ax = plt.subplots(figsize=(12, 8))

plt.scatter(x=train_df['floor'], y=train_df['price_doc_log'], c='r', alpha=0.4)

sns.regplot(x="floor", y="price_doc_log", data=train_df, scatter=False, truncate=True)

ax.set(title='Price by floor of home', xlabel='floor', ylabel='log(price)')
f, ax = plt.subplots(figsize=(12, 8))

plt.scatter(x=train_df['max_floor'], y=train_df['price_doc_log'], c='r', alpha=0.4)

sns.regplot(x="max_floor", y="price_doc_log", data=train_df, scatter=False, truncate=True)

ax.set(title='Price by max floor of home', xlabel='max_floor', ylabel='log(price)')
f, ax = plt.subplots(figsize=(12, 8))

plt.scatter(x=train_df['floor'], y=train_df['max_floor'], c='r', alpha=0.4)

plt.plot([0, 80], [0, 80], color='.5')
train_df.loc[train_df['max_floor'] < train_df['floor'], ['id', 'floor','max_floor']].head(20)
## Demographic Characteristics

demo_vars = ['area_m', 'raion_popul', 'full_all', 'male_f', 'female_f', 'young_all', 'young_female', 

             'work_all', 'work_male', 'work_female', 'price_doc']

corrmat = train_df[demo_vars].corr()
f, ax = plt.subplots(figsize=(10, 7))

plt.xticks(rotation='90')

sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
train_df['sub_area'].unique().shape[0]
train_df['area_km'] = train_df['area_m'] / 1000000

train_df['density'] = train_df['raion_popul'] / train_df['area_km']



f, ax = plt.subplots(figsize=(10, 6))

sa_price = train_df.groupby('sub_area')[['density', 'price_doc']].median()

sns.regplot(x="density", y="price_doc", data=sa_price, scatter=True, truncate=True)

ax.set(title='Median home price by raion population density (people per sq. km)')
f, ax = plt.subplots(figsize=(10, 20))

sa_vc = train_df['sub_area'].value_counts()

sa_vc = pd.DataFrame({'sub_area':sa_vc.index, 'count': sa_vc.values})

ax = sns.barplot(x="count", y="sub_area", data=sa_vc, orient="h")

ax.set(title='Number of Transactions by District')

f.tight_layout()
train_df['work_share'] = train_df['work_all'] / train_df['raion_popul']

f, ax = plt.subplots(figsize=(12, 6))

sa_price = train_df.groupby('sub_area')[['work_share', 'price_doc']].mean()

sns.regplot(x="work_share", y="price_doc", data=sa_price, scatter=True, order=4, truncate=True)

ax.set(title='District mean home price by share of working age population')
school_chars = ['children_preschool', 'preschool_quota', 'preschool_education_centers_raion', 'children_school', 

                'school_quota', 'school_education_centers_raion', 'school_education_centers_top_20_raion', 

                'university_top_20_raion', 'additional_education_raion', 'additional_education_km', 'university_km', 'price_doc']

corrmat = train_df[school_chars].corr()
#School Characteristics

f, ax = plt.subplots(figsize=(10, 7))

plt.xticks(rotation='90')

sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
train_df['university_top_20_raion'].unique()
f, ax = plt.subplots(figsize=(12, 8))

sns.stripplot(x="university_top_20_raion", y="price_doc", data=train_df, jitter=True, alpha=.2, color=".8");

sns.boxplot(x="university_top_20_raion", y="price_doc", data=train_df)

ax.set(title='Distribution of home price by # of top universities in Raion', xlabel='university_top_20_raion', 

       ylabel='price_doc')
#Cultural/Recreational Characteristics

cult_chars = ['sport_objects_raion', 'culture_objects_top_25_raion', 'shopping_centers_raion', 'park_km', 'fitness_km', 

                'swim_pool_km', 'ice_rink_km','stadium_km', 'basketball_km', 'shopping_centers_km', 'big_church_km',

                'church_synagogue_km', 'mosque_km', 'theater_km', 'museum_km', 'exhibition_km', 'catering_km', 'price_doc']

corrmat = train_df[cult_chars].corr()
f, ax = plt.subplots(figsize=(12, 7))

plt.xticks(rotation='90')

sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
f, ax = plt.subplots(figsize=(10, 6))

so_price = train_df.groupby('sub_area')[['sport_objects_raion', 'price_doc']].median()

sns.regplot(x="sport_objects_raion", y="price_doc", data=so_price, scatter=True, truncate=True)

ax.set(title='Median Raion home price by # of sports objects in Raion')
f, ax = plt.subplots(figsize=(10, 6))

co_price = train_df.groupby('sub_area')[['culture_objects_top_25_raion', 'price_doc']].median()

sns.regplot(x="culture_objects_top_25_raion", y="price_doc", data=co_price, scatter=True, truncate=True)

ax.set(title='Median Raion home price by # of culture objects in Raion')

train_df.groupby('culture_objects_top_25')['price_doc'].median()
f, ax = plt.subplots(figsize=(10, 6))

sns.regplot(x="park_km", y="price_doc", data=train_df, scatter=True, truncate=True, scatter_kws={'color': 'r', 'alpha': .2})

ax.set(title='Median Raion home price by park_km objects in Raion')
#Infrastructure Features

inf_features = ['nuclear_reactor_km', 'thermal_power_plant_km', 'power_transmission_line_km', 'incineration_km',

                'water_treatment_km', 'incineration_km', 'railroad_station_walk_km', 'railroad_station_walk_min', 

                'railroad_station_avto_km', 'railroad_station_avto_min', 'public_transport_station_km', 

                'public_transport_station_min_walk', 'water_km', 'mkad_km', 'ttk_km', 'sadovoe_km','bulvar_ring_km',

                'kremlin_km', 'price_doc']

corrmat = train_df[inf_features].corr()
f, ax = plt.subplots(figsize=(12, 7))

plt.xticks(rotation='90')

sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
f, ax = plt.subplots(figsize=(10, 6))

sns.regplot(x="kremlin_km", y="price_doc", data=train_df, scatter=True, truncate=True, scatter_kws={'color': 'r', 'alpha': .2})

ax.set(title='Home price by distance to Kremlin')
#Variable Importance

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

X_train = train_df.drop(labels=['timestamp', 'id', 'incineration_raion'], axis=1).dropna()

y_train = X_train['price_doc']

X_train.drop('price_doc', axis=1, inplace=True)

for f in X_train.columns:

    if X_train[f].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(X_train[f])

        X_train[f] = lbl.transform(X_train[f])

rf = RandomForestRegressor(random_state=0)

rf = rf.fit(X_train, y_train)
fi = list(zip(X_train.columns, rf.feature_importances_))

print('## rf variable importance')

d = [print('## %-40s%s' % (i)) for i in fi[:20]]
#Train vs Test Data

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

test_na = (test_df.isnull().sum() / len(test_df)) * 100

test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12, 8))

plt.xticks(rotation='90')

sns.barplot(x=test_na.index, y=test_na)

ax.set(title='Percent missing data by feature', ylabel='% missing')
all_data = pd.concat([train_df.drop('price_doc', axis=1), test_df])

all_data['dataset'] = ''

l = len(train_df)

all_data.iloc[:l]['dataset'] = 'train'

all_data.iloc[l:]['dataset'] = 'test'

train_dataset = all_data['dataset'] == 'train'
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)

all_data['full_sq_log'] = np.log1p(all_data['full_sq'])

all_data.drop(train_dataset)["full_sq_log"].plot.kde(ax=ax[0])

all_data.drop(~train_dataset)["full_sq_log"].plot.kde(ax=ax[1])

ax[0].set(title='test', xlabel='full_sq_log')

ax[1].set(title='train', xlabel='full_sq_log')
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)

all_data['life_sq_log'] = np.log1p(all_data['life_sq'])

all_data.drop(train_dataset)["life_sq_log"].plot.kde(ax=ax[0])

all_data.drop(~train_dataset)["life_sq_log"].plot.kde(ax=ax[1])

ax[0].set(title='test', xlabel='life_sq_log')

ax[1].set(title='train', xlabel='life_sq_log')
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)

all_data['kitch_sq_log'] = np.log1p(all_data['kitch_sq'])

all_data.drop(train_dataset)["kitch_sq_log"].plot.kde(ax=ax[0])

all_data.drop(~train_dataset)["kitch_sq_log"].plot.kde(ax=ax[1])

ax[0].set(title='test', xlabel='kitch_sq_log')

ax[1].set(title='train', xlabel='kitch_sq_log')
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)

ax[0].scatter(x=test_df['floor'], y=test_df['max_floor'], c='r', alpha=0.4)

ax[0].plot([0, 80], [0, 80], color='.5')

ax[1].scatter(x=train_df['floor'], y=train_df['max_floor'], c='r', alpha=0.4)

ax[1].plot([0, 80], [0, 80], color='.5')

ax[0].set(title='test', xlabel='floor', ylabel='max_floor')

ax[1].set(title='train', xlabel='floor', ylabel='max_floor')
years = mdates.YearLocator()   # every year

yearsFmt = mdates.DateFormatter('%Y')

ts_vc_train = train_df['timestamp'].value_counts()

ts_vc_test = test_df['timestamp'].value_counts()

f, ax = plt.subplots(figsize=(12, 6))

plt.bar(left=ts_vc_train.index, height=ts_vc_train)

plt.bar(left=ts_vc_test.index, height=ts_vc_test)

ax.xaxis.set_major_locator(years)

ax.xaxis.set_major_formatter(yearsFmt)

ax.set(title='Number of transactions by day', ylabel='count')
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey=True)

sns.countplot(x=test_df['state'], ax=ax[0])

sns.countplot(x=train_df['state'], ax=ax[1])

ax[0].set(title='test', xlabel='state')

ax[1].set(title='train', xlabel='state')
train_df.head()