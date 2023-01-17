# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt  

import matplotlib

pd.set_option('display.max_rows', 1000)

from sklearn.preprocessing import StandardScaler

from zipfile import ZipFile 

import seaborn as sns


train = pd.read_csv(ZipFile("/kaggle/input/sberbank-russian-housing-market/train.csv.zip").open('train.csv'), parse_dates=['timestamp'])

test = pd.read_csv(ZipFile("/kaggle/input/sberbank-russian-housing-market/test.csv.zip").open('test.csv'), parse_dates=['timestamp'])

macro = pd.read_csv(ZipFile("/kaggle/input/sberbank-russian-housing-market/macro.csv.zip").open('macro.csv'), parse_dates=['timestamp'])
train.head()
## checking for normal distribution of saleprice

import seaborn as sns

x = 'price_doc'

fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)    

fig.suptitle(x, fontsize=20)



### distribution    

ax[0].title.set_text('   distribution')    

variable = train[x].fillna(train[x].mean())    

breaks = np.quantile(variable, q=np.linspace(0, 1, 11))



# variable = variable[ (variable > breaks[breaks[0]]) & (variable < breaks[breaks[1]]) ]     

sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])    

des = train[x].describe()    

ax[0].axvline(des["25%"], ls='--')    

ax[0].axvline(des["mean"], ls='--')    

ax[0].axvline(des["75%"], ls='--')    

ax[0].grid(True)    

des = round(des, 2).apply(lambda x: str(x))    

box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))    

ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))

### boxplot     

ax[1].title.set_text('outliers (log scale)')    

tmp_df = pd.DataFrame(train[x])    

tmp_df[x] = np.log(tmp_df[x])    

tmp_df.boxplot(column=x, ax=ax[1])    

plt.show()
df = pd.concat((train.loc[:,'timestamp':'market_count_5000'], test.loc[:,'timestamp':'market_count_5000']))



macro_cols = ["timestamp","balance_trade","balance_trade_growth","usdrub","average_provision_of_build_contract","micex_rgbi_tr","micex_cbi_tr","deposits_rate","mortgage_value","mortgage_rate","income_per_cap","museum_visitis_per_100_cap","apartment_build"]

df = df.merge(macro[macro_cols], on='timestamp', how='left')







df['Sale_year'] = df['timestamp'].dt.year

df['Sale_month'] = df['timestamp'].dt.month

df = df.drop(['timestamp'], axis=1)

df['max_floor'] = df['max_floor'].fillna(0)



# replace NaNs with "0"

    

all_columns = list(set(df.columns))

for col in all_columns:

    df[col].fillna('N', inplace = True)









df['floor'] = df['floor'].astype(str)

df['Sale_year'] = df['Sale_year'].astype(str)

df['material'] = df['material'].astype(str)

df['product_type'] = df['product_type'].astype(str)

df['culture_objects_top_25'] = df['culture_objects_top_25'].astype(str)

df['thermal_power_plant_raion'] = df['thermal_power_plant_raion'].astype(str)

df['incineration_raion'] = df['incineration_raion'].astype(str)

df['oil_chemistry_raion'] = df['oil_chemistry_raion'].astype(str)

df['radiation_raion'] = df['radiation_raion'].astype(str)

df['railroad_terminal_raion'] = df['railroad_terminal_raion'].astype(str)

df['big_market_raion'] = df['big_market_raion'].astype(str)

df['nuclear_reactor_raion'] = df['nuclear_reactor_raion'].astype(str)

df['detention_facility_raion'] = df['detention_facility_raion'].astype(str)

df['healthcare_centers_raion'] = df['healthcare_centers_raion'].astype(str)

df['water_1line'] = df['water_1line'].astype(str)

df['big_road1_1line'] = df['big_road1_1line'].astype(str)

df['railroad_1line'] = df['railroad_1line'].astype(str)

df['ID_railroad_terminal'] = df['ID_railroad_terminal'].astype(str)

df['ecology'] = df['ecology'].astype(str)



# df['full_sq_log'] = np.log1p(df['full_sq'])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((-1,1))

object_columns = df.select_dtypes(include=[np.object])

num_columns = df.select_dtypes(exclude=[np.object])



   

# Apply label encoder for category columns

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for col in object_columns:

    df[col] = le.fit_transform(df[col].astype(str))

    

# for col in all_columns:

#     df[col] = scaler.fit_transform(df[[col]])
def reduce_memory_usage(df):

    """ The function will reduce memory of dataframe

    Note: Apply this function after removing missing value"""

    intial_memory = df.memory_usage().sum()/1024**2

    print('Intial memory usage:',intial_memory,'MB')

    for col in df.columns:

        mn = df[col].min()

        mx = df[col].max()

        if df[col].dtype != object:            

            if df[col].dtype == int:

                if mn >=0:

                    if mx < np.iinfo(np.uint8).max:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < np.iinfo(np.uint16).max:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < np.iinfo(np.uint32).max:

                        df[col] = df[col].astype(np.uint32)

                    elif mx < np.iinfo(np.uint64).max:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)

            if df[col].dtype == float:

                df[col] =df[col].astype(np.float32)

    

    red_memory = df.memory_usage().sum()/1024**2

    print('Memory usage after complition: ',red_memory,'MB')

    

reduce_memory_usage(df)
def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
#creating matrices for feature selection:

X_train = df[:train.shape[0]]

X_test_fin = df[train.shape[0]:]

y = train.price_doc

X_train['Y'] = y

df = X_train

df['Y'] = df['Y']/df['usdrub']

df.head() ## DF for Model training
import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)



X = df.drop('Y', axis=1)

y = df.Y







params = {

        'objective':'reg:linear',

        'n_estimators': 5000,

        'booster':'gbtree',

        'max_depth':5,

        'eval_metric':'mae',

        'learning_rate':0.05, 

        'min_child_weight':0,

        'subsample':0.8,

        'colsample_bytree':0.79,

        'reg_alpha':0,

        'seed':45,

        'gamma':0,

        'nthread':-1

}





x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(X_test_fin)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 5000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=10)



p_test = clf.predict(d_test)
sub = pd.DataFrame()

sub['ID'] = test['id']

sub['price_doc'] =  52 * p_test

sub.to_csv('submission.csv', index=False)
sub
# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax)

plt.show()