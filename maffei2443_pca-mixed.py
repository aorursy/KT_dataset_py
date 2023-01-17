import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

from collections import Counter



from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.manifold import TSNE

import os

from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

import json

from datetime import datetime

# In[2]:



INITIAL_DATE = str(datetime.now())

DUMP_DIR = '_DUMP_' + INITIAL_DATE



def load_year(y):

    dtypes = pickle.load(open(

        f'../input/kdd2020-cpr/{y}_dtypes.pkl', 'rb'

    ))

    del dtypes['date']

    df = pd.read_csv(

        f'../input/kdd2020-cpr/{y}.csv',

        dtype=dtypes, parse_dates=['date']

    )

    return df



def create_features(df):

    df['input_month'] = df.date.dt.month

    df['input_day'] = df.date.dt.day

    df['input_dt_sin_quarter']     = np.sin(2*np.pi*df.date.dt.quarter/4)

    df['input_dt_sin_day_of_week'] = np.sin(2*np.pi*df.date.dt.dayofweek/6)

    df['input_dt_sin_day_of_year'] = np.sin(2*np.pi*df.date.dt.dayofyear/365)

    df['input_dt_sin_day']         = np.sin(2*np.pi*df.date.dt.day/30)

    df['input_dt_sin_month']       = np.sin(2*np.pi*df.date.dt.month/12)

    return df





def date_expand(df, pipeline = True):

    def is_weekend(num):

        return num > 5

    dt = df['date'].dt

    df['input_week'] = dt.week

    df['input_weekday'] = dt.weekday + 1

    df['input_weekday_sin'] = np.sin(2*np.pi*df.date.dt.weekday/7)    

    df['input_weekofyear'] = dt.weekofyear

    df['input_weekofyear_sin'] = np.sin(2*np.pi*dt.weekofyear/52)

    df['input_weekend'] = dt.weekday.apply(is_weekend)

    return df if pipeline else None



GLOBAL="""

YEAR = 2018

COMPONENTS = 100

XGB_ESTIMATORS = 100

MAGICIAN = 'PCA'

INPUT_NA = 0

"""

# 

exec(GLOBAL)

df = load_year(YEAR)

y_cols = df.columns[df.columns.str.contains('output')]

y_train = df[y_cols]

y_cols_flat = y_cols.ravel()
COMPONENTS = 1000

params = {

	'n_estimators':XGB_ESTIMATORS,

	'random_state':0,

	'n_jobs':-1,

	'learning_rate':.2,

	'max_depth':13,

	'tree_method':'gpu_hist',

}



cp = df.copy()

cp['day'] = cp.date.dt.day

cp['month'] = cp.date.dt.month

cp.drop(['date', 'id'] + y_cols.tolist(), axis=1, inplace=True)

cp.fillna(INPUT_NA, inplace=True)





pca = PCA(n_components=COMPONENTS)

pca.fit(cp)



clf = MultiOutputRegressor(XGBRegressor(**params))





def preprocess(df):

	cp = df.copy()

	ids = df.id.copy()

	cp.drop(['id'] + y_cols.tolist(), axis=1, inplace=True, errors='ignore')

	cp.fillna(0, inplace=True)

	cp['day'] = cp.date.dt.day

	cp['month'] = cp.date.dt.month

	componentes_principais = pd.DataFrame(pca.transform(cp.drop('date', axis=1)))

	print('componentes_principais.shape:', componentes_principais.shape)

	mixed_df = pd.concat([cp.date, componentes_principais], axis=1, ignore_index=True)

	print("mixed_df.shape:", mixed_df.shape)

	mixed_df.columns = ['date', *[f'c_{i}' for i in list(range(0, 100))]]

	create_features(mixed_df);

	date_expand(mixed_df);

	mixed_df.drop('date', inplace=True, axis=1)

	return mixed_df, ids



df_train, ids_train = preprocess(df)

clf.fit( df_train, y_train.fillna(INPUT_NA))
np.cumsum(pca.explained_variance_ratio_)
%%time

d9 = load_year(2019)

X, ids = preprocess(d9)

ID_COL = []

for i in ids:

    ID_COL.extend(

    	[f'{i}_{sufix}' for sufix in y_cols_flat]

    )



y_pred = clf.predict(X)

df_sub = pd.DataFrame(

    {'id':ID_COL , 'value':y_pred.ravel()}

)

df_sub.set_index('id', inplace=True)
def dump():

	os.makedirs(DUMP_DIR, exist_ok=True)

	global CSV_PATH

	CSV_PATH = '{}/submission.csv'.format(DUMP_DIR)

	df_sub.to_csv(CSV_PATH, index='id')

	params_dump_name = '{}/params.json'.format(DUMP_DIR)

	json.dump(params, open(params_dump_name, 'w'))

	with open(f'{DUMP_DIR}/global.txt', 'w') as fp:

		fp.write(GLOBAL+'\n')

# Após treinar e criar o csv para submeter, salva os parâmetros.

dump()





# message = f"""

# - {MAGICIAN}

# - params: {str(params)}

# """



# os.system(

# 	'kaggle competitions submit -c kddbr-2020 -f {} -m "{}"'.format(CSV_PATH, message)

# )