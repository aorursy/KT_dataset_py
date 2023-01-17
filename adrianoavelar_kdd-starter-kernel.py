import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df





def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())



#from https://www.kaggle.com/marlesson/simple-pytorch-model don't know if is right

weigths = [1]*16 + [0.75]*16 + [0.6]*16 + [0.5]*16 + [0.43]*16 + [0.38]*16 + [0.33]*16

def wrmse(predictions, targets, weigths):

    return np.sqrt((((predictions - targets) ** 2)*weigths).mean())

columns_size = None

rows_size = 1000
path='/kaggle/input/kddbr-2020/'



def get_files(years=[],months=12,pre=''):

    files = []

    

    for year in years:

        for month in range(1,months + 1):

            file = F'0{month}' if month < 10 else str(month)

            files.append( F'{path}/{pre}{year}{file}.csv' )

    return files



base = [pd.read_csv(file,parse_dates=['date']) for file in get_files( ['2018'], 12) ]

base = pd.concat(base, ignore_index=True)
if (columns_size != None or rows_size != None):

    inputs = list( base.columns[ base.columns.str.contains('input')][:columns_size])

    inputs.sort()

    output = list(base.columns[ base.columns.str.contains('output')] )

    cols = list( base[ inputs + output  ].columns )

    cols.append('id')

    cols.append('date')

    base = base[cols].copy()[:rows_size]



base.head()
print(base.shape)

base.head()
base = reduce_mem_usage(base)

#test = reduce_mem_usage(test)
#Check if date was parsed

base[['date']].info()
#check first 4 feature

for i in range(4):

    col = 'input_'+str(i)

    print(col,' : ', base[col].shape[0], base[col].nunique() )

input_columns = base.columns[base.columns.str.contains('input') ]

output_columns = base.columns[base.columns.str.contains('output') ]



print(F'Inputs: {input_columns.shape} Outputs: {output_columns.shape}')

input_columns, output_columns
len(base.columns[ base.columns.str.contains('input_4_') ])

for i in range(2000):

    

    if F'input_4_{i}' in input_columns:

        print(F'input_1_{i}')
plt.figure(figsize=(36, 22))

plt.subplots_adjust(top=1.2, hspace = 0.8)

sns.set_palette("husl")

palette = itertools.cycle(sns.color_palette())

for i in range(1,15):

    plt.subplot( 5, 3 ,i)

    col = F'input_4_{i}'

    plot = base.groupby( base['date'].dt.date )[col].mean().reset_index()

    sns.lineplot( plot.date, plot[col] , color=next(palette))

    plt.xticks(rotation=45,ha='right')

    plt.title(F" Mean of {col} distributed per month ")



plt.show()
plt.figure(figsize=(36, 22))

plt.subplots_adjust(top=1.2, hspace = 0.8)

sns.set_palette("husl")

palette = itertools.cycle(sns.color_palette())

for i in range(4,20):

    plt.subplot( 5, 4 ,i - 3)

    col = F'input_{i}_1'

    plot = base.groupby( base['date'].dt.date )[col].mean().reset_index()

    sns.lineplot( plot.date, plot[col] , color=next(palette))

    plt.xticks(rotation=45,ha='right')

    plt.title(F" Mean of {col} distributed per month ")



plt.show()
def create_features(df):

    df['input_month'] = df.date.dt.month

    df['input_year'] = df.date.dt.year

    df['input_day'] = df.date.dt.day

    df['input_dt_sin_quarter']     = np.sin(2*np.pi*df.date.dt.quarter/4)

    df['input_dt_sin_day_of_week'] = np.sin(2*np.pi*df.date.dt.dayofweek/6)

    df['input_dt_sin_day_of_year'] = np.sin(2*np.pi*df.date.dt.dayofyear/365)

    df['input_dt_sin_day']         = np.sin(2*np.pi*df.date.dt.day/30)

    df['input_dt_sin_month']       = np.sin(2*np.pi*df.date.dt.month/12)

    

    return df





create_features(base)



input_columns = base.columns[base.columns.str.contains('input') ]

output_columns = base.columns[base.columns.str.contains('output') ]

X = base[input_columns[:columns_size] ].fillna(0).values

Y = base[output_columns].fillna(0).values



del base

gc.collect()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=42)

x_train.shape, y_train.shape
%%time

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 1000, max_depth=10, random_state=0)



# fit model

model.fit(x_train,y_train)

#LB = 0.79503
y_pred = model.predict(x_val)
print( rmse(y_pred, y_val) )

print( wrmse(y_pred, y_val, weigths) )
input_columns[:-1]
#Loading test file

test = [pd.read_csv(file,parse_dates=['date']) for file in get_files(['2019'], 12,'public' )]

test = pd.concat(test, ignore_index=True).fillna(0)

test.head()
test = create_features(test)

input_columns = test.columns[test.columns.str.contains('input') ]
pred = model.predict(test[input_columns].values)

pred_sub = pd.DataFrame(pred)

pred_sub.columns = output_columns

pred_sub['id']   = test['id']

pred_sub.head()
submission = []

for i, row in pred_sub.iterrows():

    for column, value in zip(output_columns, row.values):

        id = "{}_{}".format(int(row.id), column)

        submission.append([id, value])



submission = pd.DataFrame(submission)

submission.columns = ['id', 'value']

submission
submission.to_csv('submission.csv', index=False)