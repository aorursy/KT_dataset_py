# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

import os

import gc

import warnings

warnings.filterwarnings("ignore")
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16 or not. feather format does not support float16.

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
#We need to install ngboost first ;-)

!pip install ngboost
from ngboost.ngboost import NGBoost

from ngboost.learners import default_tree_learner

from ngboost.scores import MLE

from sklearn.metrics import mean_squared_error

from ngboost.distns import Normal
path = '/kaggle/input/ashrae-feather-format-for-fast-loading/'

files = os.listdir(path)

print(files)
files = ['building_metadata.feather','test.feather','weather_test.feather','weather_train.feather','train.feather','sample_submission.feather']

bmeta = pd.read_feather(path+files[0])

test = pd.read_feather(path+files[1])

wtest = pd.read_feather(path+files[2])

wtrain = pd.read_feather(path+files[3])

train = pd.read_feather(path+files[4])
test['is_train'] = 0

train['is_train'] = 1

wtotal = pd.concat([wtrain,wtest], ignore_index=True)

total = pd.concat([train,test],ignore_index=True)

p_u = bmeta['primary_use'].unique().astype(str)

p_u_dict={i :idx for idx,i in enumerate(p_u)}

bmeta.primary_use = bmeta.primary_use.map(p_u_dict)

bmeta.primary_use = bmeta.primary_use.astype(int)

total = total.merge(bmeta[['site_id','building_id','primary_use','square_feet']], on='building_id',how='left')

timestamp = total.groupby(['site_id','timestamp'],as_index=False).mean()[['site_id','timestamp']]

wtotal = timestamp.merge(wtotal,on=['site_id','timestamp'],how='left')

#Interpolation (nearest) -> (backward fill)

for i in tqdm(wtotal.site_id.unique()):

    wtotal.update(wtotal.loc[wtotal.site_id==i].interpolate('nearest',limit_direction='both'))

    wtotal.update(wtotal.loc[wtotal.site_id==i].fillna(method='bfill'))

total = total.merge(wtotal, on=['site_id','timestamp'],how='left')

total['M'] = total.timestamp.dt.month

total['D'] = total.timestamp.dt.dayofweek

total['H'] = total.timestamp.dt.hour

total['Q'] = total.timestamp.dt.quarter

total['W'] = total.timestamp.dt.week

total = reduce_mem_usage(total)
train = total.loc[total.is_train==1]

test = total.loc[total.is_train==0]

train['log1p_meter_reading'] = np.log1p(train.meter_reading)

train = train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
train.columns
# Select features to use for training.

tg = ['log1p_meter_reading']

do_not_use =  tg + ['meter','is_train'

              ,'timestamp'

              ,'meter_reading'

              ,'cloud_coverage'

              ,'precip_depth_1h'

              ,'sea_level_pressure'

                ,'precip_depth_1_hr'

              ,'row_id'

              ,'wind_direction']

cols = [c for c in train.columns if c not in do_not_use]
print('NULL CHECKING')

print('#####Train#####')

print(train[cols].isnull().sum())

print('#####Test#####')

print(test[cols].isnull().sum())
del total

del wtrain

del wtest

del bmeta
def Ngboost_training(df,tdf,meter):

    folds = 2

    seed = 7

    shuffle = False

    kf = StratifiedKFold(n_splits = folds, shuffle=shuffle , random_state=seed)

    #Down-sampling

    df = df.loc[(df.meter==meter)&(df.H==0)]

    tdf = tdf.loc[tdf.meter==meter]

    prediction = np.zeros(tdf.shape[0])

    i = 0

    ngb = NGBoost(n_estimators=50, learning_rate=0.4,

                  Dist=Normal,

                  Base=default_tree_learner,

                  natural_gradient=True,

                  minibatch_frac=0.6,

                  Score=MLE(),verbose=False)

    for tr,val in tqdm(kf.split(df, df['building_id']),total=folds):

        print(f'fold:{i+1}')

        i+=1

        print(f'Target : {tg[0]}// Meter : {meter}// # of features : {len(cols)}')

        print(f'Train_size : {len(tr)} Validation_size : {len(val)}')

        

        ngb.fit(df[cols].iloc[tr].values, df[tg[0]].iloc[tr].values)

        

        Y_preds = ngb.predict(df[cols].values)

        Y_dists = ngb.pred_dist(df[cols].values)

        

        MSE = mean_squared_error(Y_preds, df[tg[0]].values)

        print('MSE : ', MSE)

        NLL = -Y_dists.logpdf(df[tg[0]].values.flatten()).mean()

        print('NLL(Negative Log Likelihood)', NLL)

        

        #Test Prediction

        test_preds = ngb.predict(tdf[cols].values)

        print(f'Predicted Size : {len(test_preds)}')

        prediction += test_preds

        gc.collect()

    prediction = prediction/folds

    print('End')

    return prediction,ngb
sub = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/sample_submission.feather')
sub.head()
pred0,ngb0 = Ngboost_training(train,test,0)

gc.collect()

test.loc[test['meter'] == 0, 'meter_reading'] = np.clip(np.expm1(pred0), a_min=0, a_max=None)

pred1,ngb1 = Ngboost_training(train,test,1)

gc.collect()

test.loc[test['meter'] == 1, 'meter_reading'] = np.clip(np.expm1(pred1), a_min=0, a_max=None)

pred2,ngb2 = Ngboost_training(train,test,2)

gc.collect()

test.loc[test['meter'] == 2,'meter_reading'] = np.clip(np.expm1(pred2), a_min=0, a_max=None)

pred3,ngb3 = Ngboost_training(train,test,3)

gc.collect()

test.loc[test['meter'] == 3, 'meter_reading'] = np.clip(np.expm1(pred3), a_min=0, a_max=None)
sub['meter_reading'] = test['meter_reading'].values

sub.to_csv('submission.csv', index=False, float_format='%.4f')
sub.head(10)
sub.describe().astype(int)
print('Meter 0')

test.loc[test.meter==0][['timestamp','meter_reading']].set_index('timestamp').resample('H').meter_reading.mean().plot()
print('Meter 1')

test.loc[test.meter==1][['timestamp','meter_reading']].set_index('timestamp').resample('H').meter_reading.mean().plot()
print('Meter 2')

test.loc[test.meter==2][['timestamp','meter_reading']].set_index('timestamp').resample('H').meter_reading.mean().plot()
print('Meter 3')

test.loc[test.meter==3][['timestamp','meter_reading']].set_index('timestamp').resample('H').meter_reading.mean().plot()