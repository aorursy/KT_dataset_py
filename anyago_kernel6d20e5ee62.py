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
data = pd.read_csv('/kaggle/input/sputnik/train.csv',  parse_dates=['epoch'])
data.head()
grouped_by_ids = []

group = data.groupby(data.sat_id)

for i in set(data.sat_id.values):

    grouped_by_ids.append(group.get_group(i))
grouped_by_ids
from tqdm import tqdm

from IPython import get_ipython

def tqdm_clear(*args, **kwargs):

    from tqdm import tqdm

    getattr(tqdm, '_instances', {}).clear()



get_ipython().events.register('post_execute', tqdm_clear)
ids = []

errors = []

from sklearn.linear_model import LinearRegression

model = LinearRegression()

for i in tqdm(range(len(grouped_by_ids))):

    df = grouped_by_ids[i]

    df_x = df.drop(['y', 'z'], axis=1)

    df_y = df.drop(['x', 'z'], axis=1)

    df_z = df.drop(['x', 'y'], axis=1)

    

    df_x.rename(columns={'x': 'target'}, inplace=True)

    df_y.rename(columns={'y': 'target'}, inplace=True)

    df_z.rename(columns={'z': 'target'}, inplace=True)

    

    lag_period = 1

    for coord in [df_x, df_y, df_z]:

        features = []

        for period_mult in range(1, int(len(df)/10)):

            coord["lag_period_{}".format(period_mult)] = coord.target.shift(period_mult*lag_period + int(len(df)*0.25))

            features.append("lag_period_{}".format(period_mult))



        coord['lagf_mean'] = coord[features].mean(axis = 1)

        features.extend(['lagf_mean'])

 

    train_df = df_x[df_x.type == 'train'][features + ['target']].dropna()

    test_df = df_x[df_x.type == 'test'][features]

    model.fit(train_df.drop('target', axis = 1) ,train_df['target'])

    x_pred = model.predict(test_df)

    

    train_df = df_y[df_y.type == 'train'][features + ['target']].dropna()

    test_df = df_y[df_y.type == 'test'][features]

    model.fit(train_df.drop('target', axis = 1) ,train_df['target'])

    y_pred = model.predict(test_df)

    

    train_df = df_z[df_z.type == 'train'][features + ['target']].dropna()

    test_df = df_z[df_y.type == 'test'][features]

    model.fit(train_df.drop('target', axis = 1) ,train_df['target'])

    z_pred = model.predict(test_df)

    

    df_pred = pd.DataFrame(data=np.array([x_pred, y_pred, z_pred]).T,columns=["x", "y", "z"])

    error  = np.linalg.norm(df_pred[['x', 'y', 'z']].values - df[df.type == 'test'][['x_sim', 'y_sim', 'z_sim']].values, axis=1)

    errors.extend(list(error))

    ids.extend(list(df[df.type == 'test']['id'].values))
res = pd.DataFrame(np.array([list(map(int,ids)), errors]).T,columns=["id", "error"])

res['id'] = res['id'].astype(int)

res[['id', 'error']].to_csv('mysub.csv', index = False)