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
from tqdm import tqdm



import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

from itertools import cycle

pd.set_option('max_columns', 50)

plt.style.use('bmh')

color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
INPUT_DIR = '../input/m5-forecasting-accuracy'

cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')

stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')

ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')

sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')
d_cols = [c for c in stv.columns if 'd_' in c] # sales data column



q = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6','F7', 'F8', 'F9', 'F10','F11', 'F12','F13',

     'F14', 'F15', 'F16', 'F17', 'F18','F19', 'F20','F21', 'F22', 'F23', 'F24',

     'F25', 'F26', 'F27', 'F28']

m = ['id','F1', 'F2', 'F3', 'F4', 'F5', 'F6','F7', 'F8', 'F9', 'F10','F11', 'F12','F13',

     'F14', 'F15', 'F16', 'F17', 'F18','F19', 'F20','F21', 'F22', 'F23', 'F24',

     'F25', 'F26', 'F27', 'F28']



k = pd.DataFrame(ss.loc[0]).T



for i in tqdm(list(stv['id'].unique())):

#for i in tqdm(['HOBBIES_1_001_CA_1_validation', 'HOBBIES_1_002_CA_1_validation']):

    df = (stv.loc[stv['id'] == i][d_cols]

          .T

          .rename(columns={stv[stv['id']==i].index[0]:i})

          .reset_index()

          .rename(columns={'index': 'd'})

          .merge(cal[['date','d']],how='left', validate='1:1')

          .rename(columns = {i: 'y','date':'ds'})

          .drop('d',1))

    

    df['ds'] = pd.to_datetime(df['ds'])

    from fbprophet import Prophet

    m = Prophet(uncertainty_samples=False)

    m.fit(df)

    future = m.make_future_dataframe(periods=29).tail(28)

    forecast = m.predict(future)

    f = forecast[[ 'yhat']].rename(columns = {'yhat':i})

    f.index = q

    f = f.T.reset_index().rename(columns = {'index':'id'})

    #f.columns = m

    k = k.append(f)
k = (k

     .reset_index()

     .drop(0)

     .drop('index',1)

     .reset_index()

     .drop('index',1))



(k

 .append(ss.iloc[30490:])

 .to_csv('Final_1235.csv', index = False))