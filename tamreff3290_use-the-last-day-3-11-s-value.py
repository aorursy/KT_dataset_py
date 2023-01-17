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
ss = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
last_df = train_df[train_df.Date == '2020-03-11']
valid_df = train_df[train_df.Date > '2020-03-11']
valid_df = pd.merge(valid_df, last_df, on=['Province/State', 'Country/Region'])
y_true = pd.concat([valid_df['ConfirmedCases_x'], valid_df['Fatalities_x']])

y_pred = pd.concat([valid_df['ConfirmedCases_y'], valid_df['Fatalities_y']])
from sklearn.metrics import mean_squared_log_error

rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

print (rmsle) # valid score
ss.head()
sub = pd.merge(test_df, last_df, on=['Province/State', 'Country/Region'])[ss.columns]
sub.to_csv('submission.csv', index=False)