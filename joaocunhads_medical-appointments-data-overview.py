# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_appointments = pd.read_csv('../input/KaggleV2-May-2016.csv', error_bad_lines=False)
pd_columns = ['length']
pd_index   = ['appointments']
pd_data    = [len(df_appointments)]

pd.DataFrame(pd_data, index = pd_index, columns = pd_columns)
df_appointments.head()
df_appointments.describe()
df_appointments.hist(column='Scholarship')
df_appointments.hist(column='Hipertension')
df_appointments.hist(column='Diabetes')
df_appointments.hist(column='Alcoholism')
df_appointments.hist(column='Handcap')
df_appointments.hist(column='SMS_received')
df_appointments['No-show'] = df_appointments['No-show'].map({'Yes' : 1, 'No': 0})
df_appointments.hist(column='No-show')