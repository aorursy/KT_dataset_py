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
df_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df_temp=df_confirmed.loc[(df_confirmed['Country/Region']=='US')]

df_temp
df_temp.columns
pd.options.display.max_rows = 999

use_cols=['Province/State', 'Country/Region','3/9/20', '3/10/20']

df_temp=df_temp[use_cols]

df_temp
df_temp['3/10/20'].sum()