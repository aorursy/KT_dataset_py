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
data_set = pd.read_csv('../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv')
data_set.head()
data_set.columns
#deaths by location

for i, deaths in enumerate(data_set['Province/State']):

    print(f'{deaths} : {data_set["Deaths"][i]}')
state_list = data_set['Province/State']
confirmed_unique_state = data_set['Province/State'][data_set.Confirmed > 0].unique()
confirmed_unique_state
deaths_unique_state = data_set['Province/State'][data_set.Deaths > 0].unique()

deaths_unique_state
suspected_unique_state = data_set['Province/State'][data_set.Suspected > 0].unique()

suspected_unique_state
recovered_unique_state = data_set['Province/State'][data_set.Recovered > 0].unique()

recovered_unique_state
# data_set[data_set['Province/State'] in confirmed_unique_state].Confirmed.sum()