# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn

import pandas_profiling

from sklearn import preprocessing 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/donorschoose-application-screening/train.csv')

print('Train Shape',train_df.shape)

resource_df = pd.read_csv('/kaggle/input/donorschoose-application-screening/resources.csv')

print('Train Shape',resource_df.shape)
resource_Price = resource_df.groupby(by='id').sum().reset_index()

resource_Price.columns = ['id', 'Res_qty', 'Res_price']



train_df = pd.merge(train_df, resource_Price, on='id', how='left')
train_df.project_is_approved.value_counts().plot(kind='bar',stacked=True)

plt.show()
train_df.profile_report(style={'full_width':True})