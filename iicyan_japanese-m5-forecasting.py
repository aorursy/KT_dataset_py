# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sp = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sp
sp.describe()
cl = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

cl
cl[cl['wm_yr_wk']==11101]
cl[cl['wm_yr_wk']==11152]
cl[cl['wm_yr_wk']==11153]
cl[cl['wm_yr_wk']==11201]
pd.set_option("display.max_rows", 200)

cl[['event_name_1','event_type_1','event_name_2','event_type_2']].dropna(how='all')
cl.describe()
stv = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

stv
stv.describe()
smpl = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

smpl
smpl.describe()