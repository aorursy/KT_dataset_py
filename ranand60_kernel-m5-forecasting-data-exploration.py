#This Python 3 environment comes with many helpful analytics libraries installed
#It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
#For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import cycle
import matplotlib.pylab as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error

#Input data files are available in the "../input/" directory.
#For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Any results you write to the current directory are saved as output.
Calender_path = '/kaggle/input/m5-forecasting-accuracy/calendar.csv'
calender_data = pd.read_csv(Calender_path)
calender_data.columns
plt.figure(figsize=(30,25))
sns.pairplot(calender_data)
input_path = '/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv'
input_data = pd.read_csv(input_path)
input_data.info()
input_data.columns
input_data.head(10)
input_data.describe()
input_data.shape
input_data=input_data.dropna(axis=0)
input_data.shape
plt.figure(figsize=(5,5))
sns.countplot(data=input_data, x=input_data['cat_id'])
plt.figure(figsize=(10,8))
sns.countplot(data=input_data, x=input_data['store_id'])
plt.figure(figsize=(20,10))
sns.countplot(data=input_data, x=input_data['item_id'])
my_color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
data_date_cols = [column for column in input_data.columns if 'd_' in column]

input_data.loc[input_data['id'] == 'FOODS_3_090_CA_3_validation'] \
    .set_index('id')[data_date_cols] \
    .T \
    .plot(figsize=(15, 8),
          title='FOODS_3_090_CA_3 sales in Numbers : ',
          color=next(my_color_cycle))
plt.legend('')
plt.show()
for i, var in enumerate(["year", "weekday", "month", "event_name_1", "event_name_2", 
                         "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]):
    plt.figure(figsize=(10,10))
    m5_plot = sns.countplot(calender_data[var])
    m5_plot.set_xticklabels(m5_plot.get_xticklabels(), rotation=45)
    m5_plot.set_title(var)