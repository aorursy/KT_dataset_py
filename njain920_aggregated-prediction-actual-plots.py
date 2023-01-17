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
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

pred = pd.read_csv('/kaggle/input/predictions/y1.25.csv')
#Get only the validation set

pred = pred[0:30490]
#Prepare data for grouping

pred.index = pred['id']

sales.index = sales['id']

pred['dept_id'] = sales['dept_id']

pred['store_id'] = sales['store_id']

pred['state_id'] = sales['state_id']

pred['cat_id'] = sales['cat_id']

del pred['id']

del sales['id']

pred = pred.reset_index()

sales = sales.reset_index()
#Group Prediction Data

store_dept_pred = pred.groupby(['store_id','dept_id']).agg('sum')

store_cat_pred = pred.groupby(['store_id','cat_id']).agg('sum')

state_dept_pred = pred.groupby(['state_id','dept_id']).agg('sum')

state_cat_pred = pred.groupby(['state_id','cat_id']).agg('sum')

#Group Original Data

store_dept_original = sales.groupby(['store_id','dept_id']).agg('sum').iloc[:,-28:]

store_cat_original = sales.groupby(['store_id','cat_id']).agg('sum').iloc[:,-28:]

state_dept_original = sales.groupby(['state_id','dept_id']).agg('sum').iloc[:,-28:]

state_cat_original = sales.groupby(['state_id','cat_id']).agg('sum').iloc[:,-28:]
#Rename F1 - F28 to DateRange

cols = pd.date_range(start = '2016-03-28',end = '2016-04-24')



store_dept_pred.columns = cols

store_cat_pred.columns = cols

state_dept_pred.columns = cols

state_cat_pred.columns = cols



store_dept_original.columns = cols

store_cat_original.columns = cols

state_dept_original.columns = cols

state_cat_original.columns = cols
#Store the column names to iterate over them to create plots

store_dept_columns = store_dept_pred.T.columns.to_list()

store_cat_columns = store_cat_pred.T.columns.to_list()

state_dept_columns = state_dept_pred.T.columns.to_list()

state_cat_columns = state_cat_pred.T.columns.to_list()
#State Category Plots

plt.figure(figsize=(35,14))       # set dimensions of the figure

for i in range(1,len(state_cat_columns)+1):

    plt.subplot(3,3,i)         # create subplots on a grid with 2 rows and 3 columns

    plt.plot(state_cat_pred.T[state_cat_columns[i-1]])

    plt.plot(state_cat_original.T[state_cat_columns[i-1]])

    plt.ylabel(state_cat_columns[i-1])

#     plt.legend(loc=(1.05, 0.5))

plt.legend(loc=(-1.2, 3.5),labels=["Predicted","Actual"])

plt.title('STATE - CATEGORY: Predicted vs Actual',x=-0.6,y=3.50)

plt.show()
#Store Category Plots

plt.figure(figsize=(35,24))       # set dimensions of the figure

for i in range(1,len(store_cat_columns)+1):

    plt.subplot(10,3,i)         # create subplots on a grid with 2 rows and 3 columns

    plt.plot(store_cat_pred.T[store_cat_columns[i-1]])

    plt.plot(store_cat_original.T[store_cat_columns[i-1]])

    plt.ylabel(store_cat_columns[i-1])

#     plt.legend(loc=(1.05, 0.5))

plt.legend(loc=(-1.2, 12.0),labels=["Predicted","Actual"])

plt.title('STORE - CATEGORY: Predicted vs Actual',x=-0.6,y=12.0)

plt.show()
#State Department Plots

plt.figure(figsize=(35,24))       # set dimensions of the figure

for i in range(1,len(state_dept_columns)+1):

    plt.subplot(7,3,i)         # create subplots on a grid with 2 rows and 3 columns

    plt.plot(state_dept_pred.T[state_dept_columns[i-1]])

    plt.plot(state_dept_original.T[state_dept_columns[i-1]])

    plt.ylabel(state_dept_columns[i-1])

#     plt.legend(loc=(1.05, 0.5))

plt.legend(loc=(-1.2, 8.5),labels=["Predicted","Actual"])

plt.title('STATE - DEPT: Predicted vs Actual',x=-0.6,y=8.5)

plt.show()
#Store Department Plots

plt.figure(figsize=(35,80))       # set dimensions of the figure

for i in range(1,len(store_dept_columns)+1):

    plt.subplot(24,3,i)         # create subplots on a grid with 2 rows and 3 columns

    plt.plot(store_dept_pred.T[store_dept_columns[i-1]])

    plt.plot(store_dept_original.T[store_dept_columns[i-1]])

    plt.ylabel(store_dept_columns[i-1])

#     plt.legend(loc=(1.05, 0.5))

plt.legend(loc=(1.2, 29.0),labels=["Predicted","Actual"])

plt.title('STORE - DEPT: Predicted vs Actual',x=1.8,y=29.0)

plt.show()