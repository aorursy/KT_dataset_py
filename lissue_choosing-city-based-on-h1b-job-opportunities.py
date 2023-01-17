# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datashader as ds

from datashader import transfer_functions as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/h1b_kaggle.csv')

df.head()
df.describe()
df = df[(df['PREVAILING_WAGE']>30000) & (df['CASE_STATUS']=='CERTIFIED')]

df.describe()
US = x_range, y_range = ((-124.7844079, -66.9513812), (24.7433195, 49.3457868))

plot_width  = int(900)

plot_height = int(plot_width//2)



cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)

agg = cvs.points(df, 'lon', 'lat', ds.mean('PREVAILING_WAGE'))

img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='eq_hist')

img
pop_profession = df['SOC_NAME'].describe()

pop_profession
df['SOC_NAME'].value_counts().head(20).plot(kind='bar')
df_sort_wage = df.sort_values(by=['PREVAILING_WAGE'])

df_sort_wage.head(10)
df_sort_wage.tail(10)
df_sort_wage.tail(50).describe()
df_sort_wage['SOC_NAME'].tail(50).value_counts().plot(kind='bar')