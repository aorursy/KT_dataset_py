# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot



pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame

pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame



df = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
df
colm=['transaksi','date','produksku','produk','pkategori','kuantitas','aprice','pendapatan','pajak','ongkir']

df.columns = colm
df
#tanggal di atas kan berupa angka, nah biar jadi string, ngubahnya kayak gini

df['date'] = df['date'].astype(str)

df['date'] = pd.to_datetime(df['date']) 

df
#kl mau tau datanya dari tanggal berapa sampai tanggal berapa, caranya kayak gini

dates=pd.date_range('2017-01-01', periods=10)

df[df['date'].isin(dates)]
dfsell= df[(df['date'].isin(dates))]

q=sb.lineplot(data=dfsell, x='date', y='pendapatan', hue='pkategori');
w=df.groupby('pkategori')['produk'].count().reset_index(name='kuantitas')
w=df.groupby('pkategori')['produk'].count().reset_index(name='kuantitas')

w.plot(kind='bar', x='pkategori');