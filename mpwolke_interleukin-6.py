#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRq9z1onTZ_lNG_3qMm_utUW4zLfcpT4NgoWNbIJj2HM-2Koq3q&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/cusersmarildownloadsinterleukincsv/interleukin.csv', sep=';')

df
df.describe()
df['204252_at'].hist()

plt.show()
ax = df.groupby('Samples')['204252_at'].max().sort_values(ascending=True).plot(kind='barh', figsize=(12,8),

                                                                                   title='Maximum Interleukin-6 Samples')

plt.xlabel('204252_at')

plt.ylabel('Samples')

plt.show()
ax = df.groupby('Samples')['204252_at'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(12,8),

                                                                                   title='Mean Interleukin-6 Samples')

plt.xlabel('204252_at')

plt.ylabel('Samples')

plt.show()
ax = df.groupby('Samples')['204252_at'].min().sort_values(ascending=True).plot(kind='barh', figsize=(12,8), 

                                                                                  title='Minimum Interleukin-6 Samples')

plt.xlabel('204252_at')

plt.ylabel('Samples')

plt.show()
ax = df.groupby('Samples')['211804_s_at', '211803_at'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='211804_s_at x 211803_at')

plt.xlabel('Samples')

plt.ylabel('Log Scale 211804_s_at')

plt.show()
ax = df.groupby('Samples')['211804_s_at', '211803_at'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='211804_s_at x 211803_at', logx=True, linewidth=3)

plt.xlabel('Log Scale 211804_s_at')

plt.ylabel('Samples')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQNaOi7WLRM51M_Jg82Y5MIUpCPjmmt1Kavql8KQU7v4PonuNq9&usqp=CAU',width=400,height=400)