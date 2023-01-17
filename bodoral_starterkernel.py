# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_sa = pd.read_csv('/kaggle/input/trending-youtube-video/saudi_youtube_trending_videos.csv')
df_sa.head()


bins = [0,1,5,10,15,30]

df = df_sa.groupby(pd.cut(df_sa['duration'], bins=bins)).duration.count()

df.plot(kind='bar')

plt.figure(figsize = (25,8))

sns.countplot('category', data = df_sa, order = df_sa['category'].value_counts().index)