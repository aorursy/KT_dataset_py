# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
times_df = pd.read_csv("../input/timesData.csv")
times_df.columns
col = ['world_rank', 'teaching', 'international',

       'research', 'citations', 'income', 'total_score', 'num_students',

       'student_staff_ratio', 'international_students',

       'year']

times_df = times_df[col]

len(times_df)
times_df.head()
times_df = times_df[~times_df['income'].isin(['-'])]



times_df.head(7)
len(times_df)
times_df = times_df.dropna()

len(times_df)
import matplotlib.pyplot as plt

import seaborn as sns



fig, ax = plt.subplots(1,1, figsize=(10,5))

corr = times_df.corr()

sns.heatmap(corr, ax=ax)

ax.set_title("correlation")



plt.show()