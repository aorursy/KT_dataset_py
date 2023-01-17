# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 60)

plt.style.use('fivethirtyeight')



import seaborn as sns

cm = sns.light_palette("green", as_cmap=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        df = pd.read_csv(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df.head()
# pok_summary = round(df[['HP','Attack','Defense','Sp_Atk','Sp_Def','Speed','type1']]\

# .groupby(['type1'], as_index=False).agg(['mean','count']).reset_index(),0)

pok_summary = round(df[['HP','Attack','Defense','Sp_Atk','Sp_Def','Speed','type1','sum']]\

.groupby(['type1'], as_index=False).agg(['mean']).reset_index(),0)

pok_summary.columns = ['_'.join([str(c) for c in c_list]) for c_list in pok_summary.columns.values]
pok_summary.style.background_gradient(cmap=cm).set_caption('Average stats per Class')