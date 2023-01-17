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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import re

from subprocess import check_output

df = pd.read_csv("../input/earthquake-database/database.csv")#

df.head()
df['date_parsed'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df['year'] = df["date_parsed"].apply(lambda x: x.year)
df.head()
def graph (magn_floor, magn_ceil):

    new_df = df

    new_df = new_df.loc[(df.Magnitude > magn_floor) & (df.Magnitude < magn_ceil)]



    figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

    value_counts = new_df['year'].value_counts().sort_index()

    value_counts.plot(kind='bar')

    a = []

    b = []

    plt.show()
graph (5, 6)
graph (6, 6.5)
graph (6.5, 7)
graph (7, 10)