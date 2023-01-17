# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

for f in os.listdir('../input'):

    print(f.ljust(30) + str(round(os.path.getsize('../input/' + f), 2)) + 'MB')

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/presidents.csv')

print('Columns:')

df.columns.tolist()

print(df['Years in Office'])

df['Years in Office'].fillna(0, inplace=True)

print(df['Years in Office'])

df['Years in Office'].fillna(0, inplace=False)

print(df['Years in Office'])
