# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




cols_list = []

for j in range(8):

    for i in range(8):

        cols_list.append(f'S{i}R{j}')

cols_list.append('target')



df = pd.read_csv('/kaggle/input/emg-4/0.csv', header=None)

df.columns = cols_list

df
pd.wide_to_long(df.reset_index(), ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S0'], i=['index', 'target'], j='R', sep='R')