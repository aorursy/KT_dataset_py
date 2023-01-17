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
df_variables = pd.read_excel('/kaggle/input/humanbuilding-office-space-interactions/langevincodebook.xlsx', header=11, index_col='TEXT FILE COLUMN')
df_variables
df_variables['NAME'].to_list()
df_raw = pd.read_table('/kaggle/input/humanbuilding-office-space-interactions/LANGEVIN_DATA.txt', header=None)
df_raw = df_raw[0].str.split(' ',expand=True)
df_raw = df_raw.iloc[:, :-1]
df_raw.columns = df_variables['NAME'].to_list()
df_raw = df_raw.astype('float')
df_raw
df_raw.info()

