# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load only the necessary fields
fields = [
    'POS_DAT', # Position date
    'OPT_TIC', # Option Ticker
    'COV_POS', # Covered Position
    'LOC_POS', # Locked Position
    'UNC_POS', # Uncovered Position
    'TOT_POS', # Total Position
    'HOL_QTY', # Holder Quantity
    'SEL_QTY', # Seller Quantity'
]
df = pd.read_csv(
    '/kaggle/input/opes-de-empresas/CONCATENATED.csv', 
    usecols=fields
)
# Create a new filtered dataframe
# The option VIVTQ50 was recomended to sell short by a research house in April, 15
# The option COGNQ430 was recomended to buy by a research house in April, 7
df_filter = df[df.OPT_TIC == 'COGNQ430']
df_filter = df_filter[(df_filter['POS_DAT'] >= '2020-04-01') & (df_filter['POS_DAT'] < '2020-04-30')]
df_sorted = df_filter.sort_values(
    by=['POS_DAT'], 
    ascending=True
)

display(df_sorted)