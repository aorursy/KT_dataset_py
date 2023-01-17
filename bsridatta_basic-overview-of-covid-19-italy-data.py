import os

import pandas as pd

import pandas_profiling 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
national = pd.read_csv('/kaggle/input/national_data.csv')

national.dataframeName = 'national_data.csv'

nRow, nCol = national.shape

print(f'There are {nRow} rows and {nCol} columns')
national.profile_report(title='National Data', progress_bar=False)
provincial = pd.read_csv('/kaggle/input/provincial_data.csv')

provincial.dataframeName = 'provincial_data.csv'

nRow, nCol = provincial.shape

print(f'There are {nRow} rows and {nCol} columns')
provincial.profile_report(title='Provincial Data', progress_bar=False)
regional = pd.read_csv('/kaggle/input/regional_data.csv')

regional.dataframeName = 'regional_data.csv'

nRow, nCol = regional.shape

print(f'There are {nRow} rows and {nCol} columns')
regional.profile_report(title='Regional Data', progress_bar=False)