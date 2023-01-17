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
FILEPATH = '/kaggle/input/indian-migration-history/IndianMigrationHistory1.3.csv'
df = pd.read_csv(FILEPATH)
df.shape
df.describe()
df.info()
df.isnull().any().any()
df.sample(3)
df = df.drop(['Country Origin Name', 'Country Origin Code', 'Migration by Gender Name'], axis = 1)
df = df.rename({

    '1960 [1960]' : '1960',

    '1970 [1970]' : '1970',

    '1980 [1980]' : '1980',

    '1990 [1990]' : '1990',

    '2000 [2000]' : '2000',

    'Migration by Gender Code' : 'GenderCode',

    'Country Dest Name' : 'DestName',

    'Country Dest Code' : 'DestCode'

}, axis = 1)
df[df['DestCode'] == 'USA']
len(list(df['DestName'].unique()))
found = df[df['DestName'].str.contains('States')]

print(found.count())
# United States



df_usa = df[df['DestName'] == 'United States']



df_usa = df_usa.drop(['DestName', 'DestCode'], axis = 1)



df_usa