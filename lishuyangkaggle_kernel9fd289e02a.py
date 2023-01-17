# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



BASE_DIR = '../input'

csv_path = os.path.join(BASE_DIR, os.listdir(BASE_DIR)[0])

print('Cocktails CSV located at {}'.format(csv_path))



# Any results you write to the current directory are saved as output.
'''

Load data into pandas DataFrame

'''

start = datetime.now()

df = pd.read_csv(csv_path, engine='c')

print('{} - Loaded data for {:,} cocktails'.format(

    datetime.now() - start, len(df)

))
'''

With ingredients

'''

df = df[~df['Preparation'].isna()]

print('{:,} cocktails have steps'.format(len(df)))