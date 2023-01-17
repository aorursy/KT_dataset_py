# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/taiwan-taipei-city-real-estate-transaction-records/taipei_city_real_estate_transaction_v2.csv')
df.info()

df.head()
# The notion of years below follows Taiwan local calendar

df['complete_year'].head()
# change the calendar from Taiwan local calendar to Gregorian calendar

df['complete_year'] = df['complete_year'].astype(int) + 1911
type(df['complete_year'][0])
# convert data types

df['complete_year'] = df['complete_year'].astype(str)
type(df['complete_year'][0])
type(df['transaction_year'][0]), type(df['transaction_month'][0])
# convert data types

df['transaction_year'] = df['transaction_year'].astype(str)

df['transaction_month'] = df['transaction_month'].astype(str)
df['num_partition'].head()
# some columns are not translated so replace them directly

df['num_partition'] = df['num_partition'].apply(lambda x: 1 if x == '有' else 0)

df['management_org'] = df['management_org'].apply(lambda x: 1 if x == '有' else 0)
def change_word(x):

    if  x == 'Address':

        return 'Residence'

    

    elif x == 'Quotient':

        return 'Business'

    else:

        return x



df['urban_land_use'] = df['urban_land_use'].apply(lambda x: change_word(x))
categorical_col = ['district', 'transaction_type', 'urban_land_use', 'main_use', 'main_building_material', 

                   'complete_year', 'transaction_year', 'transaction_month', 'carpark_category']
for ind, col in enumerate(categorical_col):

    print("Unique values of {}: {} \n".format(col, set(df[col])))
plt.figure(figsize = (20, 30))

for ind, col in enumerate(categorical_col):

    plt.subplot(3, 3, ind+1)

    df[col].value_counts().plot(kind='bar')

    plt.xlabel(col, size=10)

    plt.ylabel("counts")

    plt.tight_layout() # to avoid graph overlapping

# describe how an algorithm works in a simple way