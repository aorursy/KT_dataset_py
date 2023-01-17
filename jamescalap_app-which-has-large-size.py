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
#importing required packages 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#crating a Dataframe for the .csv file 

df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv',)

df.head()
#we can that in Size column other than number a string (Varies with device) is occurred

df.sort_values('Size',ascending=False)
#so there are 1695 rows having  string "Varies with device"

df.loc[df['Size'] == 'Varies with device'].shape
#we separatly take those rows

unwanted_data = df.loc[df['Size'] == 'Varies with device']

unwanted_data.shape

#And we drope the rows which has the string "Varies with device"

#Because we cannot determine size with that 

df.drop(unwanted_data.index,inplace = True)
df.sort_values('Size',ascending=False).head()
#now the values has charecters like "M" "k" "+"

#so we need to remove them inorder to convert the size column to numerics

df['Size'] = df['Size'].apply(lambda x: str(x).replace('M',''))

df['Size'] = df['Size'].apply(lambda x: str(x).replace('k',''))

df['Size'] = df['Size'].apply(lambda x: str(x).replace('+',''))
df.head()
#now converting Size column to numerics

df['Size'] = pd.to_numeric(df['Size'],errors='coerce')
#now We got Large Size app from app store

df.sort_values('Size', ascending=False).iloc[0]['App']