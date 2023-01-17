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
os.chdir('/kaggle/input/buildingdatagenomeproject2')

os.listdir()
solar = pd.read_csv('solar_cleaned.csv')

print(solar.shape)

solar.head()
solar.dtypes
solar.info()
solar.describe()
# getting rid of the Bobcat_office_Justine column

solar = solar.drop('Bobcat_office_Justine', axis=1)
# making timestamp a datetime

solar['timestamp'] = solar['timestamp'].astype('datetime64')

solar.dtypes
solar.head(20)
print(solar.isin([0]).sum())
import missingno as msno

msno.matrix(solar);
solar.isnull().sum()
solar = solar.interpolate(method="slinear")

solar.isnull().sum()
import missingno as msno

msno.matrix(solar);
#save as csv

solar.to_csv('/kaggle/working/solar_cleaned2.csv')