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
pub_df = pd.read_csv('/kaggle/input/us-schools-dataset/Public_Schools.csv')

pvt_df = pd.read_csv('/kaggle/input/us-schools-dataset/Private_Schools.csv')
pub_df.drop(['X','Y'], axis=1, inplace=True)

pvt_df.drop(['X','Y'], axis=1, inplace=True)
# Since 'OBJECT_ID' was observed to contain unique numbers(equal to total number of records) in serial/order



pub_df.set_index(['OBJECTID'], inplace=True)

pvt_df.set_index(['OBJECTID'], inplace=True)
# all the records have the same description - all are `ELEMENTARY AND SECONDARY SCHOOLS`

print(pub_df.NAICS_CODE.value_counts())

print(pvt_df.NAICS_CODE.value_counts())

print(pub_df.NAICS_DESC.value_counts())

print(pvt_df.NAICS_DESC.value_counts())
pub_df.drop(['NAICS_CODE','NAICS_DESC'], axis=1, inplace=True)

pvt_df.drop(['NAICS_CODE','NAICS_DESC'], axis=1, inplace=True)
pub_df['TYPE'].value_counts()
groups=pub_df.groupby('TYPE')



pd.set_option('display.max_colwidth', None)

for name, df in groups:

    print(name, df.iloc[0][['NCESID','NAME','SOURCE']], sep='\n')
# doing the same for private schools

pvt_df['TYPE'].value_counts()
pvt_groups = pvt_df.groupby('TYPE')

for name,df in pvt_groups:

    print(name, df.iloc[0][['NCESID','NAME','SOURCE']], sep='\n')
pub_df['STATUS'].value_counts()
pub_df.columns
pub_df[['ZIP','ZIP4']].sample(10)
pub_df.NCESID.nunique(),pvt_df.NCESID.nunique()