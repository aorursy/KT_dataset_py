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
df = pd.read_csv("/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")
df = df.drop_duplicates(subset='lesion_id')
df.shape
len(df['lesion_id'].value_counts())
df_temp = df.groupby('lesion_id').count()

# now we filter out lesion_id's that have only one image associated with it
df_temp = df_temp[df_temp['image_id'] == 1]

df_temp.reset_index(inplace=True)

df_temp.head()


def identify_duplicates(x):
    
    unique_list = list(df_temp['lesion_id'])
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
    
df['duplicates'] = df['lesion_id']
df['duplicates'] = df['duplicates'].apply(identify_duplicates)
df.head()
df['duplicates'].value_counts()
df['dx'].value_counts()
df
df.to_csv("./filtered_metadata.csv")
!ls
