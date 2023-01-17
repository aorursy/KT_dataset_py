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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
%matplotlib inline
df = pd.read_csv("../input/windows-store/msft.csv")
df.head()
df.shape
df.info()
df.isnull()
df.iloc[5321]
df.drop([5321], inplace = True)
df.info()
df.duplicated()
df.duplicated().sum()
df.columns
df.rename(columns={'No of people Rated' : 'No_of_people_Rated'} , inplace = True)
df.describe()
df.dtypes
for i, v in enumerate(df.columns):
    print(i, v , type(df[v][1]))
import datetime
df['Date'] = pd.to_datetime(df['Date'] , format='%d-%m-%Y')
type(df['Date'][1])
df.dtypes
df.Price.unique()
df['Price'] = df.Price.replace('Free', 0)
df.Price.unique()
for i in range (0 , df.shape[0]):
    r = df['Price'][i]
#     print(r)
    if (r !=0):
#         print(r[2:])
        df['Price'][i] = r[2:-3]
    else:
        pass
df['Price'].unique()
for i in range (0 , df.shape[0]):
    r = df['Price'][i]
#     print(r)
    if (r !=0):
        df['Price'][i] = df['Price'][i].replace(',' , '')
df['Price'].unique()
df['Price'] = df['Price'].astype(int)
df.info()
df.Category.unique()
df.Category.value_counts()
df.Name.unique()
df.Name.value_counts()
df[df['Name']=='http://microsoft.com']
