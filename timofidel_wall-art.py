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
# reading csv



df = pd.read_csv('../input/wall-art-sales/Wall Art sales - Sheet1.csv')

df

df.info()

# checking null values



print(df.isnull().sum())
df = df.dropna()
# Droping redundant rows 

df = df.drop_duplicates()
# renaming the column



df = df.rename(columns = {'Link':'Product'})

df
df.dtypes
# cleaning the data



import re



df['Shipping'] = df['Shipping'].apply( lambda x : int(re.findall( '\d+' , x )[0]) )

df['Discount'] = df['Discount'].apply(lambda x : int(re.findall('\d+' , x )[0]))

df['Price'] = df['Price'].apply(lambda x : int(re.sub(',','',x)[1:]))

df['Brand'] = df['Brand'].apply( lambda x: re.sub('by' , '' , x))

df
# Identifying how strong each variable are related

import seaborn as sns

sns.heatmap(df.corr())
# Maximum discount provided by each Brand

import plotly.express as px

px.scatter(df.groupby(['Brand']).max()['Discount'])
# Minimum discount provided by each Brand

px.scatter(df.groupby(['Brand']).min()['Discount'])