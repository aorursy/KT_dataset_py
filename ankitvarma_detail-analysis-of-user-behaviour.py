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
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
df = pd.read_csv('../input/groceries-dataset/Groceries_dataset.csv', parse_dates=['Date'])
df.head()
df.dtypes
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False);
df.info()
len(df['itemDescription'].unique())
len(df['Member_number'].unique())
df['itemDescription'].value_counts().head(10)
px.bar(df,df['itemDescription'].value_counts().head(10).index,df['itemDescription'].value_counts().head(10).values)
df['itemDescription'][0].split()[1] == 'vegetables '
df['itemDescription']
df['itemDescription'] = df['itemDescription'].astype(str)
def chek_val(val):
    return df[df['itemDescription'].str.contains(val)]['itemDescription'].value_counts()
chek_val('fruit')
px.bar(chek_val('fruit'),chek_val('fruit').index,chek_val('fruit').values)
chek_val('vegetable')
px.bar(chek_val('vegetable'),chek_val('vegetable').index,chek_val('vegetable').values)
z = []
for i in range(len(val)):
    x = val[i].split()
    z.extend(x)
z = pd.DataFrame(z,columns=['value'])
z.head()
val = z['value'].value_counts().head(10)
val
px.bar(val,val.index,val.values)
chek_val('milk')
df.head()
df[df['Member_number'] ==1808]
df1 = df
df1['itemDescription']  =df1['itemDescription'].apply(lambda x:x+', ')
val = df1.groupby(['Member_number','Date'])['itemDescription'].sum()
val
val = pd.DataFrame(val).reset_index()
val
len(val['itemDescription'][0].split(','))-1
val['itemDescription'][0]
val['items_bought'] = val['itemDescription'].apply(lambda x :len(x.split(','))-1)
val.head()
val['itemDescription'] = val['itemDescription'].apply(lambda x:x[0:-2])
val
