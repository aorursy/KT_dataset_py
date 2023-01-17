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
filepath ='/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv'
df = pd.read_csv(filepath)
df.head()
df.shape
df.dtypes
df['Tags']
df['Tags'] = df['Tags'].apply(lambda x:x.replace('><',','))
df['Tags'] = df['Tags'].apply(lambda x:x.replace('>',''))
df['Tags'] = df['Tags'].apply(lambda x:x.replace('<',''))

df.head()
def split_count(val):
    z=[]
    for i in range(len(val)):
        z.extend(val[i].split(','))
    return z    
        
df1['Tags']
top_tags = split_count(df['Tags'])
top_tags = pd.DataFrame(top_tags,columns=['top'])
top_tags['top'].value_counts().nlargest(10)
import plotly.express as px
import seaborn as sns
import matplotlib. pyplot as plt
%matplotlib inline 
px.bar(top_tags['top'].value_counts().nlargest(20),labels='counts')

df['Y'].value_counts()
df1 = df[df['Y']=='HQ'].reset_index().drop(['index'],axis  =1)
df2 = df[df['Y']=='LQ_CLOSE'].reset_index().drop(['index'],axis  =1)
df3 = df[df['Y']=='LQ_EDIT'].reset_index().drop(['index'],axis  =1)
df1.head()
split_count(df1['Tags'])
top_tags = pd.DataFrame(top_tags,columns=['top'])
top_tags['top'].value_counts().nlargest(10)
top_20_hq = pd.DataFrame(split_count(df1['Tags']),columns=['val'])
top_20_hq['val'].value_counts().nlargest(20)
px.bar(top_20_hq['val'].value_counts().nlargest(20))
top_20_lq_close = pd.DataFrame(split_count(df2['Tags']),columns=['val'])
top_20_lq_close['val'].value_counts().head(20)
px.bar(top_20_lq_close['val'].value_counts().head(20))
top_20_lq_edit = pd.DataFrame(split_count(df3['Tags']),columns=['val'])
top = top_20_lq_edit['val'].value_counts().head(20)
top
px.bar(top_20_lq_edit['val'].value_counts().head(20))
top.values.tolist()
top.index.tolist()
import plotly.graph_objects as go
def donut_plot(val):
    fig  =go.Figure(data=[go.Pie(labels =val.index.tolist(),values = val.values.tolist(), hole = 0.3)])
    fig.show()
donut_plot(top_20_hq['val'].value_counts().nlargest(20))
fig  =go.Figure(data=[go.Pie(labels =top.index.tolist(),values = top.values.tolist(), hole = 0.3)])
fig.show()
