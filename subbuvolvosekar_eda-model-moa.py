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
import plotly.express as px
train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train.head()
print("Total numper of Drug Samples :",train.sig_id.nunique())
cp = train.groupby('cp_type')['sig_id'].count().reset_index(name = 'count')

fig = px.pie(cp, values='count', names='cp_type', title='Cp_Type')

fig.show()
cp_time = train.groupby('cp_time')['sig_id'].count().reset_index(name = 'count')

fig = px.pie(cp_time, values='count', names='cp_time', title='Cp_Time')

fig.show()
cp_dose = train.groupby('cp_dose')['sig_id'].count().reset_index(name = 'count')

fig = px.pie(cp_dose, values='count', names='cp_dose', title='Cp_Dose')

fig.show()
cp = train.groupby(['cp_type','cp_time'])['sig_id'].count().reset_index(name = 'count')

fig = px.bar(cp, x="cp_type", y="count", color="cp_time")

fig.show()
cp = train.groupby(['cp_type','cp_dose'])['sig_id'].count().reset_index(name = 'count')

fig = px.bar(cp, x="cp_type", y="count", color="cp_dose")

fig.show()
cp = train.groupby(['cp_dose','cp_time'])['sig_id'].count().reset_index(name = 'count')

fig = px.bar(cp, x="cp_time", y="count", color="cp_dose")

fig.show()
train.head()
g_col = [col for col in train if col.startswith('g-')]

print("Gene Expression data count : ",+len(g_col))



c_col = [col for col in train if col.startswith('c-')]

print("Cell Expression data count : ",+len(c_col))
fig = px.line(train[['sig_id','g-5']], x="sig_id", y="g-5", title='Gene vs Sigid:G-5')

fig.show()
fig = px.line(train[['sig_id','g-500']], x="sig_id", y="g-500", title='Gene vs Sigid:g-500')

fig.show()
fig = px.line(train[['sig_id','g-100']], x="sig_id", y="g-100", title='Gene vs Sigid:g-100')

fig.show()
fig = px.line(train[['sig_id','g-700']], x="sig_id", y="g-700", title='Gene vs Sigid:g-700')

fig.show()
one = train[train['sig_id']=='id_000779bfc']

one_ = one[g_col].T

fig1 = px.line(one_, x=None, y=1, title='one sig_id')

fig1.show()
tow = train[train['sig_id']=='id_001626bd3']

fig = px.line(tow[g_col].T, x=None, y=4, title='one sig_id')

fig.show()
fig = px.line(train[['sig_id','c-5']], x="sig_id", y="c-5", title='Gene vs Sigid:c-5')

fig.show()
fig = px.line(train[['sig_id','c-50']], x="sig_id", y="c-50", title='Gene vs Sigid:C-50')

fig.show()
one = train[train['sig_id']=='id_000779bfc']

one_ = one[c_col].T

fig1 = px.line(one_, x=None, y=1, title='one sig_id')

fig1.show()
one = train[train['sig_id']=='id_0015fd391']

one_ = one[c_col].T

fig1 = px.line(one_, x=None, y=3, title='one sig_id')

fig1.show()
target = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

cols = target.columns.to_list()

cols.remove('sig_id')

df = pd.DataFrame(columns = ['item','count'])

it = []

cnt = []

for item in cols:

    it.append(item)

    cnt.append(target[item].sum())

df['item'] = it

df['count'] = cnt

#print(df)



fig = px.bar(df, y='count', x='item', text='count')

fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()