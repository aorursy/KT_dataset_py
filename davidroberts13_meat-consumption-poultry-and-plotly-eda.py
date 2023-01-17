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



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import cufflinks as cf

cf.go_offline()

import plotly.express as px
df=pd.read_csv('/kaggle/input/meatconsumption/meat_consumption_worldwide.csv')
df.head()
df.info()
df['LOCATION'].unique()
df['LOCATION'].nunique()
fig = px.scatter(df, x="TIME", y="Value", hover_name='LOCATION',hover_data=['MEASURE'],color='SUBJECT',title='Meat Consumption Change through the Years')

fig.show()
df1=df[df['LOCATION'].isin(['WLD','BRICS','OECD','EU28'])]
fig = px.scatter(df1, x="TIME", y="Value", hover_name='SUBJECT',hover_data=['MEASURE'],color='SUBJECT',symbol='LOCATION',

                title='Aggrogate Measurments of Meat Production')

fig.show()
df2=df[df['LOCATION'].isin(['WLD','BRICS','OECD','EU28'])==False]
fig = px.scatter(df2, x="TIME", y="Value",symbol='SUBJECT',hover_data=['MEASURE'],color='LOCATION',hover_name='SUBJECT',

                title='Meat Production by Country and Type')

fig.show()