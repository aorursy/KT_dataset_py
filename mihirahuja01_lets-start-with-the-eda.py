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

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt  

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df
#1460 Rows

#81 Columns
import plotly.express as px

fig = px.histogram(df, x="SalePrice")

fig.show()
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='Neighborhood', y='SalePrice', data=df, showfliers=False);
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='OverallQual', y='SalePrice', data=df, showfliers=False);
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='OverallCond', y='SalePrice', data=df, showfliers=False);
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='MSZoning', y='SalePrice', data=df, showfliers=False);
import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(y=df['LotArea'], x=df['SalePrice'],

                    mode='markers', name='markers'))



fig, axes = plt.subplots(1, 1, figsize=(40, 6))

temp = df[['YearBuilt','SalePrice']]

temp = temp[temp['YearBuilt']>1950]

sns.boxplot(x='YearBuilt', y='SalePrice', data=temp, showfliers=False);
import plotly.express as px

fig = px.histogram(df, x="LotArea")

fig.show()
pd.set_option('display.max_columns', None)



df.head()
numerical_list = ['LotArea','MasVnrArea','GrLivArea','BedroomAbvGr','GarageArea','YrSold']

for i in numerical_list:

    sns.distplot(df[i], hist=False, rug=True)

    plt.show();
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='GarageCond', y='SalePrice', data=df, showfliers=False);
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='GarageQual', y='SalePrice', data=df, showfliers=False);
import plotly.express as px

fig = px.scatter(df, x="GarageArea", y="SalePrice",color='GarageArea')

fig.show()
#GarageType	GarageYrBlt	GarageFinish	GarageCars
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='GarageType', y='SalePrice', data=df, showfliers=False);
import plotly.express as px

fig = px.scatter(df, x="GarageYrBlt", y="SalePrice",color="GarageYrBlt")

fig.show()
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='GarageFinish', y='SalePrice', data=df, showfliers=False);


fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='GarageCars', y='SalePrice', data=df, showfliers=False);