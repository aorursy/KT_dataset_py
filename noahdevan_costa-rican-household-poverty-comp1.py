# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_samp = pd.read_csv("../input/sample_submission.csv")
print("Rows & columns of sample_submission ", df_samp.shape)
df_samp.head()
df_test = pd.read_csv("../input/test.csv", index_col = 0)
print("Rows & columns of test ", df_test.shape)
df_test.head()
df_test = pd.read_csv("../input/test.csv")
#df_test['parentesco1'].head(10)
print("Rows & columns of train ", df_test.shape)
df_test.describe()
df_train = pd.read_csv("../input/train.csv")
df_train.head()
sns.countplot(df_train['Target'])
sns.distplot(df_train['Target'])
#df_test['idhogar'].value_counts().head(10).plot.bar()
#df_test['idhogar'].value_counts().head(10).plot.line()
#df_test['idhogar'].value_counts().sort_index().head(10).plot.bar()
#df_test['idhogar'].value_counts().head(10).plot.area()
df_test['idhogar'].value_counts().head(10).plot.pie(figsize=(12, 12))
df_test['hacdor'].value_counts().plot.pie(figsize=(12, 12))
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))

df_test['hacdor'].value_counts().head(10).plot.pie(
ax = axarr[0][0])
axarr[0][0].set_title("Pie chart", fontsize=18)

df_test['hacdor'].value_counts().head(10).plot.bar(
    ax = axarr[0][1])
axarr[0][1].set_title("Bar chart", fontsize=18)

df_test['idhogar'].value_counts().head(10).plot.line(
    ax = axarr[1][0])
axarr[1][0].set_title("Line chart", fontsize=18)
df_test['idhogar'].value_counts().head(10).plot.area(
    ax = axarr[1][1])
axarr[1][1].set_title("Area chart", fontsize=18)
df = df_train[df_train['Target'].isin(['1', '2', '3', '4'])]
g = sns.FacetGrid(df, col="Target")
g.map(sns.kdeplot, "rooms")
sns.pairplot(df_train[['Target', 'rooms', 'refrig', 'v18q', 'computer', 'television', 'mobilephone']])
sns.lmplot(x='Target', y='refrig', markers=['o', 'x'], hue='mobilephone',
           data=df_train, fit_reg=True
          )
sns.heatmap(
    df_train.loc[:, ['Target', 'rooms', 'refrig', 'v18q', 'computer', 'television', 'mobilephone' ]].corr(),
    annot=True
)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import plotly.graph_objs as go

iplot([go.Scatter(x=df_train['Target'], y=df_train['rooms'], mode='markers')])
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))

df_test['hacdor'].value_counts().plot.pie(
ax = axarr[0][0])
axarr[0][0].set_title("Overcrowd", fontsize=18)

df_test['rooms'].value_counts().plot.bar(
    ax = axarr[0][1])
axarr[0][1].set_title("No of rooms", fontsize=18)

df_test['v18q1'].value_counts().plot.line(
    ax = axarr[1][0])
axarr[1][0].set_title("No of tablets", fontsize=18)
df_test['r4h3'].value_counts().plot.area(
    ax = axarr[1][1])
axarr[1][1].set_title("Males", fontsize=18)
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))

df_test['tamhog'].value_counts().plot.bar(
ax = axarr[0][0])
axarr[0][0].set_title("Size of the household", fontsize=18)

df_test['qmobilephone'].value_counts().plot.bar(
ax = axarr[0][1])
axarr[0][1].set_title("No of mobile phones", fontsize=18)