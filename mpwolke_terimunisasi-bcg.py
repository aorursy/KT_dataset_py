#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTw-LIs0eT22FJw5AY3XJXoxBFcj92AKjnURsKPbn4sh_yXx-Sp&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/balita-terimunisasi-di-indonesia-bps-19952017/Balita Terimunisasi BCG 1995-2017.csv', encoding='ISO-8859-2')

df.head()
px.histogram(df, x='Tahun', color='% Balita yang pernah mendapat imunisasi BCG')
fig = px.bar(df, 

             x='% Balita yang pernah mendapat imunisasi BCG', y='Tahun', color_discrete_sequence=['#D63230'],

             title='Indonesia Terimunisasi BCG', text='% Balita yang pernah mendapat imunisasi BCG')

fig.show()
#seaborn.set(rc={'axes.facecolor':'3F3FBF', 'figure.facecolor':'3F3FBF'})

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Terimunisasi BCG")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['% Balita yang pernah mendapat imunisasi BCG'])



# Add label for vertical axis

plt.ylabel("Tahun")
fig = px.line(df, x="% Balita yang pernah mendapat imunisasi BCG", y="Tahun", 

              title="% Balita yang pernah mendapat imunisasi BCG")

fig.show()
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['% Balita yang pernah mendapat imunisasi BCG']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Tahun']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
df['Tahun'].hist(figsize=(10,4), bins=20)
ax = df['% Balita yang pernah mendapat imunisasi BCG'].value_counts().plot.barh(figsize=(10, 4))

ax.set_title('% Balita yang pernah mendapat imunisasi BCG Distribution', size=18)

ax.set_ylabel('% Balita yang pernah mendapat imunisasi BCG', size=10)

ax.set_xlabel('Tahun', size=10)
import matplotlib.ticker as ticker

ax = sns.distplot(df['Tahun'])

plt.xticks(rotation=45)

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

figsize=(10, 4)
from scipy.stats import norm, skew #for some statistics

import seaborn as sb

from scipy import stats #qqplot

#Lets check the ditribution of the target variable (Placement?)

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 4,2



sb.distplot(df['Tahun'], fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['Tahun'], plot=plt)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTBn9zjAnSB3h2NoNLonDvaasz4zAZh08xwQSR4BKUkV9uLNeQD&usqp=CAU',width=400,height=400)