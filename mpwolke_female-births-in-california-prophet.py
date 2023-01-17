#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

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
df = pd.read_csv('../input/female-births-in-ca/daily-total-female-births-ca.csv')

df.head()
print(f"data shape: {df.shape}")
df.describe()
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("Daily total births in California 1959", "Daily total births in California 1959", df,4)
px.histogram(df, x='Daily total births in California 1959', color='Date')
fig = px.bar(df, 

             x='Date', y='Daily total births in California 1959', color_discrete_sequence=['#D63230'],

             title='Daily Births in California in 1959', text='Daily total births in California 1959')

fig.show()
fig = px.line(df, x="Date", y="Daily total births in California 1959", 

              title="Daily Births in California 1959")

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



cols_selected = ['Date']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Daily total births in California 1959']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
df['Daily total births in California 1959'].hist(figsize=(10,4), color='g', bins=20)
import matplotlib.ticker as ticker

ax = sns.distplot(df['Daily total births in California 1959'], color='r')

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



sb.distplot(df['Daily total births in California 1959'], fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['Daily total births in California 1959'], plot=plt)

plt.show()
from fbprophet import Prophet

df1=df.rename(columns={"Date": "ds", "Daily total births in California 1959": "y"})

df1

m = Prophet()

m.fit(df1)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
_ = pd.pivot_table(df, values='Daily total births in California 1959', index='Date').plot(style='-o', title="Daily Births in California 1959")

plt.xticks(rotation=45)