# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

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
df = pd.read_csv('../input/kensho-ohio-voter-project/oh_counties_black_2018.csv', encoding='ISO-8859-2')

df.head()
df.isnull().sum()
# Distribution of different type of amount

fig , ax = plt.subplots(1,3,figsize = (12,5))



state_fips = df.state_fips.values

county_fips= df.county_fips.values

value = df.value.values



sns.distplot(state_fips , ax = ax[0] , color = 'blue').set_title('State Fips' , fontsize = 14)

sns.distplot(county_fips , ax = ax[1] , color = 'cyan').set_title('County Fips' , fontsize = 14)

sns.distplot(value , ax = ax[2] , color = 'purple').set_title('Value' , fontsize = 14)





plt.show()
import matplotlib.gridspec as gridspec

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler,MinMaxScaler

from scipy import stats

import matplotlib.style as style

style.use('seaborn-colorblind')
def plotting_3_chart(df, feature): 

    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,6))

    ## crea,ting a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

 



print('Skewness: '+ str(df['value'].skew())) 

print("Kurtosis: " + str(df['value'].kurt()))

plotting_3_chart(df, 'value')
stats.probplot(df['state_fips'].values, dist="norm", plot=plt)

plt.show()
stats.probplot(df['county_fips'].values, dist="norm", plot=plt)

plt.show()
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap='cubehelix')

plt.title('Correlations', fontsize = 20)

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)



ax1.plot(df.geoid, df.value, c = 'green')

ax1.set_title('Geoid vs. Value', c = 'green')

ax2.scatter(df.county_fips, df.value, c='red')

ax2.set_title('County Fips vs. Value', c ='red')



plt.ylabel('County fips', fontsize = 20)



plt.show()
sns.boxplot(x=df['value'], color = 'cyan')

plt.title('Value Boxplot', fontsize = 20)

plt.show()
sns.boxplot(x=df['county_fips'], color = 'magenta')

plt.title('County Fips Boxplot', fontsize = 20)

plt.show()
#Code from Mario Filho

from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['value']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['county_fips']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
ax = df.groupby('value')['county_fips'].mean().plot(kind='barh', figsize=(12,8),

                                                           title='Mean estimated Value')

plt.xlabel('Mean estimated Value')

plt.ylabel('County Fips')

plt.show()
ax = df.groupby('value')['state_fips', 'county_fips'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='DATA EXPLORATION')

plt.xlabel('DATA EXPLORATION')

plt.ylabel('Log')



plt.show()
ax = df.groupby('value')['county_fips', 'state_fips'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='', logx=True, linewidth=3)

plt.xlabel('Log')

plt.ylabel('DATA EXPLORATION')

plt.show()
fig=sns.lmplot(x='value', y="county_fips",data=df)
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
fig = px.bar(df,

             y='county_fips',

             x='value',

             orientation='h',

             color='geoid',

             title='DATA EXPLORATION',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.bar(df,

             y='state_fips',

             x='geoid',

             orientation='h',

             color='state_code',

             title='DATA EXPLORATION',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.bar(df, 

             x='geoid', y='county_fips', color_discrete_sequence=['#27F1E7'],

             title='DATA EXPLORATION', text='value')

fig.show()
fig = px.bar(df, 

             x='acs_primary_id', y='county_fips', color_discrete_sequence=['crimson'],

             title='DATA EXPLORATION', text='geoid')

fig.show()
fig = px.line(df, x="geoid", y="county_fips", color_discrete_sequence=['darkseagreen'], 

              title="DATA EXPLORATION")

fig.show()