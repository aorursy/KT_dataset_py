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
df = pd.read_csv('../input/hackathon/task_2-Tuberculosis_infection_estimates_for_2018.csv')

df.head()
df.isnull().sum()
del df['ptsurvey_newinc']

del df['ptsurvey_newinc_con04_prevtx']
# filling missing values with NA

df[['prevtx_data_available', 'newinc_con04_prevtx', 'newinc_con04_prevtx', 'e_prevtx_eligible', 'e_prevtx_eligible_lo', 'e_prevtx_eligible_hi', 'e_prevtx_kids_pct', 'e_prevtx_kids_pct_lo', 'e_prevtx_kids_pct_hi']] = df[['prevtx_data_available', 'newinc_con04_prevtx', 'newinc_con04_prevtx', 'e_prevtx_eligible', 'e_prevtx_eligible_lo', 'e_prevtx_eligible_hi', 'e_prevtx_kids_pct', 'e_prevtx_kids_pct_lo', 'e_prevtx_kids_pct_hi']].fillna('NA')
# Let's See The Correlation Among The Features .



# Below chart is used to visualize how one feature is correlated with every other Features Present in the dataset .

# if we have two highly correlated features then we will consider only one of them to avoid overfitting .



# since in our Dataset There is now two  features which are highly correlated ,

# hence we have consider all the features for training our Model .





plt.rcParams['figure.figsize'] = (10, 6)

sns.heatmap(df.corr(),annot = True ,cmap = 'summer',annot_kws = {"Size":14})

plt.title( "Chart Shows Correlation Among Features   : ")
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['e_prevtx_kids_pct']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['iso_numeric']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['e_prevtx_eligible']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['iso_numeric']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
ax = df.groupby('g_whoregion')['e_hh_size'].mean().plot(kind='barh', figsize=(12,8),

                                                           title='Mean estimated household size by Who region')

plt.xlabel('Mean estimated household size')

plt.ylabel('Who region')

plt.show()
ax = df.groupby('g_whoregion')['e_hh_size'].min().sort_values(ascending=True).plot(kind='barh', figsize=(12,8), color='r',

                                                                                  title='Min.estimated household size by Who region')

plt.xlabel('Min.estimated household size')

plt.ylabel('Who region')

plt.show()
ax = df.groupby('g_whoregion')['e_hh_size'].max().sort_values(ascending=True).plot(kind='barh', figsize=(12,8), color='g',

                                                                                   title='Max. estimated household size by Who region')

plt.xlabel('Max. estimated household size ')

plt.ylabel('Who Region')

plt.show()
ax = df.groupby('g_whoregion')['e_prevtx_kids_pct', 'e_prevtx_kids_pct_hi'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Children received TB preventive therapy under 5 by Who Region')

plt.xlabel('g_whoregion')

plt.ylabel('Log Scale Children under 5 received TB preventive therapy')



plt.show()
ax = df.groupby('g_whoregion')['e_prevtx_eligible', 'e_prevtx_eligible_hi'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Children under 5, household contacts/elegible for TB by Who Region')

plt.xlabel('g_whoregion')

plt.ylabel('Log Scale Children under 5 who are household contacts of TB cases and eligible for TB')



plt.show()
ax = df.groupby('g_whoregion')['e_prevtx_kids_pct_hi', 'e_prevtx_eligible_hi'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Children under 5, household contacts/elegible for TB by Who Region')

plt.xlabel('g_whoregion')

plt.ylabel('Log Scale Children under 5 who are household contacts of TB cases high eligibility ')



plt.show()
ax = df.groupby('g_whoregion')['e_prevtx_eligible', 'e_prevtx_eligible_lo'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='', logx=True, linewidth=3)

plt.xlabel('Log Scale Children under 5 who are household contacts of TB cases eligible for TB treatment ')

plt.ylabel('Who Region')

plt.show()
ax = df.groupby('g_whoregion')['e_hh_size'].mean().sort_values(ascending=True).plot(kind='barh', figsize=(20,6), color='g',

                                                                                    title='Mean estimated household size by Who region ')

plt.xlabel('Estimated average household size')

plt.ylabel('Who Regions')

plt.show()
df["iso_numeric"].plot.hist()

plt.show()
fig=sns.lmplot(x="iso_numeric", y="e_hh_size",data=df)
import matplotlib.ticker as ticker

ax = sns.distplot(df['iso_numeric'])

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



sb.distplot(df['iso_numeric'], fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['iso_numeric'], plot=plt)

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.g_whoregion)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()