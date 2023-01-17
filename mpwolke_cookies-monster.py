# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

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
df = pd.read_excel('/kaggle/input/cookie-business/cookie_business.xlsx')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
df = df.rename(columns={'Customer ID':'ID', 'Cookies bought each week': 'weekly', 'Favourite Cookie': 'favourite'})
# Distribution of different type of amount

fig , ax = plt.subplots(1,3,figsize = (12,5))



ID = df.ID.values

Age = df.Age.values

weekly = df.weekly.values



sns.distplot(ID , ax = ax[0] , color = 'blue').set_title('Cookies Customer ID' , fontsize = 14)

sns.distplot(Age , ax = ax[1] , color = 'cyan').set_title('Cookies Customer Age' , fontsize = 14)

sns.distplot(weekly , ax = ax[2] , color = 'purple').set_title('Cookies bought weekly' , fontsize = 14)





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

 



print('Skewness: '+ str(df['weekly'].skew())) 

print("Kurtosis: " + str(df['weekly'].kurt()))

plotting_3_chart(df, 'weekly')
train_heat=df[df["weekly"].notnull()]

train_heat=train_heat.drop(["weekly"],axis=1)

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (10,8))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train_heat.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train_heat.corr(), 

            cmap=sns.diverging_palette(255, 133, l=60, n=7), 

            mask = mask, 

            annot=True, 

            center = 0, 

           );

## Give title. 

plt.title("Heatmap of All Cookies", fontsize = 30);
fig = px.bar(df, 

             x='weekly', y='Age', color_discrete_sequence=['#2B3A67'],

             title='Cookies!!!', text='Postcode')

fig.show()
fig = px.bar(df, 

             x='Age', y='weekly', color_discrete_sequence=['crimson'],

             title='Cookies!!!', text='ID')

fig.show()
ax = df.groupby('weekly')['Age'].mean().plot(kind='barh', figsize=(12,8),

                                                           title='Cookies Weekly')

plt.xlabel('Mean estimated Cookies Weekly')

plt.ylabel('Count')

plt.show()
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['weekly']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Age']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
def plot_weekly(col, df, title):

    fig, ax = plt.subplots(figsize=(18,6))

    df.groupby(['weekly'])[col].sum().plot(rot=45, kind='bar', ax=ax, legend=True, cmap='bone')

    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

    ax.set(Title=title, xlabel='weekly')

    return ax
plot_weekly('weekly', df, 'Cookies bought each week');
ax = df.groupby('weekly')['ID', 'Age'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Cookies')

plt.xlabel('Cookies bought each week')

plt.ylabel('Count Log')



plt.show()
#Code from Gabriel Preda

#plt.style.use('dark_background')

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set2')

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
plot_count("favourite", "Favourite Cookie", df,4)