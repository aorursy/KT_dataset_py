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
df = pd.read_excel('/kaggle/input/english-words/english.xlsx')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(df['Word_len'], color='blue')

plt.title(f"Word_len Distribution [\u03BC : {df['Word_len'].mean():.2f} length | \u03C3 : {df['Word_len'].std():.2f} length]")

plt.xlabel("Word_len")

plt.ylabel("Count")

plt.show()
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(df['raw_len'], color='red')

plt.title(f"raw_len Distribution [\u03BC : {df['raw_len'].mean():.2f} length | \u03C3 : {df['raw_len'].std():.2f} length]")

plt.xlabel("raw_len")

plt.ylabel("Count")

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

 



print('Skewness: '+ str(df['character_len'].skew())) 

print("Kurtosis: " + str(df['character_len'].kurt()))

plotting_3_chart(df, 'character_len')
# Scatter Plot

fig = go.Figure(data=go.Scatter(x=df['index'],y=df['raw_len'],mode='markers',marker=dict(size=10,color=df['Word_len']),text=df['character_len']))

fig.update_layout(title='Index & Raw length',xaxis_title='Index Rating',yaxis_title='Raw lenght')

fig.show()