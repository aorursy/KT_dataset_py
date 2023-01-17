# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import gc

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', 150)

pd.set_option('max_rows', 150)
import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

from scipy import stats

#To plot figs on jupyter

%matplotlib inline

# figure size in inches

rcParams['figure.figsize'] = 14,6



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', 200)

pd.set_option('max_rows', 200)
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH =   r"/kaggle/input/features/FEATURES-2014-2015/"

df = pd.read_csv(PATH+"part-067.csv")

df.shape
df.columns
df.head()
df.dtypes
df.isAnomaly.value_counts()
df[df['isAnomaly']==True].head()
df[df['isAnomaly']==True].describe()
from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected = True)



def line_plot_check_nan(df1, df2, x, y, title, width, height):

    

    trace1 = go.Scatter(

        x = df1[x],

        y = df1[y],

        mode='lines',

        name='with_nans',

        marker = dict(

            color = '#1E90FF', 

        ), 

    )

    

    df3 = df2.dropna()

    trace2 = go.Scatter(

        x = df3[x],

        y = df3[y],

        mode='markers',

        name='no_nans',

        marker = dict(

            color = 'red', 

        ), 

    )

    

    layout = go.Layout(

        title = go.layout.Title(

            text = title,

            x = 0.5

        ),

        font = dict(size = 14),

        width = width,

        height = height,

    )

    

    data = [trace1, trace2]

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig, filename = 'line_plot')
line_plot_check_nan(df[df['isAnomaly']==True], df[df['isAnomaly']==True], 'timestamp', "Heap usage activity : (d/dx (MXBean(java.lang:type=Memory).HeapMemoryUsage.used))"," Memory space usage by date", 1400, 600)
line_plot_check_nan(df[df['isAnomaly']==False], df[df['isAnomaly']==False], 'timestamp', "Heap usage activity : (d/dx (MXBean(java.lang:type=Memory).HeapMemoryUsage.used))"," Memory space usage by date", 1400, 600)
# Plots the disribution of a variable colored by value of the target

def kde_target(var_name, df):

    

    # Calculate the correlation coefficient between the new variable and the target

    corr = df['isAnomaly'].corr(df[var_name])

    

    # Calculate medians for repaid vs not repaid

    avg_highr = df.loc[df['isAnomaly'] == 0, var_name].median()

    avg_lowr = df.loc[df['isAnomaly'] == 1, var_name].median()

    

    plt.figure(figsize = (12, 6))

    

    # Plot the distribution for target == 0 and target == 1

    sns.kdeplot(df.loc[df['isAnomaly'] == 0, var_name], label = 'isAnomaly == 0')

    sns.kdeplot(df.loc[df['isAnomaly'] == 1, var_name], label = 'isAnomaly == 1')

    

    # label the plot

    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)

    plt.legend();

    

    # print out the correlation

    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))

    # Print out average values

    print('Median value for request with high runtime value = %0.4f' % avg_highr)

    print('Median value for request with low runtime value =     %0.4f' % avg_lowr)

    



kde_target('Heap usage activity : (d/dx (MXBean(java.lang:type=Memory).HeapMemoryUsage.used))', df[['Heap usage activity : (d/dx (MXBean(java.lang:type=Memory).HeapMemoryUsage.used))','isAnomaly']].dropna(),)