# Including a title Image
from IPython.display import Image
%matplotlib inline
Image("../input/venture-funding-image/vcf.jpg", width=500, height=500)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plotly
# plotly standard imports
import plotly.graph_objs as go
import plotly.plotly as py

# Cufflinks wrapper on plotly
import cufflinks as cf

# Options for pandas
#pd.options.display.max_columns = 30

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot, init_notebook_mode, plot
cf.go_offline()

init_notebook_mode(connected=True)

# Set global theme
cf.set_config_file(world_readable=True, theme='pearl')

import warnings  
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')
# Import the companies dataset
# Make a list of the missing value types
missing_values = [' -   ']
dataset = pd.read_csv('../input/crunchbase-monthly-export/Companies.csv',na_values=missing_values,thousands=',')
dataset.head(10)
dataset.info()
dataset.isnull().sum()
# Keep a copy of the origional dataset
dataset_orig = dataset.copy()

# Drop unwanted features
features_dropped = ['permalink','homepage_url','category_list'] 
dataset.drop(features_dropped,axis=1,inplace=True)
dataset.describe()
dataset = dataset.rename(columns = {' funding_total_usd ':'funding_total_USD'})
import seaborn as sns; sns.set(style='white')
sns.palplot(sns.color_palette("Blues"))

%matplotlib inline
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
# Variables that define subsets of the data, which will be drawn on separate facets in the grid.
sns.lineplot(x=dataset['founded_year'],y=dataset['funding_total_USD'],hue='status',data=dataset)
# Declutter graph
sns.despine(offset=0, trim=False)
ax.spines['left'].set_color('lightgray')
ax.spines['bottom'].set_color('lightgray')
# Title and Labels
plt.title('Total funding in USD per year',fontsize=14)
plt.xlabel('Year founded',fontsize=12)
plt.ylabel('Funding total (USD)',fontsize=12)
plot_data = dataset[dataset['founded_year'] > 1980]
plot_data = plot_data.sort_values('founded_year',ascending=True)

%matplotlib inline
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
sns.countplot(y=plot_data['founded_year'].astype(int),color='lightgray')
#Get rid of top and right border:
sns.despine(offset=0, trim=False)
# Change the colors of the left and bottom borders (fade into the background)
ax.spines['left'].set_color('lightgray')
ax.spines['bottom'].set_color('lightgray')
# Attract attention
ax.patches[30].set_fc('cornflowerblue')
ax.patches[31].set_fc('cornflowerblue')
# Annotate the figure
#ax.text(4800, 6, 'Most startups were created in the years \textcolor{blue}{2011 and 2012}', ha='right', va='top', fontsize=22)
plt.annotate('Most startups were created\nin the years 2011 and 2012.',
             xy=(3300,4),xytext=(3300,4),color='cornflowerblue',fontsize=14)

#ncount = len(dataset['founded_year'])
#for p in ax.patches:
#    x=p.get_bbox().get_points()[:,0]
#    y=p.get_bbox().get_points()[1,1]
#    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
#            ha='center', va='bottom') # set the alignment of the text

plt.title('Startup frequency per year',fontsize=14)
plt.xlabel('Number of Startups',fontsize=12)
plt.ylabel('Year founded',fontsize=12)
fig.text(
        0.05, 0.05,
        'Data taken from: https://www.crunchbase.com/',
        ha='left')
%matplotlib inline
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
sns.countplot(x=dataset['funding_rounds'],color='lightgray')
#Get rid of top and right border:
sns.despine(offset=0, trim=False)
# Change the colors of the left and bottom borders (fade into the background)
ax.spines['left'].set_color('lightgray')
ax.spines['bottom'].set_color('lightgray')
# Attract attention
ax.patches[0].set_fc('cornflowerblue')

# Add percentages
ncount = len(dataset['funding_rounds'])
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom',color='black') # set the alignment of the text

plt.title('Funding rounds',fontsize=14)
plt.xlabel('Funding rounds',fontsize=12)
plt.ylabel('Counts',fontsize=12)
fig.text(
        0.05, 0.05,
        'Data taken from: https://www.crunchbase.com/',
        ha='left')
# Go more indepth into the the case of funding_rounds=1
one_round = dataset[dataset['funding_rounds']==1] 
rounds_plot = one_round['founded_year'].value_counts()
%matplotlib inline
rounds_plot.iplot(kind='bar',title='Distribution of startups funded only once',
                      xTitle='Year founded',yTitle='Number of startups',color='lightgray')
plot_data = dataset['country_code'].value_counts()
%matplotlib inline
plot_data[0:20].iplot(kind='bar',title='Number of startups across different countries',
                      xTitle='Countries',yTitle='Number of Startups',color='gray')
%matplotlib inline
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
sns.countplot(dataset['status'],color='lightgray')
#Get rid of top and right border:
sns.despine(offset=0, trim=False)
# Change the colors of the left and bottom borders (fade into the background)
ax.spines['left'].set_color('lightgray')
ax.spines['bottom'].set_color('lightgray')
# Attract attention
ax.patches[1].set_fc('cornflowerblue')

ncount = len(dataset['status'])
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom',color='black') # set the alignment of the text

plt.title('Operational status of startups',fontsize=14)
plt.xlabel('Operational Status',fontsize=12)
plt.ylabel('Number of startups',fontsize=12)
fig.text(
        0.05, 0.05,
        'Data taken from: https://www.crunchbase.com/',
        ha='left',fontsize=10)
dataset.rename(columns={dataset.columns[1]:'Market'})
plot_m = dataset.iloc[:,1].value_counts()
%matplotlib inline
fig = plt.figure(figsize=(10,8))
plot_m[0:8].iplot(kind='bar',title='Markets with the most startups',
                      xTitle='Market',yTitle='Number of Startups',color='lightgray')
