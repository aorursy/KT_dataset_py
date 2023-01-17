# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns
import plotly
import plotly.offline as pyoff
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
import squarify # for tree maps
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
init_notebook_mode(connected = True)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.columns
train.shape
test.shape
train.isnull().sum()
train = train.dropna()
test.isnull().sum()
def extractColTypes(dataset):
    """This functions extracts numeric, categorical , datetime and boolean column types.
    Returns 4 lists with respective column types"""
    num_cols_list = [i for i in dataset.columns if dataset[i].dtype in ['int64','float64']]
    cat_cols_list = [i for i in dataset.columns if dataset[i].dtype in ['object']]
    date_cols_list = [i for i in dataset.columns if dataset[i].dtype in ['datetime64[ns]']]
    bool_cols_list = [i for i in dataset.columns if dataset[i].dtype in ['bool']]
    print ("Numeric Columns:", len(num_cols_list))
    print ("Categorical/Character Columns:", len(cat_cols_list))
    print ("Date Columns:",len(date_cols_list))
    print ("Boolean Columns:",len(bool_cols_list))
    return(num_cols_list,cat_cols_list,date_cols_list,bool_cols_list)
num_cols_list,cat_cols_list,date_cols_list,bool_cols_list = extractColTypes(train)
def generateLayoutBar(col_name):
    layout_bar = go.Layout(
        autosize=False, # auto size the graph? use False if you are specifying the height and width
        width=800, # height of the figure in pixels
        height=600, # height of the figure in pixels
        title = "Distribution of {} column".format(col_name), # title of the figure
        # more granular control on the title font 
        titlefont=dict( 
            family='Courier New, monospace', # font family
            size=14, # size of the font
            color='black' # color of the font
        ),
        # granular control on the axes objects 
        xaxis=dict( 
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the x axis
            color='black'  # color of the font
            )
        ),
        yaxis=dict(
#         range=[0,100],
            title='Percentage',
            titlefont=dict(
                size=14,
                color='black'
            ),
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the y axis
            color='black' # color of the font
            )
        ),
        font = dict(
            family='Courier New, monospace', # font family
            color = "white",# color of the font
            size = 12 # size of the font displayed on the bar
                )  
        )
    return layout_bar
def plotBar(dataframe_name, col_name):
    """
    Plot a bar chart for the categorical columns

    Arguments:
    dataframe name
    categorical column name

    Output:
    Plot
    """
    # create a table with value counts
    temp = dataframe_name[col_name].value_counts()
    # creating a Bar chart object of plotly
    data = [go.Bar(
            x=temp.index.astype(str),  # x axis values
            y=np.round(temp.values.astype(float) / temp.values.sum(), 4) * 100,  # y axis values
            text=['{}%'.format(i) for i in np.round(temp.values.astype(float) / temp.values.sum(), 4) * 100],
            # text to be displayed on the bar, we are doing this to display the '%' symbol along with the number on the bar
            textposition='auto',  # specify at which position on the bar the text should appear
            marker=dict(color='#0047AB'),)]  # change color of the bar
    # color used here Cobalt Blue

    layout_bar = generateLayoutBar(col_name=col_name)

    fig = go.Figure(data=data, layout=layout_bar)
    return iplot(fig)

for i in cat_cols_list[3:]:
    print ("Train Distribution")
    plotBar(train, i)
    print ("Test Distribution")
    plotBar(test, i)
# Compute the correlation matrix
corr = train.corr()
# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,
#             mask=mask,
            cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5,annot=True)

ax.set_title('Correlation Matrix', size=20)
plt.show()
temp = train.groupby(['matchType']).agg({'matchDuration':np.mean})
data = [go.Bar(
            x=temp.index.astype(str),  # x axis values
            y=temp.values,  # y axis values
            text=['{}%'.format(i) for i in temp.values],
            # text to be displayed on the bar, we are doing this to display the '%' symbol along with the number on the bar
            textposition='auto',  # specify at which position on the bar the text should appear
            marker=dict(color='#0047AB'),)]  # change color of the bar
    # color used here Cobalt Blue

layout_bar = generateLayoutBar(col_name='matchDuration')

fig = go.Figure(data=data, layout=layout_bar)
iplot(fig)
data = []
for i in train.matchType.unique():
    trace = go.Box(y = train.matchDuration[train.matchType==i])
    data.append(trace)
iplot(data)
train.groupId.value_counts().index[0]
player_kills_df = train.groupby(['Id']).agg({'kills': np.sum})
player_kills_df.sort_values(['kills'],ascending= False).head(1)
player_revives_df = train.groupby(['Id']).agg({'revives': np.sum})
player_revives_df.sort_values(['revives'],ascending= False).head(1)
# these line of code will get the max,min,mean, min for all the numeric columns
max_num_dict = {'{}_max'.format(i):np.max(train[i]) for i in num_cols_list}
min_num_dict = {'{}_min'.format(i):np.min(train[i]) for i in num_cols_list}
mean_num_dict = {'{}_mean'.format(i):np.mean(train[i]) for i in num_cols_list}
median_num_dict = {'{}_median'.format(i):np.median(train[i]) for i in num_cols_list}
print(train.Id[train.rideDistance == max_num_dict['rideDistance_max']])
print(max_num_dict['rideDistance_max'])
train[train.groupId == train.groupId[train.rideDistance == max_num_dict['rideDistance_max']].values[0]]
print(train.Id[train.swimDistance == max_num_dict['swimDistance_max']])
print(max_num_dict['swimDistance_max'])
train[train.groupId == train.groupId[train.swimDistance == max_num_dict['swimDistance_max']].values[0]]
group_mem_df = train.groupby(['groupId','matchId']).agg({'Id': len})
group_mem_df[group_mem_df['Id']>4].shape
dist_columns = [i for i in train.columns if 'Dist' in i]
dist_columns
train[dist_columns].head(10).sum(axis = 1)
train['totalDistance'] = train[dist_columns].sum(axis = 1)
test['totalDistance'] = test[dist_columns].sum(axis = 1)
train[train.totalDistance == np.max(train.totalDistance)]
