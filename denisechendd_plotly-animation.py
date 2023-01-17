!pip install chart_studio
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# merge train with item data
def file_loc(path):
    filename = '/kaggle/input/competitive-data-science-predict-future-sales/'
    file_dir = os.path.join(filename,path)
    return file_dir

train_file = file_loc('sales_train.csv')
item_file = file_loc('items.csv')
shops_file = file_loc('shops.csv')
item_cate_file = file_loc('item_categories.csv')
test = file_loc('test.csv')

df_train = pd.read_csv(train_file)
df_item = pd.read_csv(item_file)
df_shops = pd.read_csv(shops_file)
df_item_cate = pd.read_csv(item_cate_file)
df_test = pd.read_csv(test)

print('training data shape: ', df_train.shape)
print(df_train.head())
## merge item cate into training data
df_train_pro = pd.merge(df_train, df_item, how='left', on=['item_id'])
df_train_pro.head()
# no null value

# process into datetime
df_train_pro['date'] = pd.to_datetime(df_train_pro['date'], format='%d.%m.%Y')

df_train_pro.head()

# date: '2013-01-01' to '2015-10-31'
# groupby df on a monthly basis and 
df_item_top10 = df_train_pro.groupby(['item_id'])['item_cnt_day'].sum().reset_index()
df_item_top10.sort_values(by=['item_cnt_day'], ascending=False, inplace=True)
df_item_top10.iloc[:10, :]
import chart_studio
import chart_studio.plotly as py
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
# https://plotly.com/python/animations/#moving-frenet-frame-along-a-planar-curve
# set credentials
username = 'sos7113' # your username
api_key = 'Mnv2ID1QwNClW29CAv2T' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

df_item_mon_cnt_sum = df_train_pro.groupby(['item_id', pd.Grouper(key='date', freq='M')])['item_cnt_day'].sum().reset_index()
df_item_mon_cnt_sum['year'] = df_item_mon_cnt_sum['date'].dt.year
df_item_mon_cnt_sum['month'] = df_item_mon_cnt_sum['date'].dt.month
df_20949 = df_item_mon_cnt_sum.loc[df_item_mon_cnt_sum['item_id']==20949]

## Year string to numeric
years = ["2013", "2014", "2015"]

# make list of top5 items sold
items = [20949, 2808, 3732, 17717, 5822]
# color matched with each item line
colors = ['firebrick', 'dodgerblue', 'crimson', 'dimgrey', 'forestgreen']
        
# make figure
fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}

# fill in most of layout
fig_dict["layout"]["xaxis"] = {"range": [0, 13], "title": "Month"}
fig_dict["layout"]["yaxis"] = {"title": "Total Items Sold"}
fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["title"] = "Top 5 items sold from 2013 and 2015"
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 300,
                                                                    "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Year:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}

# colored continent, sized population
# make data
year = 2013
#TWO
for item, k in zip(items, colors):
    dataset_by_year = df_item_mon_cnt_sum[df_item_mon_cnt_sum["year"] == year]
    dataset_by_year_item = dataset_by_year[dataset_by_year["item_id"] == item]

    data_dict = {
        "x": list(dataset_by_year_item["month"]),
        "y": list(dataset_by_year_item["item_cnt_day"]),
        'name' : 'item_{}'.format(item),
        "mode": "lines+markers",
        'marker': dict(
                color= k,
                size=12,
                line=dict(
                    color='Black',
                    width=2
                )
            ),
        }
    fig_dict["data"].append(data_dict)

# make frames
for year in years:
    frame = {"data": [], "name": str(year)}
    for item, k in zip(items, colors):
        dataset_by_year = df_item_mon_cnt_sum[df_item_mon_cnt_sum["year"] == int(year)]
        dataset_by_year_item = dataset_by_year[dataset_by_year["item_id"] == item]

        data_dict = {
            "x": list(dataset_by_year_item["month"]),
            "y": list(dataset_by_year_item["item_cnt_day"]),
            'name' : 'item_{}'.format(item),
            "mode": "lines+markers",
            'marker': dict(
                color=k,
                size=12,
                line=dict(
                    color='Black',
                    width=2
                )
            ),

        }
        frame["data"].append(data_dict)
    
    

    fig_dict["frames"].append(frame)
    
    slider_step = {"args": [
        [year],
        {"frame": {"duration": 300, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 300}}
    ],
        "label": year,
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)


fig_dict["layout"]["sliders"] = [sliders_dict]

fig = go.Figure(fig_dict)

# fig.show()

iplot(fig)





