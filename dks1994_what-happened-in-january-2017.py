# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train_df = load_df()
test_df = load_df("../input/test.csv")
date_browser = train_df[['date','device.browser']]
import datetime
date_browser['date'] = date_browser['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))


def custom_func(series):
    Chrome = 0
    Firefox = 0
    IE = 0
    custom_list = series.tolist()
    for x in custom_list:
        if x == 'Chrome':
            Chrome = Chrome + 1
        if x == 'Firefox':
            Firefox = Firefox + 1
        if x == 'Internet Explorer':
            IE = IE + 1
    return Chrome,Firefox,IE


grpd = date_browser.groupby('date')['device.browser'].agg([custom_func])

chrome = []
firefox =[]
ie = []

for x in range(len(grpd)):
    chrome.append(grpd['custom_func'][x][0])
    firefox.append(grpd['custom_func'][x][1])
    ie.append(grpd['custom_func'][x][2])
    
chrome = np.asarray(chrome)
firefox = np.asarray(firefox)
ie = np.asarray(ie)

from sklearn.preprocessing import normalize
chrome = normalize(chrome[:,np.newaxis], axis=0).ravel()
firefox = normalize(firefox[:,np.newaxis], axis=0).ravel()
ie = normalize(ie[:,np.newaxis], axis=0).ravel()

grpd['chrome'] = chrome
grpd['firefox'] = firefox
grpd['ie'] = ie
def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

trace1 = scatter_plot(grpd['chrome'], 'red')
trace2 = scatter_plot(grpd['firefox'], 'blue')
trace3 = scatter_plot(grpd['ie'], 'green')

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Chrome", "Firefox","Internet Explorer"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig['layout'].update(height=800, width=1200, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')
grpd['chrome_rol'] = grpd['chrome'].rolling(10).mean()
grpd['firefox_rol'] = grpd['firefox'].rolling(10).mean()
grpd['ie_rol'] = grpd['ie'].rolling(10).mean()

trace1_rol = scatter_plot(grpd['chrome_rol'], 'red')
trace2_rol = scatter_plot(grpd['firefox_rol'], 'blue')
trace3_rol = scatter_plot(grpd['ie_rol'], 'green')

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Chrome", "Firefox","Internet Explorer"])
fig.append_trace(trace1_rol, 1, 1)
fig.append_trace(trace2_rol, 2, 1)
fig.append_trace(trace3_rol, 3, 1)
fig['layout'].update(height=800, width=1200, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')
date_continent = train_df[['date','geoNetwork.continent']]
import datetime
date_continent['date'] = date_continent['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))


def custom_func(series):
    Americas = 0
    Asia = 0
    Europe = 0
    Oceania = 0
    Africa = 0
    custom_list = series.tolist()
    for x in custom_list:
        if x == 'Americas':
            Americas = Americas + 1
        if x == 'Asia':
            Asia = Asia + 1
        if x == 'Europe':
            Europe = Europe + 1
        if x == 'Oceania':
            Oceania = Oceania + 1
        if x == 'Africa':
            Africa = Africa + 1
    return Americas,Asia,Europe,Oceania,Africa


cont_grpd = date_continent.groupby('date')['geoNetwork.continent'].agg([custom_func])

americas = []
asia =[]
europe = []
oceania = []
africa = []

for x in range(len(cont_grpd)):
    americas.append(cont_grpd['custom_func'][x][0])
    asia.append(cont_grpd['custom_func'][x][1])
    europe.append(cont_grpd['custom_func'][x][2])
    oceania.append(cont_grpd['custom_func'][x][3])
    africa.append(cont_grpd['custom_func'][x][4])
    
americas = np.asarray(americas)
asia = np.asarray(asia)
europe = np.asarray(europe)
oceania = np.asarray(oceania)
africa = np.asarray(africa)

'''from sklearn.preprocessing import normalize
chrome = normalize(chrome[:,np.newaxis], axis=0).ravel()
firefox = normalize(firefox[:,np.newaxis], axis=0).ravel()
ie = normalize(ie[:,np.newaxis], axis=0).ravel()'''

grpd['americas'] = americas
grpd['asia'] = asia
grpd['europe'] = europe
grpd['oceania'] = oceania
grpd['africa'] = africa

trace1 = scatter_plot(grpd['americas'], 'red')
trace2 = scatter_plot(grpd['asia'], 'blue')
trace3 = scatter_plot(grpd['europe'], 'green')
trace4 = scatter_plot(grpd['oceania'], 'black')
trace5 = scatter_plot(grpd['africa'], 'pink')

fig = tools.make_subplots(rows=5, cols=1, vertical_spacing=0.08,
                          subplot_titles=["americas", "asia","europe","oceania","africa"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)
fig.append_trace(trace5, 5, 1)
fig['layout'].update(height=800, width=1200, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')
grpd['americas_rol'] = grpd['americas'].rolling(10).mean()
grpd['asia_rol'] = grpd['asia'].rolling(10).mean()
grpd['europe_rol'] = grpd['europe'].rolling(10).mean()
grpd['oceania_rol'] = grpd['oceania'].rolling(10).mean()
grpd['africa_rol'] = grpd['africa'].rolling(10).mean()

trace1_rol = scatter_plot(grpd['americas_rol'], 'red')
trace2_rol = scatter_plot(grpd['asia_rol'], 'blue')
trace3_rol = scatter_plot(grpd['europe_rol'], 'green')
trace4_rol = scatter_plot(grpd['oceania_rol'], 'black')
trace5_rol = scatter_plot(grpd['africa_rol'], 'pink')


fig = tools.make_subplots(rows=5, cols=1, vertical_spacing=0.08,
                          subplot_titles=["americas", "asia","europe","oceania","africa"])
fig.append_trace(trace1_rol, 1, 1)
fig.append_trace(trace2_rol, 2, 1)
fig.append_trace(trace3_rol, 3, 1)
fig.append_trace(trace4_rol, 4, 1)
fig.append_trace(trace5_rol, 5, 1)
fig['layout'].update(height=800, width=1200, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')
date_device = train_df[['date','device.deviceCategory']]
import datetime
date_device['date'] = date_device['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))


def custom_func(series):
    desktop = 0
    mobile = 0
    tablet = 0
    custom_list = series.tolist()
    for x in custom_list:
        if x == 'desktop':
            desktop = desktop + 1
        if x == 'mobile':
            mobile = mobile + 1
        if x == 'tablet':
            tablet = tablet + 1
    return desktop,mobile,tablet


dev_grpd = date_device.groupby('date')['device.deviceCategory'].agg([custom_func])

desktop = []
mobile = []
tablet = []

for x in range(len(cont_grpd)):
    desktop.append(cont_grpd['custom_func'][x][0])
    mobile.append(cont_grpd['custom_func'][x][1])
    tablet.append(cont_grpd['custom_func'][x][2])
    
desktop = np.asarray(desktop)
mobile = np.asarray(mobile)
tablet = np.asarray(tablet)

'''from sklearn.preprocessing import normalize
chrome = normalize(chrome[:,np.newaxis], axis=0).ravel()
firefox = normalize(firefox[:,np.newaxis], axis=0).ravel()
ie = normalize(ie[:,np.newaxis], axis=0).ravel()'''

grpd['desktop'] = desktop
grpd['mobile'] = mobile
grpd['tablet'] = tablet

trace1 = scatter_plot(grpd['desktop'], 'red')
trace2 = scatter_plot(grpd['mobile'], 'blue')
trace3 = scatter_plot(grpd['tablet'], 'green')

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.08,
                          subplot_titles=["desktop", "mobile","tablet"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig['layout'].update(height=800, width=1200, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')
grpd['desktop_rol'] = grpd['desktop'].rolling(10).mean()
grpd['mobile_rol'] = grpd['mobile'].rolling(10).mean()
grpd['tablet_rol'] = grpd['tablet'].rolling(10).mean()

trace1_rol = scatter_plot(grpd['desktop_rol'], 'red')
trace2_rol = scatter_plot(grpd['mobile_rol'], 'blue')
trace3_rol = scatter_plot(grpd['tablet_rol'], 'green')


fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.08,
                          subplot_titles=["desktop", "mobile","tablet"])
fig.append_trace(trace1_rol, 1, 1)
fig.append_trace(trace2_rol, 2, 1)
fig.append_trace(trace3_rol, 3, 1)
fig['layout'].update(height=800, width=1200, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')