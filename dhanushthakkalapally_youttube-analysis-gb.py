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
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import iplot,init_notebook_mode
init_notebook_mode()
import plotly.graph_objs as go
import plotly.express as px
from plotly import tools
vgb_df = pd.read_csv('/kaggle/input/youtube/GBvideos.csv',encoding  ='utf8',error_bad_lines = False)
vgb_df.head()
cgb_df = pd.read_csv('../input/youtube/GBcomments.csv',encoding='utf8',error_bad_lines = False)
cgb_df.head()
vgb_df.describe()
vgb_cols = vgb_df.columns
print(vgb_cols)
vgb_df.info()
for i in vgb_cols:
    if (vgb_df[i].isna().sum()>0):
        print(i)
    else:
        print('No nulls in '+i)
cgb_df.info()
cols = cgb_df.columns
for i in cols:
    if(cgb_df[i].isna().sum()>0):
        print(i+': '+ str(cgb_df[i].isna().sum() ))
    else: 
        print('No null in '+i)
cgb_df.describe()
cgb_df.loc[cgb_df['comment_text'].isna()].video_id.unique().shape
# Drops rows in the dataframe 
cgb_df = cgb_df.dropna()
cols = cgb_df.columns
for i in cols:
    if(cgb_df[i].isna().sum()>0):
        print(i+': '+ str(cgb_df[i].isna().sum() ))
    else: 
        print('No null in '+i)
vgb_df.head()
import datetime
def formatDate(date):
    day,month = list(map(lambda x:int(x),"{:.2f}".format(date).split('.')))
    x=datetime.datetime(2017,month,day)
    return x.strftime("%Y-%b-%d")
vgb_df['date']
vgb_df['date']=vgb_df['date'].apply(formatDate)
vgb_df.head()
def Bar_plot(x_data,y_data,title,x_title,y_title,color='rgb(55, 83, 109)',hovertext=None):
    trace = go.Bar(x=x_data,y=y_data,marker_color=color,hovertext =hovertext)
    layout = {'title':title,'xaxis':{'title':x_title},'yaxis':{'title':y_title},'template':'plotly_dark'}
    data = [trace]
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)
    
x=vgb_df['date'].unique()
y=list(vgb_df['date'].value_counts()[vgb_df['date'].unique()])
trace = go.Bar(x=x,y=y,marker_color='#fe9500')
layout = {'title':'Number of Videos per day','xaxis':{'title':'dates'},'yaxis':{'title':'Number of videos per day'},'template':'plotly_dark'}
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
x=vgb_df.groupby(['video_id'])[['video_id']].count()
x.shape
min = x.min()['video_id']
max = x.max()['video_id']
x_data = []
y_data = []
hover_data = []
for j in range(min,max+1):
    count = 0
    x_data.append(j)
    for i in x.iterrows():
        if(i[1]['video_id']>=j):
               count = count + 1
    y_data.append(count)
    hover_data.append(str(count/x.shape[0]*100)+" % of videos")
Bar_plot(x_data,y_data,'Number of videos present Based on different Thersholds','Thershold','Number of Videos',hovertext = hover_data,color='#fe9500')
vgb_df.head()
print("There are "+str(vgb_df['channel_title'].unique().size)+" from where the 1736 videos are produced we assume that from one channel there are multiple videos are produced")
x= vgb_df.groupby(['channel_title']).size()
max = x.max()
min = x.min()
data = []
for j in range(min,max+1):
    count = 0 
    for i in x.axes[0]:
        if (x[i]==j):
            data.append(j)
counts, bins = np.histogram(data, bins=range(min,45, 5))
bins = 0.5 * (bins[:-1] + bins[1:])
trace = go.Bar(x=bins,y=counts ,marker_color = '#fe9500')
data = [trace]
layout = {'title':'Number of days a unique channel is in trending status','xaxis':{'title':'Days'},'yaxis':{'title':"count"},'template':'plotly_dark'}    
iplot({'data':data,'layout':layout})
channel_titles = {i:0 for i in vgb_df['channel_title'].unique()}
series =vgb_df.groupby(['channel_title','video_id'])['video_id'].count()
for i in series.axes[0]:
    channel_titles[i[0]] = channel_titles[i[0]]+1 
values = []
channels = []
for i in channel_titles.keys():
    value = channel_titles[i]
    if (value > 1):
            channels.append(i)
            values.append(value)
values_ = []
for i in set(values):
    count = 0 
    for j in values:
        if (i==j):
            count = count + 1 
    values_.append(count)  
trace = go.Bar(x=values_,y=list(set(values)),orientation='h',marker_color ='#fe9500')
data = [trace]
layout = ({'template':'plotly_dark','title':'Number of Channels Uploaded Multiple Trending Videos','xaxis':{'title':'Number of channels'},'yaxis':{'title':'Number of videos'}})
iplot({'data':data,'layout':layout})
vgb_df.head(2)
df = pd.read_json('../input/youtube/GB_category_id.json')
categories = {}
for i in range(df.shape[0]):
    categories[df.iloc[i]['items']['id']] = df.iloc[i]['items']['snippet']['title']
categories['29'] = 'unknown'
vgb_df['category'] = [categories[str(i)] for i in vgb_df['category_id']]
vgb_df.head()
categories_ = vgb_df['category'].value_counts()
norm_cat = [i*100 for i in list(vgb_df['category'].value_counts(normalize = True))]
cat_names = categories_.axes[0]
values = list(categories_)
trace = go.Bar(y=cat_names,x=values,marker_color='#fe9500',orientation='h',hovertext = norm_cat)
data = [trace]
layout = {'title':'Trending Videos Categories','xaxis':{'title':'Number of videos belongs to respective categories are in trending status'},'yaxis':{'title':'Categories'},'template':'plotly_dark'}
iplot({'data':data,'layout':layout})
from collections import Counter
freq = Counter(list(vgb_df.groupby(['category','video_id'])['video_id'].count().axes[0].to_frame()['category']))
categories=freq.keys()
values = freq.values()
hover_text = (np.array(list(values))/(np.array(list(values)).sum()))*100
trace = go.Bar(x=list(categories),y=list(values),marker_color='#fe9500',hovertext = hover_text)
data = [trace]
layout = {'title':'Trending Videos Categories','xaxis':{'title':'Categories','categoryorder':'total descending'},'yaxis':{'title':'Count'},'template':'plotly_dark'}
iplot({'data':data,'layout':layout})
dates = vgb_df['date'].unique()
groups  = vgb_df.groupby(['date','category']).size()
categories =  vgb_df['category'].unique()
dict = {}
dict2 = {}
for i in categories:
    dict[i] = []
    for j in dates:
        try:
            dict[i].append(groups[j][i])
        except:
            dict[i].append(0)
fig = go.Figure()
for i in dict:
#     print("on an average "+ str(np.array(dict[i]).mean()) + " this "+ i + " category")
    dict2[i]=np.array(dict[i]).mean()
    fig.add_trace(go.Bar(x=dates , y=dict[i],name = i))
fig.update_layout(barmode='stack',template = 'plotly_dark',title = "Every Trending Videos Stats Based on Categories")
fig.show()
import plotly.graph_objects as go
labels = list(dict2.keys())
values =list(dict2.values())
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()
vgb_df.head(2)
vgb_df['channel_title'].unique().shape
from collections import Counter
data = Counter(vgb_df['channel_title']) 
data1 = {i : data[i] for i in data if data[i]>1 }
df = pd.DataFrame({'Channel':list(data1.keys()),'Counts':list(data1.values())})
fig = px.sunburst(df.loc[df['Counts']>10],path =['Counts','Channel'],width = 750, height = 750,title='Channels With videos in trending status for >10 days', color_discrete_sequence=px.colors.qualitative.T10,
)
fig.show()
details_df = pd.read_csv('../input/gbvideo-details/GBVideo_details.csv')
details_df.head()
fig = px.sunburst(df.loc[df['Counts']>=30],path =['Counts','Channel'],width = 750, height = 750,title='Top Channels In Trending status for >=30 days ', color_discrete_sequence=px.colors.qualitative.T10,
)
fig.show()
data = {i:vgb_df.groupby('channel_title').get_group(i).groupby('title').size().count() for i in list(df.loc[df['Counts']>=30]['Channel'])}
text = [list(df['Counts'][df['Channel']==i]) for i in data.keys()]
data = [go.Bar(x=list(data.keys()),y=list(data.values()),marker_color = 'orange',hovertext =np.array(text).reshape(len(text)))]
layout = {'template':'plotly_dark','xaxis':{'title':'Channel_Names','categoryorder':'total descending'},'yaxis':{'title':'Number of Videos'}}
iplot({'data':data,'layout':layout})
data = {i:vgb_df.groupby('channel_title').get_group(i).groupby('title').size().count() for i in list(df.loc[df['Counts']>=30]['Channel'])}
num_of_videos = {i:vgb_df.groupby('channel_title').get_group(i).groupby('title').size().count() for i in list(df.loc[df['Counts']>=30]['Channel'])}
num_of_days= np.array([list(df['Counts'][df['Channel']==i]) for i in data.keys()]).reshape(25)
famous_channels = pd.DataFrame({'channel_name':list(num_of_videos.keys()),'num_videos':list(num_of_videos.values()),'num_days':num_of_days})
dict = {}
for i in famous_channels.iterrows():
    dict[i[1][0]]=i[1][2]/i[1][1]
famous_channels['rank'] =[dict[i] for i in famous_channels['channel_name']]
top_videos=famous_channels.sort_values('rank',ascending=False).reset_index()
# df.drop(columns=['B', 'C'])

top_videos.drop(columns=['index'])
channels_categories={i:vgb_df[['channel_title','category']].groupby('channel_title').get_group(i)['category'].unique() for i in famous_channels['channel_name']}
famous_channels['categories'] =[channels_categories[i] for i in famous_channels['channel_name']]
famous_channels.head(10)
dict = { i:0 for i in list(vgb_df['category'].unique())}
for i in list(famous_channels['categories']):
    for j in i:
        dict[j]=dict[j]+1
iplot([go.Bar(x=list(dict.keys()),y=list(dict.values()))])
vgb_df.head()
details_df.head()
# For our intial analysis we don't need time so we can easily strip off the time part of the date so that it will be easy to compare two dates

import datetime
from dateutil.parser import parse
details_df['published_date']=pd.to_datetime(details_df['published_date']).dt.date
vgb_df['date']=pd.to_datetime(vgb_df['date']).dt.date
(vgb_df['date']-details_df['published_date'])[1001]
details_df['published_date'][1001]
vgb_df['video_id'][1001]
details_df['video_id'][1001]
vgb_df_ =pd.merge(vgb_df,details_df,on='video_id')
vgb_df_.drop('Unnamed: 0',inplace = True,axis = 1)
# Number of unique videos without the published date 
vgb_df_['video_id'][vgb_df_['description'].isna()].unique().shape
vgb_df_['video_id'][vgb_df_['description'].isna()].value_counts()
videos = vgb_df['video_id'].unique()
videos
details_df.dropna(inplace = True)
groups = vgb_df.groupby('video_id')
dates = {}
for i in videos:
    if (details_df['published_date'][details_df['video_id']==i].any()):
        dates[i]=(groups.get_group(i)['date'].sort_values().unique()[0]-details_df['published_date'][details_df['video_id']==i])
details_df.head()
days = []
for i in list(dates.values()):
    days.append(list(i)[0].days)
fig = go.Figure()
fig.add_trace(go.Box(y=days))
fig.show()
vgb_df.head()
groups = vgb_df.groupby('category')['likes']
traces = []
for i in vgb_df['category'].unique():
    traces.append(go.Box(y=groups.get_group(i),name = i))
    
iplot({'data':traces,'layout':{'title':'Likes in each category for Trending Videos','template':'plotly_dark'}})
vgb_df.head(10)
categories = { i:[] for i in vgb_df['category'].unique()}
traces = []
vids = vgb_df['video_id'].unique()
likes = {}
count = 0 
for vid in vids:
      likes[vid] = vgb_df[['likes','date']][vgb_df['video_id']==vid].sort_values('date').iloc[0]['likes']

for i in vids:
        categories[vgb_df['category'][vgb_df['video_id']==i].iloc[0]].append(likes[i])
        

for i in categories:
    traces.append(go.Box(y=categories[i],name = i))
iplot({'data':traces,'layout':{'title':'Likes in each category for  videos when first came to trending status','template':'plotly_dark'}})
groups  = vgb_df.groupby('category')['likes','date','video_id']
sdf= groups.get_group('Entertainment').groupby('video_id')['likes','date']
data = []
for i in groups.get_group('Entertainment')['video_id'].unique():
    data.append(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']))
iplot({'data':data,'layout':{'template':'plotly_dark','title':'Entertainment'}})
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=( "Music", "Comedy", "How to Style",'People and Vlogs','Films and animation','sports','Education','Gaming','News and Politics'))
sdf = groups.get_group('Music').groupby('video_id')['likes','date']
for i in groups.get_group('Music')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=1,col=1)
    count = count +1
sdf = groups.get_group('Comedy').groupby('video_id')['likes','date']
for i in groups.get_group('Comedy')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=1,col=2)

sdf = groups.get_group('Howto & Style').groupby('video_id')['likes','date']
for i in groups.get_group('Howto & Style')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=1,col=3)


sdf = groups.get_group('People & Blogs').groupby('video_id')['likes','date']
for i in groups.get_group('People & Blogs')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=2,col=1)
    

sdf = groups.get_group('Film & Animation').groupby('video_id')['likes','date']
for i in groups.get_group('Film & Animation')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=2,col=2)


sdf = groups.get_group('Sports').groupby('video_id')['likes','date']
for i in groups.get_group('Sports')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=2,col=3)
    
sdf = groups.get_group('Education').groupby('video_id')['likes','date']
for i in groups.get_group('Education')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=3,col=1)
    

sdf = groups.get_group('Gaming').groupby('video_id')['likes','date']
for i in groups.get_group('Gaming')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=3,col=2)

sdf = groups.get_group('News & Politics').groupby('video_id')['likes','date']
for i in groups.get_group('News & Politics')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['likes']),row=3,col=3)

fig.update_layout({'template':'plotly_dark','height':1000,'width':2000,'title':'Likes of videos in different Categories over time'})    
fig.show()

categories = { i:[] for i in vgb_df['category'].unique()}
vids = vgb_df['video_id'].unique()
views = {}
count = 0 
traces = []
for vid in vids:
      views[vid] =np.log(vgb_df[['views','date']][vgb_df['video_id']==vid].sort_values('date').iloc[0]['views']+1)

for i in vids:
        categories[vgb_df['category'][vgb_df['video_id']==i].iloc[0]].append(views[i])
        

for i in categories:
    traces.append(go.Box(y=categories[i],name = i))
iplot({'data':traces,'layout':{'title':'Views in each category for videos when first came to trending status (LOG(Views))','xaxis':{'title':'Categories'},'yaxis':{'title':'log(Number of Views)'},'template':'plotly_dark'}})
categories = { i:[] for i in vgb_df['category'].unique()}
vids = vgb_df['video_id'].unique()
views = {}
count = 0 
traces = []
for vid in vids:
      views[vid] =(vgb_df[['views','date']][vgb_df['video_id']==vid].sort_values('date').iloc[0]['views'])

for i in vids:
        categories[vgb_df['category'][vgb_df['video_id']==i].iloc[0]].append(views[i])
        

for i in categories:
    traces.append(go.Box(y=categories[i],name = i))
iplot({'data':traces,'layout':{'title':'Views in each category for videos when first came to trending status ','xaxis':{'title':'Categories'},'yaxis':{'title':'Number of Views'},'template':'plotly_dark'}})
categories = { i:[] for i in vgb_df['category'].unique()}
vids = vgb_df['video_id'].unique()
views = {}
count = 0 
traces = []
for vid in vids:
      views[vid] =np.log((vgb_df[['views','date']][vgb_df['video_id']==vid].sort_values('date',ascending = False).iloc[0]['views'])+1)

for i in vids:
        categories[vgb_df['category'][vgb_df['video_id']==i].iloc[0]].append(views[i])
        

for i in categories:
    traces.append(go.Box(y=categories[i],name = i))
iplot({'data':traces,'layout':{'title':'Videos Views in Each category when every video last seen in trending status Log(Views) ','xaxis':{'title':'Categories'},'yaxis':{'title':'Number of Views'},'template':'plotly_dark'}})
groups  = vgb_df.groupby('category')['views','date','video_id']
sdf= groups.get_group('Entertainment').groupby('video_id')['views','date']
data = []
for i in groups.get_group('Entertainment')['video_id'].unique():
    data.append(go.Scatter(x=sdf.get_group(i)['date'],y=np.log(sdf.get_group(i)['views'])))
iplot({'data':data,'layout':{'template':'plotly_dark','title':'Entertainment','xaxis':{'title':"Treding Dates"},'yaxis':{'title':'Log(Views)'}}})
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=( "Music", "Comedy", "How to Style",'People and Vlogs','Films and animation','sports','Education','Gaming','News and Politics'))
sdf = groups.get_group('Music').groupby('video_id')['views','date']
for i in groups.get_group('Music')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=1,col=1)
    count = count +1
sdf = groups.get_group('Comedy').groupby('video_id')['views','date']
for i in groups.get_group('Comedy')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=1,col=2)

sdf = groups.get_group('Howto & Style').groupby('video_id')['views','date']
for i in groups.get_group('Howto & Style')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=1,col=3)


sdf = groups.get_group('People & Blogs').groupby('video_id')['views','date']
for i in groups.get_group('People & Blogs')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=2,col=1)
    

sdf = groups.get_group('Film & Animation').groupby('video_id')['views','date']
for i in groups.get_group('Film & Animation')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=2,col=2)


sdf = groups.get_group('Sports').groupby('video_id')['views','date']
for i in groups.get_group('Sports')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=2,col=3)
    
sdf = groups.get_group('Education').groupby('video_id')['views','date']
for i in groups.get_group('Education')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=3,col=1)
    

sdf = groups.get_group('Gaming').groupby('video_id')['views','date']
for i in groups.get_group('Gaming')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=3,col=2)

sdf = groups.get_group('News & Politics').groupby('video_id')['views','date']
for i in groups.get_group('News & Politics')['video_id'].unique():
    fig.add_trace(go.Scatter(x=sdf.get_group(i)['date'],y=sdf.get_group(i)['views']),row=3,col=3)

fig.update_layout({'template':'plotly_dark','height':1000,'width':2000,'title':'Views of videos in different Categories over time'})    
fig.show()

categories = { i:[] for i in vgb_df['category'].unique()}
vids = vgb_df['video_id'].unique()
views = {}
count = 0 
traces = []
for vid in vids:
      views[vid] =np.log((vgb_df[['views','date']][vgb_df['video_id']==vid].sort_values('date',ascending = False).iloc[0]['views'])-(vgb_df[['views','date']][vgb_df['video_id']==vid].sort_values('date',ascending = True).iloc[0]['views'])+1)

for i in vids:
        categories[vgb_df['category'][vgb_df['video_id']==i].iloc[0]].append(views[i])
        

for i in categories:
    traces.append(go.Box(y=categories[i],name = i))
iplot({'data':traces,'layout':{'title':'Number of Views videos in each category got while the videos are in trending status','xaxis':{'title':'Categories'},'yaxis':{'title':'LoG(Number of Views)'},'template':'plotly_dark'}})
vgb_df.head()
f, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=True)
cats = vgb_df['category'].unique()
count = 0 
# plt.title('')
for i in range(4):
    for j in range(4):
        if (count<15):
            x=pd.Series(categories[cats[count]], name=cats[count])
            sns.distplot(x,ax = axes[i,j])
        count = count +1
plt.tight_layout()                
plt.show()
vgb_df.head()
categories = { i:[] for i in vgb_df['category'].unique()}
vids = vgb_df['video_id'].unique()
dislikes= {}
count = 0 
traces = []
for vid in vids:
      dislikes[vid] =np.log((vgb_df[['dislikes','date']][vgb_df['video_id']==vid].sort_values('date').iloc[0]['dislikes'])+1)

for i in vids:
        categories[vgb_df['category'][vgb_df['video_id']==i].iloc[0]].append(dislikes[i])
        

for i in categories:
    traces.append(go.Box(y=categories[i],name = i))
iplot({'data':traces,'layout':{'title':'Dislikes in each category for videos when first came to trending status ','xaxis':{'title':'Categories'},'yaxis':{'title':'Number of Dislikes'},'template':'plotly_dark'}})
categories = { i:[] for i in vgb_df['category'].unique()}
vids = vgb_df['video_id'].unique()
dislikes= {}
count = 0 
traces = []
for vid in vids:
      dislikes[vid] =np.log((vgb_df[['dislikes','date']][vgb_df['video_id']==vid].sort_values('date',ascending = False).iloc[0]['dislikes'])+1)

for i in vids:
        categories[vgb_df['category'][vgb_df['video_id']==i].iloc[0]].append(dislikes[i])
        

for i in categories:
    traces.append(go.Box(y=categories[i],name = i))
iplot({'data':traces,'layout':{'title':'Dislikes in each category for videos when Last seen in trending status ','xaxis':{'title':'Categories'},'yaxis':{'title':'Log(Number of Dislikes)'},'template':'plotly_dark'}})
vgb_df.head()
df = vgb_df[['video_id','category','date','views','likes','dislikes','comment_total']]
import time
details_df['duration'].iloc[0]
!pip install duration

from duration import (
    to_iso8601,
    to_seconds,
    to_timedelta,
    to_tuple,
)
!pip install isodate
import isodate
def fun(duration):
    time = isodate.parse_duration(duration)
    return time.total_seconds()
details_df['duration_sec'] = details_df.duration.apply(fun)
details_df.head()
details_df.duration_sec.describe()
df
df_ = details_df[['video_id','published_date','duration_sec']]
new_df = pd.DataFrame(columns = df.columns)
ids = df.video_id.unique()
groups = df.groupby('video_id')
for id in ids:
    new_df=new_df.append(groups.get_group(id).sort_values('date').iloc[0],ignore_index=True)
new_df
grouped_df = pd.merge(new_df,df_,on='video_id',how = 'left')
grouped_df['num_of_days'] = grouped_df.date - grouped_df.published_date
grouped_df.head()
grouped_df.num_of_days = grouped_df.num_of_days.fillna(grouped_df.num_of_days.median())
grouped_df.duration_sec =grouped_df.duration_sec.fillna(grouped_df.duration_sec.mean())
grouped_df.head()
final_df = grouped_df[['category','views','likes','dislikes','comment_total','duration_sec','num_of_days']]
final_df.head()
final_df.info()
final_df['num_of_days']=final_df['num_of_days'].apply(lambda x: x.days)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(np.array(final_df['category']).reshape(-1,1))
encodings = enc.transform(np.array(final_df['category']).reshape(-1,1)).toarray()
from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
scaler.fit(final_df[['views','likes','dislikes','comment_total','duration_sec','num_of_days']])

data = scaler.transform(final_df[['views','likes','dislikes','comment_total','duration_sec','num_of_days']])
scaler.mean_
model_data = np.concatenate((data,encodings),axis = 1)
from sklearn.cluster import KMeans
dict = {}
for i in range(5,30):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data)
    dict[i] = kmeans.inertia_
sns.lineplot(x=list(dict.keys()),y=list(dict.values()))
plt.show()
kmeans = KMeans(n_clusters = 20,n_init=10)
Counter(kmeans.fit_predict(data))
# data.shape
kmeans.inertia_
kmeans.predict(data[8].reshape(1,-1))
final_df['predicts'] = kmeans.predict(data)
final_df.head()
final_df.loc[(final_df['category'] == 'Music') & (final_df['predicts'] == 12)].tail(25)

final_df.groupby('predicts').get_group(12)[['views']].describe()
sns.boxplot(x=list(final_df['comment_total'][final_df['predicts']==0]))
[go.Box(y=list(final_df['views'][final_df['predicts']==24]))]
final_df['views'][final_df['predicts']==12]
from sklearn.manifold import TSNE
model = TSNE(n_components = 2,n_iter=10000,random_state=43)
tsne_data = model.fit_transform(data)
x = []
for vid in grouped_df['video_id']:
    x.append(vgb_df[['views','date']][vgb_df['video_id']==vid].sort_values('date',ascending = False).iloc[0]['date']-(vgb_df[['views','date']][vgb_df['video_id']==vid].sort_values('date',ascending = True).iloc[0]['date']))
x = list(map(lambda x:x.days+1,x))
final_df['trending_days'] = x
del dict
iplot({'data':go.Scatter(x=tsne_data[:,0],y=tsne_data[:,1],mode = 'markers',marker=dict(color = final_df['predicts']),hovertext=final_df['predicts'])})
final_df.groupby('predicts')['duration_sec'].describe().sort_values('count',ascending = False).head(10)
final_df[['views','likes','dislikes','comment_total']]=final_df[['views','likes','dislikes','comment_total']].astype('float')
final_df.groupby('predicts')[['views','trending_days']].describe().head(20)
type(final_df['duration_sec'][0])
final_df.head()
categories = final_df['category'].unique()
groups = final_df.groupby('category')
list = []
for i in categories:
    list.append(go.Box(y=np.log(groups.get_group(i)['duration_sec']+1),name=i))
    
iplot({'data':list,'layout':{'template':'plotly_dark','xaxis':{'title':'categories'},'yaxis':{'title':'LOG2(duration in sec)'}}})
vgb_df.head()
channels_df = pd.read_csv('../input/channel-details/channel_details.csv')
channels_df.head(10
                )
channels_df[['subCount','title','videoCount']].sort_values('subCount').head(25)

iplot({'data':[go.Box(y=channels_df['subCount'])]})
sns.distplot(np.log(channels_df['subCount']+1))
plt.title('Channels Subscribers Distribution')
plt.show()
x = np.log(channels_df['subCount']+1)
kwargs = {'cumulative': True}
sns.distplot(x, hist_kws=kwargs, kde_kws=kwargs)
plt.title('Cummulative Distribiution of subscribers')
plt.xlabel('log2(subscribers)')
plt.show()
# There are few channels where I am unable to get the channels details so I need to figure out an imputation strategy.
dummy_df = pd.read_csv('../input/channel-details/video_new.csv')
df2=channels_df.merge(dummy_df,on='channelId')
df2.head()
vgb_df.head()
df3 = vgb_df.merge(df2,how='left',on='video_id')[['channelId','channel_title','publish_date','viewCount_x','subCount','video_id','category','views','date','videoCount']]

df3.head()
data = dict(df3.channelId.value_counts())
x = []
y= []
for i in data.keys():
    x.append(channels_df['subCount'][channels_df['channelId'] == i].iloc[0])
    y.append(data[i])
    

iplot({'data':go.Scatter(x=np.log(np.array(x)+1),y=y,mode='markers',hovertext = x),'layout':{'title':'Number of log(Subscriber Count) vs Number of Days channels are in trending status','xaxis':{'title':'log(subcount)'},'yaxis':{'title':'Number of days a channel in trending status'},'template':'plotly_dark'}})
categories = vgb_df.category.unique()
traces = []
for i in categories:
    traces.append(go.Box(y=np.log(np.array(df3['subCount'][df3['category']==i].dropna())+1),name=i))
    
iplot({'data':traces,'layout':{'title':'Subscribers in Each Category','template':'plotly_dark','xaxis':{'title':'categories'},'yaxis':{'title':'log(subscribers)'}}})
    
vgb_df.head()
df3.head()


