import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from plotly.offline import iplot

from plotly import tools

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)



import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

import matplotlib.patches as patches
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs_Df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
train_df.shape
train_df.isnull().sum()
train_labels.head()   
train_labels.title.unique()
#temp=df.drop_duplicates('GameId', keep='last')

temp_df = train_labels.groupby(["title","accuracy_group"])["accuracy_group"].agg(["count"]).reset_index()

temp_df.columns = ["title","accuracy_group", "Count"]

#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']



fig = px.scatter(temp_df, x="accuracy_group", y="title", color="accuracy_group", size="Count")

layout = go.Layout(

    title=go.layout.Title(

        text="Accuracy group in each Assesments",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
Accuracy=pd.DataFrame()

Accuracy['Type']=train_labels.accuracy_group.value_counts().index

Accuracy['Count']=train_labels.accuracy_group.value_counts().values



import plotly.offline as pyo

py.init_notebook_mode(connected=True)

fig = go.Figure(data=[go.Pie(labels=Accuracy['Type'], values=Accuracy['Count'],hole=0.2)])

fig.show()
Success_Rate_1=pd.DataFrame()

Success_Rate_2=pd.DataFrame()

Success_Rate_3=pd.DataFrame()

Success_Rate_4=pd.DataFrame()

Success_Rate_5=pd.DataFrame()

Mushroom_Sorter=train_labels.loc[train_labels['title'] == 'Mushroom Sorter (Assessment)']

Success_Rate_1['Type']=Mushroom_Sorter.num_correct.value_counts().index

Success_Rate_1['Count']=Mushroom_Sorter.num_correct.value_counts().values

Bird_Measurer=train_labels.loc[train_labels['title'] ==  'Bird Measurer (Assessment)']

Success_Rate_2['Type']=Bird_Measurer.num_correct.value_counts().index

Success_Rate_2['Count']=Bird_Measurer.num_correct.value_counts().values

Cauldron_Filler=train_labels.loc[train_labels['title'] == 'Cauldron Filler (Assessment)']

Success_Rate_3['Type']=Cauldron_Filler.num_correct.value_counts().index

Success_Rate_3['Count']=Cauldron_Filler.num_correct.value_counts().values

Chest_Sorter=train_labels.loc[train_labels['title'] == 'Chest Sorter (Assessment)']

Success_Rate_4['Type']=Chest_Sorter.num_correct.value_counts().index

Success_Rate_4['Count']=Chest_Sorter.num_correct.value_counts().values

Cart_Balancer=train_labels.loc[train_labels['title'] == 'Cart Balancer (Assessment)']

Success_Rate_5['Type']=Cart_Balancer.num_correct.value_counts().index

Success_Rate_5['Count']=Cart_Balancer.num_correct.value_counts().values
import plotly.graph_objects as go

from plotly.subplots import make_subplots



labels = [0,1]



fig = make_subplots(3, 2, specs=[[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}],[{'type':'domain'}, {'type':'domain'}]],

                    subplot_titles=['Mushroom Sorter', 'Bird Measurer','Cauldron Filler','Chest Sorter','Cart Balancer'])

fig.add_trace(go.Pie(labels=Success_Rate_1['Type'], values=Success_Rate_1['Count'], scalegroup='one',

                     name="Success Rate"), 1, 1)

fig.add_trace(go.Pie(labels=Success_Rate_2['Type'], values=Success_Rate_2['Count'], scalegroup='one',

                     name="Success Rate"), 1, 2)

fig.add_trace(go.Pie(labels=Success_Rate_3['Type'], values=Success_Rate_3['Count'], scalegroup='one',

                     name="Success Rate"), 2, 1)

fig.add_trace(go.Pie(labels=Success_Rate_4['Type'], values=Success_Rate_4['Count'], scalegroup='one',

                     name="Success Rate"), 2, 2)

fig.add_trace(go.Pie(labels=Success_Rate_5['Type'], values=Success_Rate_5['Count'], scalegroup='one',

                     name="Success Rate"), 3, 1)



fig.update_layout(title_text='Success Rate of Each Group')

fig.show()
def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings

            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            

            #Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

               NAlist.append(col)

               df[col].fillna(-999,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] =df[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df,NAlist
train_df,train_Na=reduce_mem_usage(train_df)
import gc

gc.collect()
test_df,test_Na=reduce_mem_usage(test_df)
train_df.columns
data={'Unique_event':[train_df.event_id.nunique()],

      'Unique_gamesession':[train_df.game_session.nunique()],

      'Unique_title':[train_df.title.nunique()]}

Count_df=pd.DataFrame(data)

Count_df
# Format and make date / hour features

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

train_df['date'] = train_df['timestamp'].dt.date

train_df['hour'] = train_df['timestamp'].dt.hour

train_df['weekday_name'] = train_df['timestamp'].dt.weekday_name



# Same for test

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

test_df['date'] = test_df['timestamp'].dt.date

test_df['hour'] = test_df['timestamp'].dt.hour

test_df['weekday_name'] = test_df['timestamp'].dt.weekday_name
train_df.groupby('date')['event_id'].agg('count').plot(figsize=(15, 3),title='Numer of Event Observations by Date',

                                                       color="blue")

test_df.groupby('date')['event_id'].agg('count').plot(figsize=(15, 3),title='Numer of Event Observations by Date'

                                                      ,color="yellow")

train_patch = mpatches.Patch(color='blue', label='Train data')

test_patch = mpatches.Patch(color='yellow', label='Test data')

plt.legend(handles=[train_patch, test_patch])

plt.grid()

plt.show()
train_df.groupby('hour')['event_id'].agg('count').plot(figsize=(15, 3),title='Numer of Event Observations by Hour',color="blue")

test_df.groupby('hour')['event_id'].agg('count').plot(figsize=(15, 3),title='Numer of Event Observations by Hour',color="yellow")

train_patch = mpatches.Patch(color='blue', label='Train data')

test_patch = mpatches.Patch(color='yellow', label='Test data')

plt.legend(handles=[train_patch, test_patch])

plt.grid()

plt.show()
train_df.groupby('weekday_name')['event_id'].agg('count').T[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].T.plot(figsize=(15, 3),title='Numer of Event Observations by Day of Week',color="blue")

test_df.groupby('weekday_name')['event_id'].agg('count').T[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].T.plot(figsize=(15, 3),title='Numer of Event Observations by Day of Week',color="yellow")

train_patch = mpatches.Patch(color='blue', label='Train data')

test_patch = mpatches.Patch(color='yellow', label='Test data')

plt.legend(handles=[train_patch, test_patch])

plt.grid()

plt.show()
Game=pd.DataFrame()

Game['Type']=train_df.type.value_counts().index

Game['Count']=train_df.type.value_counts().values



import plotly.offline as pyo

py.init_notebook_mode(connected=True)

fig = go.Figure(data=[go.Pie(labels=Game['Type'], values=Game['Count'],hole=0.2)])

fig.show()
Game=pd.DataFrame()

Game['Title']=train_df.title.value_counts().index

Game['Count']=train_df.title.value_counts().values



fig = px.bar(Game, x='Title', y='Count',

             hover_data=['Count'], color='Count',

             labels={'pop':'Total Number of game titles'}, height=400)

fig.show()
avg_time=[]

type_=[]

for i in train_df.type.unique():

    type_.append(i)

    avg_time.append(train_df.loc[train_df['type'] ==i]['game_time'].mean())

    

Avg_Timeplayed=pd.DataFrame()

Avg_Timeplayed['Type']=type_

Avg_Timeplayed['Average']=avg_time



fig = px.bar(Avg_Timeplayed, x='Type', y='Average',

             hover_data=['Average'], color='Average',

             labels={'pop':'Average time played on each types'}, height=400)

fig.show()
avg_time=[]

title_=[]

for i in train_df.title.unique():

    title_.append(i)

    avg_time.append(train_df.loc[train_df['title'] ==i]['game_time'].mean())

    

Avg_Timeplayed=pd.DataFrame()

Avg_Timeplayed['Title']=title_

Avg_Timeplayed['Average']=avg_time



fig = px.bar(Avg_Timeplayed, x='Title', y='Average',

             hover_data=['Average'], color='Average',

             labels={'pop':'Average time played on each titles'}, height=400)

fig.show()