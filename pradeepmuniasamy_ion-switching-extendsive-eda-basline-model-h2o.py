import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.figure_factory as ff ## For distiribution plot

import plotly.offline as py

import plotly.express as px

import plotly.graph_objects as go





import matplotlib.pylab as plt

import seaborn as sns



py.init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_df  = pd.read_csv("/kaggle/input/liverpool-ion-switching/sample_submission.csv")

test_df = pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")

train_df = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")
train_df.shape , test_df.shape
train_df.isnull().sum()
train_df.dtypes
train_df.time.is_monotonic
print("No of batches in Train Data",train_df.shape[0]/500000)

print("No of batches in Test Data",test_df.shape[0]/500000)
train_df['batch'] = 0

test_df['batch'] = 0

for i in range(0, 10):

    train_df.iloc[i * 500000: 500000 * (i + 1), 3] = i

for i in range(0, 4):

    test_df.iloc[i * 500000: 500000 * (i + 1), 2] = i
train_df.open_channels.unique()
from IPython.display import Image

Image("/kaggle/input/ion-channel/ion image.jpg")
temp_df = train_df.groupby(["open_channels"])["open_channels"].agg(["count"]).reset_index()



fig = px.bar(temp_df, x='open_channels', y='count',

             hover_data=['count'], color='count',

             labels={'pop':'Distribtuion of Open Channels'}, height=400)

fig.show()
temp_df = train_df.groupby(["open_channels","batch"])["open_channels"].agg(["count"]).reset_index()

temp_df.columns = ["open_channels","batch","count"]

#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']



fig = px.scatter(temp_df, x="open_channels", y="batch", color="open_channels", size="count")

layout = go.Layout(

    title=go.layout.Title(

        text="Ground Truth across each batches",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
fig, axs = plt.subplots(5, 2, figsize=(15, 20))

axs = axs.flatten()

i = 0

for b, d in train_df.groupby('batch'):

    sns.violinplot(x='open_channels', y='signal', data=d, ax=axs[i])

    axs[i].set_title(f'Batch {b:0.0f}')

    i += 1

plt.tight_layout()
fig = px.line(train_df[:200000] , x='time', y='signal')

fig.show()
from plotly.subplots import make_subplots

import plotly.graph_objects as go





fig = make_subplots(rows=5, cols=2,  subplot_titles=tuple(["Batch No"+str(i) for i in range(0,10)]))



batch_no = 0 

for i in range(0,5):

    for j in range(0,2):

        temp = train_df.loc[train_df["batch"]==batch_no]

        temp = temp[:10000]

        batch_no+=1

        fig.add_trace(

            go.Scatter(

            x=temp['time'],

            y=temp['signal'],

               ),

            row=i+1, col=j+1      

         )

        fig.update_xaxes(title_text="Time", row=i+1, col=j+1)

        fig.update_yaxes(title_text="Signal", row=i+1, col=j+1)



fig.update_layout(height=1000, width=1200, title_text="Signal spread across the batches")



fig.show()
from plotly.subplots import make_subplots

import plotly.graph_objects as go





fig = make_subplots(rows=2, cols=2,  subplot_titles=tuple(["Batch No"+str(i) for i in range(0,4)]))



batch_no = 0 

for i in range(0,2):

    for j in range(0,2):

        temp = test_df.loc[train_df["batch"]==batch_no]

        temp = temp[:10000]

        batch_no+=1

        fig.add_trace(

            go.Scatter(

            x=temp['time'],

            y=temp['signal'],

               ),

            row=i+1, col=j+1      

         )

        fig.update_xaxes(title_text="Time", row=i+1, col=j+1)

        fig.update_yaxes(title_text="Signal", row=i+1, col=j+1)



fig.update_layout(height=1000, width=1200, title_text="Signal spread across the batches")
