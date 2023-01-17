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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
original = pd.read_csv('../input/ks-projects-201801.csv')
original['deadline']=pd.to_datetime(original['deadline'], format="%Y/%m/%d").dt.date
original['launched']=pd.to_datetime(original['launched'], format="%Y/%m/%d").dt.date
original.head(4)
df = original.drop(['ID','goal','pledged','usd_pledged_real'],1)
df['duration(days)'] = (df['deadline'] - df['launched']).dt.days
df['launch_year']=pd.to_datetime(original['launched'], format="%Y/%m/%d").dt.year
df.head()
state_count = df.state.value_counts()
go1 = go.Bar(
            x=state_count.index,
            y=state_count.values,
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )

data = [go1]
layout=go.Layout(title="Distribution of States", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=600, height=400)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
go1 = go.Bar(
            x=df.main_category.value_counts().index,
            y=df.main_category.value_counts().values,
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )

data = [go1]
layout=go.Layout(title="Distribution of Main Categories", xaxis={'title':'Main Category'}, yaxis={'title':'No of Campaigns'}, width=600, height=400)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
go1 = go.Bar(
            x=df.category.value_counts()[:20].index,
            y=df.category.value_counts()[:20].values,
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )
data = [go1]
layout=go.Layout(title="Distribution of Top 20 Categories", xaxis={'title':'Category'}, yaxis={'title':'No of Campaigns'}, width=600, height=400)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
go1 = go.Bar(
            x=df.country.value_counts().index,
            y=df.country.value_counts().values,
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )
data = [go1]
layout=go.Layout(title="Distribution of Countries", xaxis={'title':'Countries'}, yaxis={'title':'No of Campaigns'}, width=600, height=400)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
fig = sns.countplot(df.launch_year)
plt.xlabel("Year")
plt.ylabel("Number of Campaigns")
plt.title("No of Campaigns vs Year")
plt.show(fig)
failed = df.loc[df.state=='failed']
successful = df.loc[df.state=='successful']
canceled = df.loc[df.state=='canceled']
print('Mean duration of failed campaigns',failed['duration(days)'].mean())
print('Mean duration of successful campaigns',successful['duration(days)'].mean())
print('Mean duration of canceled campaigns',canceled['duration(days)'].mean())
trace1 = go.Bar(
            x=successful.main_category.value_counts().index,
            y=successful.main_category.value_counts().values,
            opacity=0.65
        )

trace2 = go.Bar(
            x=failed.main_category.value_counts().index,
            y=failed.main_category.value_counts().values,
            opacity=0.65
        )

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Successful', 'Failed'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=300, width=900, title='Distribution of main categories in Sucessful & Failed Campaigns')

iplot(fig)
trace1 = go.Bar(
            x=successful.category.value_counts()[:20].index,
            y=successful.category.value_counts()[:20].values,
            opacity=0.65
        )

trace2 = go.Bar(
            x=failed.category.value_counts()[:20].index,
            y=failed.category.value_counts()[:20].values,
            opacity=0.65
        )

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Successful', 'Failed'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=300, width=900, title='Distribution of top 20 categories in Sucessful & Failed Campaigns')

iplot(fig)
groupby_main_category = successful.groupby(['main_category']).mean()
groupby_main_category
trace1 = go.Bar(
            x=groupby_main_category.backers.index,
            y=groupby_main_category.backers.values,
            opacity=0.65
        )

trace2 = go.Bar(
            x=groupby_main_category['usd pledged'].index,
            y=groupby_main_category['usd pledged'].values,
            opacity=0.65
        )

trace3 = go.Bar(
            x=groupby_main_category.usd_goal_real.index,
            y=groupby_main_category.usd_goal_real.values,
            opacity=0.65
        )

trace4 = go.Bar(
            x=groupby_main_category['duration(days)'].index,
            y=groupby_main_category['duration(days)'].values,
            opacity=0.65
        )

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Backers', 'USD Pledged','USD Goal Real','Duration(days)'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)

fig['layout'].update(height=600, width=900, title='Distribution according to Main Category of Successful Campaigns')

iplot(fig)
sns.regplot(x='backers',y='usd pledged', data=successful)
sns.regplot(x='usd pledged',y='duration(days)', data=successful)
successful['success_measure'] = successful['usd pledged']/successful['usd_goal_real']
successful_cleaned = successful[successful['usd_goal_real']>successful['usd_goal_real'].median()]
successful_cleaned.nlargest(10,'success_measure')