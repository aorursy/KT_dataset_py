
import numpy as np # linear algebra
import plotly.graph_objs as go
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import pycountry
import os
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/100-mostfollowed-twitter-accounts-as-of-dec2019/Most_followed_Twitter_acounts_2019.csv")
df.head(2)
df['Followers'] = df['Followers'].str.replace(',', '')
df['Following'] = df['Following'].str.replace(',', '')
df['Tweets'] = df['Tweets'].str.replace(',', '')

df[['Followers','Following','Tweets']] = df[['Followers','Following','Tweets']].astype(int)
# Convert all the names in the same case
df['Name'] = df['Name'].str.title()

def treemap(data,title):
    import squarify 
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(16, 4.5)
    squarify.plot(sizes=data['Activity'].value_counts().values, label=data['Activity'].value_counts().index, 
              alpha=0.5,
              text_kwargs={'fontsize':10,'weight':'bold'},
              color=["red","green","blue", "grey","orange","pink","blue","cyan"])
    plt.axis('off')
    plt.title(title,fontsize=20)
    plt.show()
    
def barplot(data):

    x1 = data['Followers']/1000000
    x1 = x1.round(2)

    trace = go.Bar(x = data['Followers'][:15],
               y = data['Name'][:15],
               orientation='h',
               marker = dict(color='#00acee',line=dict(color='black',width=0)),
               text=x1,
               textposition='auto',
               hovertemplate = "<br>Followers: %{x}")

    layout = go.Layout(
                   #title='Most followed Political accounts on Twitter worldwide as of December,2019',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Number of Followers in Millions'),
                   yaxis=dict(autorange="reversed"),
                   showlegend=False)

    fig = go.Figure(data = [trace], layout = layout)

    fig.update_layout(width=700,height=500)
    fig.show()
most_followed_account = df[["Name","Followers"]]
most_followed_account = most_followed_account.sort_values("Followers",ascending=False)[:10]
most_followed_account = most_followed_account.reset_index(drop=True)
most_followed_account




counts = df['Nationality/headquarters'].value_counts()
labels = counts.index
values = counts.values

pie = go.Pie(labels=labels, values=values,pull=[0.05, 0], marker=dict(line=dict(color='#000000', width=1)))
layout = go.Layout(title='Region wise Distribution')

fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)
df['Industry'].replace({'music':'Music',
                       'news':'News',
                       'sports':'Sports'},inplace=True)




counts = df['Industry'].value_counts()
y= counts.values
trace = go.Bar(x= counts.index,
               y= counts.values,
               marker={'color': y,'colorscale':'Picnic'})

layout = go.Layout(
                   #title='Most followed accounts on Twitter worldwide as of December,2019',
                   width=800, 
                   height=500, 
                   xaxis= dict(title='Industry'),
                   yaxis=dict(title='Count'),
                   showlegend=False)

fig = go.Figure(data = [trace], layout = layout)

fig.update_layout(width=700,height=500)
fig.show()

music = df[df['Industry'] == 'Music']
music['Activity'].replace({'singer-songwriter':'Singer and Songwriter',
                            'Singer and songwriter':'Singer and Songwriter'},inplace=True)


barplot(music)
treemap(music,'Various Categories in Music')


USA=df[df['Nationality/headquarters']=='U.S.A']
barplot(USA)
