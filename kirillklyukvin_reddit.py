#common

import numpy as np

import pandas as pd 

import IPython

from IPython.display import display



#visualisation

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px
reddit = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv', low_memory=False)
reddit.info()
reddit.head(3).T
reddit.loc[reddit['awarders'] == "[]", 'awarders'] = 0

reddit.loc[reddit['awarders'] == "['stompstumpstamp']", 'awarders'] = 1

reddit.loc[reddit['awarders'].isna(), 'awarders'] = 0
nan_replacements = {"author_flair_text": 'none', "removed_by": 'avalible', "total_awards_received": 0, "title": 'unknown'}

reddit = reddit.fillna(nan_replacements)
reddit.isna().sum()
reddit['created_loc_time'] = pd.to_datetime(reddit['created_utc'], unit='s')
reddit['year'] = [d.year for d in reddit['created_loc_time']]

reddit['month'] = [d.month for d in reddit['created_loc_time']]

reddit['day'] = [d.day for d in reddit['created_loc_time']]

reddit['dayofweek'] = [d.isoweekday() for d in reddit['created_loc_time']]

reddit['hour'] = [d.hour for d in reddit['created_loc_time']]
reddit.removed_by.value_counts()
reddit.loc[reddit['removed_by'] == 'avalible', 'is_avalible'] = 1

reddit.loc[reddit['removed_by'] != 'avalible', 'is_avalible'] = 0
reddit.head(3).T
top_authors = reddit.query('author != "[deleted]"')['author'].value_counts().reset_index()

top_authors.head(5)
(reddit.query('author != "[deleted]"').

 groupby('author')['score'].sum().

 reset_index().sort_values(by='score', ascending=False).head()

)
%%time

reddit[['title', 'score', 'author', 'year']].sort_values(by='score', ascending=False).head()
reddit[['title', 'num_comments', 'author', 'year']].sort_values(by='num_comments', ascending=False).head()
adult_content = len(reddit.query('over_18 == True')) / len(reddit.query('over_18 == False'))

print('Total amount of adult-only content = {:.1%}'.format(adult_content))
fig, axes = plt.subplots(3,3, figsize=(15,12), sharey=True)

fig.suptitle('Number of posts by months', fontsize=20, y=0.92)



temp = reddit.query('is_avalible == 1')

y = 2012



for i in range(3):

    for k in range(3):

        fig = sns.countplot(data=temp[temp['year'] == y], x='month', ax=axes[i,k], color='royalblue')

        axes[i,k].set(xlabel=y)

        axes[i,k].set(ylabel='')

        y += 1

        

plt.show()
# Create a list of most posted authors

temp = top_authors.head(10)

temp_1 = temp['index']



# Create a table with the valid data

temp_2 = reddit.query('author in @temp_1')[['author', 'year', 'id']]



temp_pivot = (

    temp_2.pivot_table(index=['year', 'author'], values='id', aggfunc='count')

    .reset_index().sort_values(by='year', ascending=False)

)



temp_pivot.rename(columns={"id": "posts"}, inplace=True)



# Plot a figure with the posts distribution, grouped by years

fig = go.Figure()



fig = px.line(temp_pivot, x="year", y="posts", color="author")

    

annotations = []



annotations.append(dict(xref='paper', yref='paper', x=0.5, y=1.05,

                              #xanchor='left', 

                              yanchor='bottom',

                              text='Top 10 of most writing authors by years',

                              font=dict(size=20,

                                        color='black'),

                              showarrow=False))

# Set the fig' layout

fig.update_layout(annotations=annotations, plot_bgcolor='white')

fig.update_xaxes(gridwidth=0.9, gridcolor='silver', linecolor='black', zerolinewidth=1, zerolinecolor='dimgray')

fig.update_yaxes(gridwidth=0.9, gridcolor='silver', linecolor='black', zerolinewidth=1, zerolinecolor='dimgray')



fig.show()
hour_distr = reddit.hour.value_counts().sort_values()



#colors = px.colors.cyclical.Twilight



fig = px.pie(values=hour_distr.values, names=hour_distr.index, color_discrete_sequence=px.colors.cyclical.Edge)

             

fig.update_traces(

                  textposition="inside", 

                  textinfo="value+percent+label", insidetextorientation='radial', 

                  hole=.2, 

                 )



fig.update_layout(title_text="Proportion of posts by hour")



fig.show()
x = np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])

y = np.arange(0, 26)



#colors = color_discrete_sequence=px.colors.cyclical.Edge



#fig = go.Figure(data=[go.Pie(labels=x,

                             #values=y,

                             #template='plotly_dark'

                           # )])



fig = px.pie(values=y, names=x, 

             color_discrete_sequence=(px.colors.cyclical.Twilight), 

             #template='plotly_white'

            )



fig.update_traces(textposition="inside", 

                  textinfo="value+label", insidetextorientation='radial', 

                  hole=.2, 

                  marker=dict(

                              #colors=colors, 

                              line=dict(color='dimgray', width=1))

                )



fig.show()
daily_posts = reddit.groupby('day')['id'].count().reset_index()



plt.figure(figsize=(10,6))



sns.set(style='whitegrid')

ax = sns.barplot(data=daily_posts, x='day', y='id', color='royalblue')

plt.title('Total number of daily posts', fontstyle='italic', size=15)

ax.set(xlabel='Day of month', ylabel='Posts')

plt.show()
# Remove 2012-2014 and 2020 from our chart

reddit_1 = reddit.query('2020 > year >= 2015')

daily_posts_1 = reddit_1.groupby(['month', 'day'])['id'].count().reset_index()



plt.figure(figsize=(15,10))

ax = sns.boxplot(data=daily_posts_1, x='month', y='id', 

                 showfliers=False, color='royalblue', linewidth=2

                )



sns.despine(offset=10, trim=True)

ax.set(xlabel='Month', ylabel='Posts')

plt.title('Distribution of monthly post from 2015 till 2019', size=15)

plt.show()
from wordcloud import WordCloud, STOPWORDS
words = reddit["title"].values

len(words)
type(words[80324])
ls_1 = []



for i in words:

    ls_1.append(str(i))
ls = []



for i in range(len(reddit)):

    ls.append(reddit.author[i])
plt.figure(figsize=(16,13))

wc = WordCloud(background_color="white", stopwords = STOPWORDS, max_words=1000, max_font_size= 200,  width=1600, height=800)

wc.generate(" ".join(ls))

plt.title("Most discussed terms", fontsize=20)

plt.imshow(wc.recolor(colormap='viridis' , random_state=17), alpha=0.98, interpolation="bilinear")

plt.axis('off')
most_pop = reddit.sort_values('score', ascending =False)[['title', 'score']].head(12)



most_pop['score1'] = most_pop['score']/1000
most_pop
import matplotlib.style as style
style.available
style.use('fivethirtyeight')
plt.figure(figsize = (17,15))



sns.barplot(data = most_pop, y = 'title', x = 'score1', color = 'c')

plt.xticks(fontsize=20, rotation=0)

plt.yticks(fontsize=21, rotation=0)

plt.xlabel('Votes in Thousands', fontsize = 21)

plt.ylabel('')

plt.title('Most popular posts', fontsize = 30)