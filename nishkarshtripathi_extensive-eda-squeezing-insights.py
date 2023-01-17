# Importing the Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import plotly.figure_factory as ff

import plotly.graph_objs as go

import plotly

from wordcloud import WordCloud, STOPWORDS





import json

from datetime import datetime
# Getting our data



youtube = pd.read_csv("../input/youtube-new/INvideos.csv")
youtube.head()
print("Number of rows in data :",youtube.shape[0])

print("Number of columns in data :", youtube.shape[1])
youtube.columns
print(youtube.nunique())
youtube.info()
temp = youtube.copy() # creating a copy just in case
youtube[youtube.description.isnull()].head()
desc = youtube['description'].isnull().sum()/(len(youtube))*100



print(f"Description column has {desc.round(2)}% null values.")
# Replacing all the NaN values to a empty string



youtube["description"] = youtube["description"].fillna(value="")
# Making format of date and time better



youtube.trending_date = pd.to_datetime(youtube.trending_date, format='%y.%d.%m')

youtube.publish_time = pd.to_datetime(youtube.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')

youtube.category_id = youtube.category_id.astype(str)



youtube.head()
# creating a new category column by loading json 



id_to_category = {}



with open('../input/youtube-new/IN_category_id.json' , 'r') as f:

    data = json.load(f)

    for category in data['items']:

        id_to_category[category['id']] = category['snippet']['title']

        

youtube['category'] = youtube['category_id'].map(id_to_category)
youtube.head()
# Looking at each category and number of unique values



youtube['category'].value_counts()
zero_dislikes = len(youtube.dislikes)-youtube.dislikes.astype(bool).sum(axis=0)

zero_likes = len(youtube.likes)-youtube.likes.astype(bool).sum(axis=0)



print(f"There are {zero_likes} videos with 0 likes.")

print(f"There are {zero_dislikes} videos with 0 dislikes.")
# this will hold all the ratios

likes_dislikes = {}



for i in range(len(youtube['likes'])):

    

    # if the value of dislikes is not zero

    if youtube['dislikes'][i]!=0:

        

        # compute the ratio

        likes_dislikes[i]=youtube['likes'][i]/youtube['dislikes'][i]

        

    else:

        

        # simply use the likes value

        likes_dislikes[i]=youtube['likes'][i]

        

youtube['likes_dislikes_ratio'] = likes_dislikes.values()
youtube.head()
print(f"Does the data contain duplicate video_ids? - {youtube.video_id.duplicated().any()}")
print(f"Before Deduplication : {youtube.shape}")

youtube = youtube[~youtube.video_id.duplicated(keep='last')]

print(f"After Deduplication : {youtube.shape}")



print(f"Does the data contain duplicate video_ids now? - {youtube.video_id.duplicated().any()}")
# Creating a custom formatter for pandas describe function



import contextlib

import numpy as np

import pandas as pd

import pandas.io.formats.format as pf

np.random.seed(2015)



pd.set_option('display.max_colwidth', 100)



@contextlib.contextmanager

def custom_formatting():

    orig_float_format = pd.options.display.float_format

    orig_int_format = pf.IntArrayFormatter



    pd.options.display.float_format = '{:0,.2f}'.format

    class IntArrayFormatter(pf.GenericArrayFormatter):

        def _format_strings(self):

            formatter = self.formatter or '{:,d}'.format

            fmt_values = [formatter(x) for x in self.values]

            return fmt_values

    pf.IntArrayFormatter = IntArrayFormatter

    yield

    pd.options.display.float_format = orig_float_format

    pf.IntArrayFormatter = orig_int_format



with custom_formatting():

    print(youtube[['views','likes','dislikes','comment_count']].describe())
# Plotting the Heatmap of the columns using correlation matrix



f,ax = plt.subplots(figsize=(20, 10))

sns.heatmap(youtube.corr(), annot=True, linewidths=0.5,linecolor="red",ax=ax)

plt.show()
# Extracting the year from the 'trending date' and converting to a list

video_by_year = temp["trending_date"].apply(lambda x: '20' + x[:2]).value_counts().tolist()
# Plotting a pie chart for number of videos by year



labels = ['2017','2018']

values = [video_by_year[1],video_by_year[0]]

colors = ['turquoise', 'royalblue']



trace = go.Pie(labels=labels, values=values, textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, line=dict(color='#000000', width=2)))



plotly.offline.iplot([trace], filename='styled_pie_chart')
def plot_distributions(col, i, colors):



    column_name = col+'_log'

    youtube[column_name] = np.log(youtube[col] + 1)



    group_labels = [column_name]

    hist_data = [youtube[column_name]]

    

    colors = [colors]



    # Create distplot with curve_type set to 'normal'

    fig = ff.create_distplot(hist_data, group_labels = group_labels, colors=colors,

                             bin_size=0.1, show_rug=False)



    # Add title

    title_dict = {1:'Views', 2:'Likes', 3:'Dislikes', 4:'Likes and Dislikes Ratio', 5:'Comment Count'}

    fig.update_layout(width=700, title_text= title_dict[i]+' Log Distribution')

    fig.show()
columns_list = ['views', 'likes', 'dislikes', 'likes_dislikes_ratio', 'comment_count']

colors = ['coral', 'darkmagenta', 'green', 'red', 'blue']



for i in range(0,5):

    plot_distributions(columns_list[i], i+1, colors[i])
one_mil = youtube[youtube['views'] > 1000000]['views'].count() / youtube['views'].count() * 100



print(f"Only {round(one_mil, 2)}% videos have more than 1 Million views.")
hundered_k = youtube[youtube['likes'] > 100000]['likes'].count() / youtube['likes'].count() * 100



print(f"Only {round(hundered_k, 2)}% videos have more than 1OOK Likes.")
five_k = youtube[youtube['dislikes'] > 5000]['dislikes'].count() / youtube['dislikes'].count() * 100



print(f"Only {round(five_k, 2)}% videos have more than 5K Dislikes.")
five_k = youtube[youtube['comment_count'] > 5000]['comment_count'].count() / youtube['comment_count'].count() * 100



print(f"Only {round(five_k, 2)}% videos have more than 5K Comments.")
most_likes = youtube.loc[youtube[['views']].idxmax()]['title']

most_views = youtube.loc[youtube[['likes']].idxmax()]['title']

most_dislikes = youtube.loc[youtube[['dislikes']].idxmax()]['title']

most_comments = youtube.loc[youtube[['comment_count']].idxmax() ]['title']



print(f"Most Viewed Video : {most_likes.to_string(index=False)}\n")

print(f"Most Liked Video : {most_views.to_string(index=False)}\n")

print(f"Most Disliked Video : {most_dislikes.to_string(index=False)}\n")

print(f"Video with most comments : {most_comments.to_string(index=False)}")
most_likes_ratio = youtube.loc[youtube[['likes_dislikes_ratio']].idxmax() ]['title']



print(f"Video with highest likes ratio : {most_likes_ratio.to_string(index=False)}\n")
# category had the largest number of trending videos



Category = youtube.category.value_counts().index

Count = youtube.category.value_counts().values



fig = px.bar(youtube, x=Category, y=Count, labels={'x':'Category', 'y' : 'Number of Videos'})



fig.update_traces(marker_color='mistyrose', marker_line_color='darkmagenta',

                  marker_line_width=1.5)



fig.update_layout(title_text='Video Per Category')

fig.show()
# Plotting a pie chart for top 10 channels with most trending videos



x = youtube.channel_title.value_counts().head(10).index

y = youtube.channel_title.value_counts().head(10).values



trace = go.Pie(labels=x, values=y, textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, line=dict(color='#000000', width=2)))



plotly.offline.iplot([trace], filename='styled_pie_chart')
sort_by_views = youtube.sort_values(by="views" , ascending = False)



Title = sort_by_views['title'].head(10)

Views = sort_by_views['views'].head(10)



fig = px.bar(youtube, x=Title, y=Views, labels={'x':'Title', 'y' : 'Number of views'})



fig.update_traces(marker_color='gold', marker_line_color='darkmagenta',

                  marker_line_width=1.5)



fig.update_layout(title_text='Top 10 Most Watched Videos')

fig.show()
sort_by_likes = youtube.sort_values(by ="likes" , ascending = False)



Title = sort_by_likes['title'].head(10)

Likes = sort_by_likes['likes'].head(10)



fig = px.bar(youtube, x=Title, y=Likes, labels={'x':'Title', 'y' : 'Number of Likes'})



fig.update_traces(marker_color='dodgerblue', marker_line_color='olive',

                  marker_line_width=2.5)



fig.update_layout(title_text='Top 10 Most Liked Videos')

fig.show()
sort_by_dislikes = youtube.sort_values(by = "dislikes" , ascending = False)



Title = sort_by_dislikes['title'].head(10)

Dislikes = sort_by_dislikes['dislikes'].head(10)



fig = px.bar(youtube, x=Title, y=Dislikes, labels={'x':'Title', 'y' : 'Number of Dislikes'})



fig.update_traces(marker_color='tomato', marker_line_color='#000000',

                  marker_line_width=1.5)



fig.update_layout(title_text='Top 10 Most Disliked Videos',width=1200,

    height=800)

fig.show()
sort_by_comments = youtube.sort_values(by = "comment_count" , ascending = False)



Title = sort_by_comments['title'].head(10)

Comments = sort_by_comments['comment_count'].head(10)



fig = px.bar(youtube, x=Title, y=Comments, labels={'x':'Title', 'y' : 'Number of Comments'})



fig.update_traces(marker_color='papayawhip', marker_line_color='darkblue',

                  marker_line_width=2.5)



fig.update_layout(title_text='Top 10 Videos with Most Comments',width=950,

    height=700)

fig.show()
sort_by_ldr = youtube.sort_values(by = "likes_dislikes_ratio" , ascending = False)



Title = sort_by_ldr['title'].head(10)

Comments = sort_by_ldr['likes_dislikes_ratio'].head(10)



fig = px.bar(youtube, x=Title, y=Comments, labels={'x':'Title', 'y' : 'Like/Dislike'})



fig.update_traces(marker_color='cyan', marker_line_color='darkred',

                  marker_line_width=2.5)



fig.update_layout(title_text='Top 10 Videos with Most Like to Dislike Ratio',width=1100, height = 800)

fig.show()
# Utility Function for creating word cloud



def createwordcloud(data, bgcolor, title):

    plt.figure(figsize=(15,10))

    wc = WordCloud(width=1200, height=500, 

                         collocations=False, background_color=bgcolor, 

                         colormap="tab20b").generate(" ".join(data))

    plt.imshow(wc, interpolation='bilinear')

    plt.title(title)

    plt.axis('off')
# WordCloud for Title Column



title = youtube['title']

createwordcloud(title , 'black' , 'Commonly used words in Titles')
# WordCloud for Description Column



description = youtube['description'].astype('str')

createwordcloud(description , 'black' , 'Commonly used words in Description')
# WordCloud for Tags Column





tags = youtube['tags'].astype('str')

createwordcloud(tags , 'black' , 'Commonly used words in Tags')
youtube['publish_date'] = pd.to_datetime(youtube['publish_time'])

youtube['difference'] = (youtube['trending_date'] - youtube['publish_date']).dt.days
fig = px.bar(youtube, x=youtube['trending_date'], y=youtube['views'], labels={'x':'Trending Date', 'y' : 'Number of Views'})



fig.update_traces(marker_color='darkred', marker_line_color='#000000',

                  marker_line_width=0.5)



fig.update_layout(title_text='Trending Date VS Number of Views')

fig.show()
error_or_removed = youtube["video_error_or_removed"].value_counts().tolist()



labels = ['No','Yes']

values = [error_or_removed[0],error_or_removed[1]]

colors = ['orange', 'yellow']



trace = go.Pie(labels=labels, values=values, textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, line=dict(color='#000000', width=2)))



plotly.offline.iplot([trace], filename='styled_pie_chart')
error_or_removed = youtube["comments_disabled"].value_counts().tolist()



labels = ['No','Yes']

values = [error_or_removed[0],error_or_removed[1]]

colors = ['pink', 'purple']



trace = go.Pie(labels=labels, values=values, textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, line=dict(color='#000000', width=2)))



plotly.offline.iplot([trace], filename='styled_pie_chart')
error_or_removed = youtube["ratings_disabled"].value_counts().tolist()



labels = ['No','Yes']

values = [error_or_removed[0],error_or_removed[1]]

colors = ['khaki', 'olive']



trace = go.Pie(labels=labels, values=values, textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, line=dict(color='#000000', width=2)))



plotly.offline.iplot([trace], filename='styled_pie_chart')
youtube[(youtube["comments_disabled"] == True) & (youtube["ratings_disabled"] == True)]['category'].value_counts()