import numpy as np

import pandas as pd

from scipy.optimize import curve_fit

import seaborn as sns



import matplotlib.pyplot as plt

import matplotlib.colors as colors

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff
in_videos = pd.read_csv('../input/INvideos.csv')

in_videos_categories = pd.read_json('../input/IN_category_id.json')
in_videos.head(1)
in_videos.info()
in_videos = in_videos.drop(['description'], axis = 1)

################################### Use only once (Fails after 1st Attempt) ##################################

# Transforming Trending date column to datetime format

in_videos['trending_date'] = pd.to_datetime(in_videos['trending_date'], format='%y.%d.%m').dt.date



# Transforming Trending date column to datetime format and splitting into two separate ones

publish_time = pd.to_datetime(in_videos['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

in_videos['publish_date'] = publish_time.dt.date

in_videos['publish_time'] = publish_time.dt.time

in_videos['publish_hour'] = publish_time.dt.hour
in_videos.head(1)
################################### Use only once (Fails after 1st Attempt) ##################################

# We'll use a very nice python featur - dictionary comprehension, to extract most important data from IN_category_id.json

categories = {category['id']: category['snippet']['title'] for category in in_videos_categories['items']}



# Now we will create new column that will represent name of category

in_videos.insert(4, 'category', in_videos['category_id'].astype(str).map(categories))

in_videos.tail(3)
in_videos_first = in_videos.copy() 

in_videos_first['dislike_percentage'] = in_videos['dislikes'] / (in_videos['dislikes'] + in_videos['likes'])

print(in_videos_first['dislike_percentage'].head(5))
# Helper function

def numberOfUpper(string):

    i = 0

    for word in string.split():

        if word.isupper():

            i += 1

    return(i)



in_videos_first["all_upper_in_title"] = in_videos["title"].apply(numberOfUpper)

print(in_videos_first["all_upper_in_title"].tail(5))
in_videos_first['likes_log'] = np.log(in_videos['likes'] + 1)

in_videos_first['views_log'] = np.log(in_videos['views'] + 1)

in_videos_first['dislikes_log'] = np.log(in_videos['dislikes'] + 1)

in_videos_first['comment_log'] = np.log(in_videos['comment_count'] + 1)



plt.figure(figsize = (12,6))



plt.subplot(221)

g1 = sns.distplot(in_videos_first['views_log'])

g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)



plt.subplot(224)

g2 = sns.distplot(in_videos_first['likes_log'],color='green')

g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)



plt.subplot(223)

g3 = sns.distplot(in_videos_first['dislikes_log'], color='r')

g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)



plt.subplot(222)

g4 = sns.distplot(in_videos_first['comment_log'])

g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)



plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)



plt.show()
in_videos_last = in_videos.drop_duplicates(subset=['video_id'], keep='last', inplace=False)

in_videos_first = in_videos.drop_duplicates(subset=['video_id'], keep='first', inplace=False)

print(in_videos_last.head(2))
print("in_videos dataset contains {} videos".format(in_videos.shape[0]))

print("in_videos_first dataset contains {} videos".format(in_videos_first.shape[0]))

print("in_videos_last dataset contains {} videos".format(in_videos_last.shape[0]))
in_videos["days_before_trend"] = (in_videos.trending_date - in_videos.publish_date) / np.timedelta64(1, 'D')

in_videos["days_before_trend"] = in_videos["days_before_trend"].astype(int)

in_videos.tail(3)
in_videos.isnull().sum()
null_data = in_videos[in_videos["category"].isnull()]

null_data.head(2)
in_videos["category"].fillna("Nonprofits & Activism", inplace = True) 

in_videos[in_videos["category_id"]  == 29]

in_videos[in_videos["category_id"]  == 29].tail(3)
in_videos.loc[(in_videos['days_before_trend'] < 1), 'days_before_trend'] = 1

in_videos["views_per_day"] = in_videos["views"].astype(int) / in_videos["days_before_trend"]

in_videos["views_per_day"] = in_videos["views_per_day"]

in_videos.tail(3)
in_videos.isnull().sum()
in_videos.to_csv('preprocessedIndia.csv',index=False)
# Initialization of the list storing counters for subsequent publication hours

publish_h = [0] * 24



for index, row in in_videos_first.iterrows():

    publish_h[row["publish_hour"]] += 1

    

values = publish_h

ind = np.arange(len(values))





# Creating new plot

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

ax.yaxis.grid()

ax.xaxis.grid()

bars = ax.bar(ind, values)



# Sampling of Colormap

for i, b in enumerate(bars):

    b.set_color(plt.cm.viridis((values[i] - min(values))/(max(values)- min(values))))

    

plt.ylabel('Number of videos that got trending', fontsize=20)

plt.xlabel('Time of publishing', fontsize=20)

plt.title('Best time to publish video', fontsize=35, fontweight='bold')

plt.xticks(np.arange(0, len(ind), len(ind)/6), [0, 4, 8, 12, 16, 20])



plt.show()
h_labels = [x.replace('_', ' ').title() for x in 

            list(in_videos.select_dtypes(include=['number', 'bool']).columns.values)]



fig, ax = plt.subplots(figsize=(10,6))

_ = sns.heatmap(in_videos.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
from IPython.display import HTML, display



# We choose the 10 most trending videos

selected_columns = ['title', 'channel_title', 'thumbnail_link', 'publish_date', 'category']



most_frequent = in_videos.groupby(selected_columns)['video_id'].agg(

    {"code_count": len}).sort_values(

    "code_count", ascending=False

).head(10).reset_index()



# Construction of HTML table with miniature photos assigned to the most popular movies

table_content = ''

max_title_length = 50



for date, row in most_frequent.T.iteritems():

    HTML_row = '<tr>'

    HTML_row += '<td><img src="' + str(row[2]) + '"style="width:100px;height:100px;"></td>'

    HTML_row += '<td>' + str(row[1]) + '</td>'

    HTML_row += '<td>' + str(row[0])  + '</td>'

    HTML_row += '<td>' + str(row[4]) + '</td>'

    HTML_row += '<td>' + str(row[3]) + '</td>'

    

    table_content += HTML_row + '</tr>'



display(HTML(

    '<table><tr><th>Photo</th><th>Channel Name</th><th style="width:250px;">Title</th><th>Category</th><th>Publish Date</th></tr>{}</table>'.format(table_content))

)
max_title_length = 20

number_of_creators = 20



top_creators = in_videos.groupby(['channel_title'])['channel_title'].agg(

    {"code_count": len}).sort_values(

    "code_count", ascending=False

).head(number_of_creators).reset_index()



trace1 = go.Bar(

    y = [(x if len(x) <= max_title_length else x[:max_title_length] + "...") for x in top_creators.channel_title.values][::-1],

    x = top_creators['code_count'].tolist()[::-1],

    name = "Top creators",

    orientation = 'h',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    ),

)



data = [trace1]



layout = go.Layout(

    title = 'Most influential creators',

    width=900,

    height=600,

    margin=go.Margin(

        l=180,

        r=50,

        b=80,

        t=80,

        pad=10

    ),

    paper_bgcolor='rgb(244, 238, 225)',

    plot_bgcolor='rgb(244, 238, 225)',

    yaxis = dict(

        anchor = 'x',

        rangemode='tozero',

        tickfont=dict(

            size=10

        ),

        ticklen=1

    ), 

    xaxis = dict(

        title= 'Number of times video made by creator got trending',

        anchor = 'x',

        rangemode='tozero'

    ), 

    legend=dict(x=0.6, y=0.07)

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
max_title_length = 30

number_of_creators = 12



top_creators = in_videos.groupby(['category'])['category'].agg(

    {"code_count": len}).sort_values(

    "code_count", ascending=False

).head(number_of_creators).reset_index()



trace1 = go.Bar(

    y = [(x if len(x) <= max_title_length else x[:max_title_length] + "...") for x in top_creators.category.values][::-1],

    x = top_creators['code_count'].tolist()[::-1],

    name = "Top categories",

    orientation = 'h',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    ),

)



data = [trace1]



layout = go.Layout(

    title = 'Most popular categories',

    width=900,

    height=600,

    margin=go.Margin(

        l=180,

        r=50,

        b=80,

        t=80,

        pad=10

    ),

    paper_bgcolor='rgb(244, 238, 225)',

    plot_bgcolor='rgb(244, 238, 225)',

    yaxis = dict(

        anchor = 'x',

        rangemode='tozero',

        tickfont=dict(

            size=10

        ),

        ticklen=1

    ), 

    xaxis = dict(

        title= 'The number of times the video of a given category was trending',

        anchor = 'x',

        rangemode='tozero'

    ), 

    legend=dict(x=0.6, y=0.07)

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# Average time interval between published and trending

in_videos['interval'] = (pd.to_datetime(in_videos['trending_date']).dt.date - pd.to_datetime(in_videos['publish_date']).dt.date).astype('timedelta64[D]')

df_t = pd.DataFrame(in_videos['interval'].groupby(in_videos['category']).mean())

plt.figure(figsize = (32,12))

plt.plot(df_t, color='skyblue', linewidth=2)

plt.title("Average Days to be trending video", fontsize=25)

plt.xlabel('Category',fontsize=22)

plt.ylabel('Average Time Interval',fontsize=22)

plt.tick_params(labelsize=14)

plt.show();

print(type(in_videos["video_id"]))
# dropping passed values 



#in_videos.drop(in_videos.video_id == '"zUZ1z7FwLc8","CLl1RbxDRAs","z3V9LUA6VQM", "jElRtesCnlA", "qP67alYxSiU", "JSkOecmAFFo", "l3fRny54z4U", "UTVFNrRwL1o", "K6JyjjNnTlg", "4tEqzEo5uKY", "8vBjlhp73hU", "KskjXRkmJW4", "NTiSvK7c810", "sOwXjFMy17Y", "h6Z9mmSNJcw"'), inplace = True) 

max_title_length = 20

number_of_late_bloomers = 15

in_videos_first["days_before_trend"]= in_videos["days_before_trend"].astype(float)

late_bloomers = in_videos_first.sort_values(["days_before_trend"], ascending=False).head(number_of_late_bloomers)

late_bloomers_title = [(x if len(x) <= max_title_length else x[:max_title_length] + "...") for x in late_bloomers.title.values]

late_bloomers_days = late_bloomers.days_before_trend.values

late_bloomers_views = late_bloomers.views.values



trace1 = go.Bar(

    x = late_bloomers_title,

    y = late_bloomers_days,

    name='Number of days',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    )

)

trace2 = go.Bar(

    x = late_bloomers_title,

    y = late_bloomers_views,

    name='total views',

    marker=dict(

        color='rgba(219, 64, 82, 0.7)',

        line=dict(

            color='rgba(219, 64, 82, 1.0)',

            width=2,

        )

    ),

    yaxis='y2'

)





data = [trace1, trace2]

layout = go.Layout(

    barmode='group',

    title = 'Late bloomers',

    width=900,

    height=500,

    margin=go.Margin(

        l=75,

        r=75,

        b=120,

        t=80,

        pad=10

    ),

    paper_bgcolor='rgb(244, 238, 225)',

    plot_bgcolor='rgb(244, 238, 225)',

    yaxis = dict(

        title= 'Number of days until becoming trending',

        anchor = 'x',

        rangemode='tozero'

    ),   

    yaxis2=dict(

        title='Total number of views',

        titlefont=dict(

            color='rgb(148, 103, 189)'

        ),

        tickfont=dict(

            color='rgb(148, 103, 189)'

        ),

        overlaying='y',

        side='right',

        anchor = 'x',

        rangemode = 'tozero',

        dtick = 61000

    ),

    #legend=dict(x=-.1, y=1.2)

    legend=dict(x=0.1, y=0.05)

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
max_title_length = 20

number_of_late_bloomers = 10

in_videos_first["dislikes"]= in_videos["dislikes"]

in_videos_first['dislike_percentage'] = in_videos['dislikes'] / (in_videos['dislikes'] + in_videos['likes'])

most_disliked = in_videos_first.sort_values(["dislikes"], ascending=False).head(number_of_late_bloomers)

most_disliked_title = [(x if len(x) <= max_title_length else x[:max_title_length] + "...") for x in late_bloomers.title.values]

most_disliked_l_number = most_disliked.likes.values

most_disliked_dl_number = most_disliked.dislikes.values

most_disliked_dl_percentage = most_disliked.dislike_percentage.values



trace1 = go.Bar(

    x = most_disliked_title,

    y = most_disliked_l_number,

    name='Number of likes',

    marker=dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    )

)

trace2 = go.Bar(

    x = most_disliked_title,

    y = most_disliked_dl_number,

    name='Number of dislikes',

    marker=dict(

        color='rgba(219, 64, 82, 0.7)',

        line=dict(

            color='rgba(219, 64, 82, 1.0)',

            width=2,

        )

    )

)



trace3 = go.Scatter(

    x = most_disliked_title,

    y = most_disliked_dl_percentage,

    name='Dislike percentage',

    mode = 'markers',

    marker=dict(

        symbol="hexagon-dot",

        size=15

    ),

    yaxis='y2'

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='group',

    title = 'No such thing as bad press, right?',

    width=900,

    height=500,

    margin=go.Margin(

        l=75,

        r=75,

        b=120,

        t=80,

        pad=10

    ),

    paper_bgcolor='rgb(244, 238, 225)',

    plot_bgcolor='rgb(244, 238, 225)',

    yaxis = dict(

        title= 'Number of likes/dislikes',

        anchor = 'x',

        rangemode='tozero'

    ),   

    yaxis2=dict(

        title='Dislike percentage',

        titlefont=dict(

            color='rgb(148, 103, 189)'

        ),

        tickfont=dict(

            color='rgb(148, 103, 189)'

        ),

        overlaying='y',

        side='right',

        anchor = 'x',

        rangemode = 'tozero',

        dtick = 0.165

    ),

    legend=dict(x=0.75, y=1)

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import urllib

import requests

import numpy as np

import matplotlib.pyplot as plt





mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png', stream=True).raw))



# This function takes in your text and your mask and generates a wordcloud. 

def generate_wordcloud(mask):

    word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=STOPWORDS, mask=mask).generate(str(in_videos["tags"]))

    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')

    plt.imshow(word_cloud)

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()

    

#Run the following to generate your wordcloud

generate_wordcloud(mask)
in_videos_first['likes_log'] = np.log(in_videos['likes'] + 1)

in_videos_first['dislikes_log'] = np.log(in_videos['dislikes'] + 1)

hist_data = [in_videos_first["dislikes_log"].values, in_videos_first["likes_log"].values]



group_labels = ['Dislikes log distribution', 'Likes log distribution']

colors = ['#A6ACEC', '#63F5EF']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, colors=colors,

                         bin_size=0.5, show_rug=False)



# Add title

fig['layout'].update(title='Likes vs dislikes', legend=dict(x=0.65, y=0.8))



# Plot!

py.iplot(fig, filename='Hist and Curve')