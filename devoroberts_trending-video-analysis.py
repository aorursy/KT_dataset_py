import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

from datetime import datetime

import dateutil.parser

import seaborn as sns 

sns.set_style('whitegrid')

init_notebook_mode(connected=True)

df = pd.read_csv("../input/youtube-new/USvideos.csv")

# Convert timestamp strings to datetime object.

def convert_date(timestamp, date_format='%Y-%m-%d'):

    '''

    Converts timestamp to datetime object with specified format.

    default format: '%Y-%m-%d'

    '''

    if 'Z' not in timestamp: # trending_date has format 'yy.dd.mm'

        parsed_date = datetime.strptime(timestamp, '%y.%d.%m')

        return parsed_date

    else:               # publish_time has format 'yyyy-mm-ddThh:mm:ss.000Z'

        d = dateutil.parser.parse(timestamp)

        parsed_date = d.strftime('%Y-%m-%d')

        return d.strptime(parsed_date, date_format)





# Map category labels, add category_label column.

def map_categories(df, map_dict, column='category_label'):

    '''

    Adds category label column to dataframe.

    Accepts df: Dataframe to perform map on.

            column: Column name in df to perform map on. (default='category_label')

            map_dict: Dictionary containing map items. (default=category_dict)

    '''

    df[column] = df.category_id.map(map_dict)

    return df



# Find top youTube video producers for specified year.

def top_video_producing_for_yr(df, year, top_range=5):

    '''

    Finds top video producers for specified year.

    Accepts df: Dataframe.

            year: Year to filter by.

            top_range: Number of top entries to print / return. (default=5)

    '''



    year_filter = [date.year == year for date in df['publish_time']]

    sliced_df = df[year_filter]



    channel_vid_groups = sliced_df.groupby(['channel_title'])['video_id'].count()

    sorted_groups = channel_vid_groups.sort_values(ascending=False)

    top_producers = sorted_groups[:top_range]



    print('#'*30)

    print(f'Top {top_range} video producers in {year} were:')

    print()

    i = 0

    for vid_count in top_producers:

        print('\t', f'{str(i+1)}) {top_producers.index[i]} : {vid_count} videos.', end='')

        print('\n')

        i += 1



    print('#'*30)



    return top_producers





# Plotly Bar graph 

def plotly_bar(x, y, name, title, x_title, y_title, filename, colors=None):

    trace1 = go.Bar(x=x, 

                    y=y,

                    name=name,

                    marker={'color':colors})

    layout = go.Layout(title=title, 

                        xaxis={'title':x_title},

                        yaxis={'title':y_title})



    data = [trace1]

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=filename)



# Plotly Bar graph for year

def plot_category_bar_for_yr(df, year, colors=None):

    year_filter = [date.year == year for date in df['publish_time']]

    sliced_df = df[year_filter]



    category_counts_for_yr = sliced_df.category_label.value_counts()

    print(category_counts_for_yr)



    x = category_counts_for_yr.index

    y = category_counts_for_yr

    name = f'Category Bar For {year}'

    title = f'Category Count ({year})'

    x_title = 'Category'

    y_title = 'Count'

    filename = f'category_bar_{year}'



    plotly_bar(x=x, y=y, name=name, title=title, x_title=x_title, y_title=y_title, filename=filename, colors=colors)



date_columns = ['trending_date', 'publish_time']



for col in date_columns:

    df[col] = [convert_date(ts) for ts in df[col]]

# Take a peak at the first 5 rows (head).

df.head()

# Take a peak at the dataframes info function.

df.info()



# 40,949 total entries.

# Every column looks complete with the exception of the video descriptions.

# Take a peak at the dataframes describe function.

df.describe()

# Let add category labels to the dataframe.

category_dict = {

    

    1 :  'Film & Animation',

    2 : 'Autos & Vehicles',

    10 : 'Music',

    15 : 'Pets & Animals',

    17 : 'Sports',

    18 : 'Short Movies',

    19 : 'Travel & Events',

    20 : 'Gaming',

    21 : 'Videoblogging',

    22 : 'People & Blogs',

    23 : 'Comedy',

    24 : 'Entertainment',

    25 : 'News & Politics',

    26 : 'Howto & Style',

    27 : 'Education',

    28 : 'Science & Technology',

    29 : 'Nonprofits & Activism',

    30 : 'Movies',

    31 : 'Anime/Animation',

    32 : 'Action/Adventure',

    33 : 'Classics',

    34 : 'Comedy',

    35 : 'Documentary',

    36 : 'Drama',

    37 : 'Family',

    38 : 'Foreign',

    39 : 'Horror',

    40 : 'Sci-Fi/Fantasy',

    41 : 'Thriller',

    42 : 'Shorts',

    43 : 'Shows',

    44 : 'Trailers',

}



# Store new dataframe in a new variable.

mapped_df = map_categories(df, map_dict=category_dict)

year_list = [int(date.year) for date in mapped_df['publish_time']]

mapped_df['year'] = year_list



# Rearrange columns so that category_id 

# and category_label are next to one another.

mapped_df = mapped_df[['video_id', 'trending_date', 'title', 'channel_title', 'category_id',

                    'category_label', 'publish_time', 'tags', 'views', 'likes', 'dislikes', 

                    'comment_count', 'thumbnail_link', 'comments_disabled', 'ratings_disabled',

                    'video_error_or_removed', 'description', 'year']]



category_counts = mapped_df.category_label.value_counts()

category_counts



category_list = category_counts.index

cat_colors = ['#016e29', '#7f2171', '#b241d0', '#decad2', '#54d7a1', '#10e481', '#2ec580', '#4fa15c',

                '#94d85c', '#df5aad', '#6dd279', '#91a9ed', '#bb232b', '#b6d41b', '#359fe1', '#4985fd']



cat_colors_dict = {cat:color for cat, color in zip(category_list, cat_colors)}

i = 0

for count in category_counts:

    print(f'Of the {mapped_df.shape[0]} videos uploaded, {count} ({(count/mapped_df.shape[0]) * 100:0.2f}%) videos were of the {category_counts.index[i]} category.')

    print()

    i += 1

# PLot bar chart for all years.

x = category_counts.index

y = category_counts

name = 'Category Bar'

colors = cat_colors

title = 'Category Count (2006 - 2018)'

x_title = 'Category'

y_title = 'Count'

filename = 'category_bar'

plotly_bar(x, y, name, title, x_title, y_title, filename, colors=colors)



# PLot bar chart for 2018.

plot_category_bar_for_yr(mapped_df, 2018, colors=cat_colors)

# PLot bar chart for 2018.

plot_category_bar_for_yr(mapped_df, 2017, colors=cat_colors)

# Table of year and category labels. Here we see the general number of videos

# (That are tracked) uploaded since 2006.

grouped_df = mapped_df.groupby(['category_label', 'year'])['video_id'].count()

unstacked_df = grouped_df.unstack().fillna(value=0)

unstacked_df



channel_groups = mapped_df.groupby(['channel_title'])['video_id'].count().sort_values(ascending=False)

channel_groups[:30]

end = 30

name = f'Top {end}'

title = f'Top  {end} Video Producing Channels'

x_title = 'Channels'

y_title = 'Count'

filename = f'top_ {end}_producing'



plotly_bar(x=channel_groups.index[:end], 

            y=channel_groups[:end], 

            name=name, 

            title=title, 

            x_title=x_title, 

            y_title=y_title, 

            filename=filename)
# Top category for ESPN channel.

top_channel_df = mapped_df[mapped_df['channel_title'] == 'ESPN']

top_channel_df.groupby(['category_label'])['video_id'].count().sort_values(ascending=False)

# Top category for The Tonight Show.

channel = 'The Tonight Show Starring Jimmy Fallon'

top_channel_df = mapped_df[mapped_df['channel_title'] == channel]

top_channel_df.groupby(['category_label'])['video_id'].count().sort_values(ascending=False)

# Report out categories and the amount of view accrued from 2006 - 2018.

view_counts = mapped_df.groupby(['category_label'])['views'].sum().sort_values(ascending=False)



i = 0

for count in view_counts:

    print(f'{view_counts.index[i]} has accrued {count} views all-time (2006-2018).')

    print()

    i += 1



print('View Count Series')

print(view_counts)

# Report out categories and the amount of view accrued in 2017.

df_2017 = mapped_df[mapped_df['year'] == 2017]

view_counts_2017 = df_2017.groupby(['category_label'])['views'].sum().sort_values(ascending=False)



i = 0

for count in view_counts_2017:

    print(f'{view_counts_2017.index[i]} has accrued {count} views in 2017.')

    print()

    i += 1



print()

print('View Count Series 2017', '#'*10)

print(view_counts_2017)

print('#'*35)

# Report out categories and the amount of view accrued in 2018.

df_2018 = mapped_df[mapped_df['year'] == 2018]

view_counts_2018 = df_2018.groupby(['category_label'])['views'].sum().sort_values(ascending=False)



i = 0

for count in view_counts_2018:

    print(f'{view_counts_2018.index[i]} has accrued {count} views in 2018.')

    print()

    i += 1



print()

print('View Count Series 2018', '#'*10)

print(view_counts_2018)

print('#'*35)




