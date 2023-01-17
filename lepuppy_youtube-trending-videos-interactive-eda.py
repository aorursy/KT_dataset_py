# Essentials

import numpy as np

import pandas as pd



# Visualisation

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import seaborn as sns

import matplotlib.pyplot as plt



# Others

from wordcloud import WordCloud
# Let's start with the trending French videos

fr_videos_raw = pd.read_csv('../input/youtube-new/FRvideos.csv', sep=',')
fr_videos_raw.head()
fr_category_id = pd.read_json('../input/youtube-new/FR_category_id.json')
fr_category_id.head()
fr_category_id['items'].iloc[0]
# We retreive category_id and category_title in two lists (with same order) contained in a dict

fr_category_id_dict = {'category_id':[key for key in fr_category_id['items'].keys()],

                       'category_title':[y['snippet']['title'] for x,y in fr_category_id['items'].items()]}

fr_category_id_dict.keys(), fr_category_id_dict.values()
# Create dataframe from dict

fr_category_id_df = pd.DataFrame.from_dict(fr_category_id_dict)



# Merge on category_id then drop it

fr_videos = fr_videos_raw.merge(fr_category_id_df, how='inner', on='category_id').drop(columns='category_id')

fr_videos.loc[:5, ['title', 'category_title']]
# Dataset dimensions

fr_videos.shape
# Missing values by column

fr_videos.isna().sum()
# Renaming columns for cleaner code

fr_videos = fr_videos.rename(columns={'category_title':'category'})

px.histogram(fr_videos, x='category', title='Number of videos per category').update_xaxes(categoryorder='total descending')
# Trending_date  & publish_time

fr_videos.loc[:5, ['video_id', 'trending_date', 'publish_time']]
# Converting series to datetime series

fr_videos['trending_date'] = pd.to_datetime(fr_videos['trending_date'], format='%y.%d.%m')

fr_videos['publish_time'] = pd.to_datetime(fr_videos['publish_time'], format='%Y-%m-%d')



# Adding a time to trending_date in order to compare with publish_time

# Input last minute of day in order to avoid negative differences

fr_videos['trending_date'] = pd.to_datetime(fr_videos['trending_date'].astype(str) + ' ' + pd.Series(['23:59:59+00:00']*fr_videos.shape[0]),

                                            format='%Y-%m-%d %H:%M:%S')



# Create new feature trending_time in seconds

fr_videos['trending_time'] = pd.to_timedelta(fr_videos['trending_date'] - fr_videos['publish_time']).apply(lambda x: int(x.total_seconds()))



# Assert there's no negative time

try:

    if (fr_videos['trending_time'] < 0).any():

        raise ValueError

except ValueError:

    print("Negative timedelta found ! You should have a look.")
# I first used px.histogram but the data was so spread again it didn't help

# Even a boxplot is stretched too much to have a good overview

# We plot in hours

(fr_videos['trending_time']//3600).describe()
# More precision

for quantile, trd_time in fr_videos['trending_time'].quantile([0.80, 0.85, 0.90, 0.95, 0.97, 0.99]).iteritems():

    print("{}% of videos become trending in less than {} hours".format(int(quantile*100), int(trd_time//(3600))))
# Renaming columns for cleaner code

fr_videos = fr_videos.rename(columns={'comment_count':'comments'})



fig = px.scatter_matrix(fr_videos, dimensions=['views', 'likes', 'dislikes', 'comments'])

# You can add diagonal_visible=False as argument in update_traces if you want to skip the diagonal

fig.update_traces(opacity=0.3, showupperhalf=False)

fig.show()
px.histogram(fr_videos, x='comments_disabled', facet_col='video_error_or_removed', color='ratings_disabled')
any_disabled = pd.Series([True if any([com, rat, err]) else False for com, rat, err in zip(fr_videos['comments_disabled'],

                                         fr_videos['ratings_disabled'],

                                         fr_videos['video_error_or_removed'])])
# Let's quickly check if any_disabled did the trick:

try:

    assert fr_videos[(fr_videos['comments_disabled'] == False) & 

          (fr_videos['ratings_disabled'] == False) & 

          (fr_videos['video_error_or_removed'] == False)].shape[0] == (any_disabled == False).sum()

    fr_videos['any_disabled'] = any_disabled

except AssertionError:

    print("any_disabled was not successfully computed !")
fr_videos['description'].head()
fr_videos['description'].isna().sum()
# Count length of the description

fr_videos['description_length'] = fr_videos['description'].str.len()



# Input 0 for missing values and convert series to integer type

fr_videos['description_length'] = fr_videos['description_length'].fillna(0).astype(int)
fr_videos['tags'].head()
# Lower case tags, remove "" then retreive each tag separated from '|'

# It's delicate to work with accents & encoding because some characters might be erased e.g. arabic characters

split_tags = fr_videos['tags'].str.replace('"', '').str.lower().str.split('|')

split_tags.head()
# Second row contains [[none]] which is weird: is it a tag itself or an error ? Let's find out

split_tags.iloc[1]
# First check if there are empty lists

print(split_tags.apply(lambda l: len(l) == 0).sum())



# Check if there are videos with 'none' as tag

matchers = ['none','None', 'NONE']

# This retreive matchers only

# Convert to tuple temporarily because using value_counts() on lists objects raise error with pandas 0.25.0

nones = split_tags.apply(lambda l: tuple(s for s in l if any(xs in s for xs in matchers)))

nones.value_counts()
# We don't want to remove tags containing 'none' but the [[none]]

split_tags.apply(lambda l: l == ['[none]']).sum()
split_tags_cleaned = split_tags.apply(lambda l: np.nan if l == ['[none]'] else l)



# Input number of tags in the list and 0 if there's none (NaN)

fr_videos['tags_count'] = split_tags_cleaned.apply(lambda x: int(len(x)) if type(x) == list else 0)
# I'm not sure what to do with all these tags. I guess the order may be important therefore I'll add 5 features for the first 5 tags



def input_n_tag(tags, n):

    try:

        n_tag = tags[n]

    # When dealing with NaN

    except TypeError:

        n_tag = 'notag'

    # When list too short

    except IndexError:

        n_tag = 'notag' 

    return n_tag 

    

fr_videos['tag1'] = split_tags_cleaned.apply(lambda l: input_n_tag(l, 0))

fr_videos['tag2'] = split_tags_cleaned.apply(lambda l: input_n_tag(l, 1))

fr_videos['tag3'] = split_tags_cleaned.apply(lambda l: input_n_tag(l, 2))

fr_videos['tag4'] = split_tags_cleaned.apply(lambda l: input_n_tag(l, 3))

fr_videos['tag5'] = split_tags_cleaned.apply(lambda l: input_n_tag(l, 4))
fr_videos.loc[:5, ['title', 'tags_count', 'tag1', 'tag2', 'tag5']]
# Adding all tags in a single list

all_tags = split_tags_cleaned.explode().astype(str)

text = ', '.join(all_tags)



# Create wordcloud from single string

wordcloud = WordCloud().generate(text)

wordcloud = WordCloud(background_color="white", max_words=1000, max_font_size=40, relative_scaling=.5).generate(text)



plt.figure(figsize=(14, 10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
fr_videos['title_length'] = fr_videos['title'].str.len()
fr_videos.loc[:5, ['title', 'title_length']]
print("Number of videos: {} for {} different channels.".format(fr_videos.shape[0], len(fr_videos['channel_title'].unique())))
fr_videos['channel_title'] = fr_videos['channel_title'].str.lower()
corr = fr_videos.loc[:, ['views', 'likes', 'dislikes', 'comments', 'trending_time', 'tags_count', 'description_length', 'title_length']].corr()
fig2 = go.Figure(data=go.Heatmap(

        z=corr.values,

        x=corr.index,

        y=corr.index,

        colorscale="Earth",

        zmin=-1,

        zmax=1

    # negative values

))

fig2.update_layout(title='Correlations all Categories combined')

fig2.show()
# Prepare correlation dataframes for heatmaps

categories = fr_videos['category'].unique()

interactions_corr_list = [fr_videos[fr_videos['category'] == cat].loc[:, ['views', 'likes', 'dislikes', 'comments', 'trending_time', 'tags_count',

                                                                          'description_length', 'title_length']].corr() for cat in categories]



#Initialize figure

fig3 = go.Figure()



# Add each heatmap, let the first one visible only to avoid traces stacked

for idx, corr in enumerate(interactions_corr_list):

    if idx==0:

        fig3.add_trace(

            go.Heatmap(

                z=corr.values,

                x=corr.index,

                y=corr.index,

                colorscale="Earth",

                zmin=-1,

                zmax=1,

                visible=True))

    else:

         fig3.add_trace(

            go.Heatmap(

                z=corr.values,

                x=corr.index,

                y=corr.index,

                colorscale="Earth",

                zmin=-1,

                zmax=1,

                visible=False)) 



# Add buttons

fig3.update_layout(

    updatemenus=[

        go.layout.Updatemenu(

            active=0,

            x=0.8,

            y=1.2,

            buttons=list([

                dict(label=cat,

                     method="update",

                     # This comprehension list let visible the current trace only by setting itself to True and others to False

                     args=[{"visible": [False if sub_idx != idx else True for sub_idx, sub_cat in enumerate(categories)]},

                           {"title": "Correlation heatmap for category: " + cat}])

                for idx, cat in enumerate(categories)

            ] ) 

        )

    ])