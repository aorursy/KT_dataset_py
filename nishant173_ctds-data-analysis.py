# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings(action='ignore')

import os
from os.path import isfile, join

import re
import string
import nltk
from textblob import TextBlob

import random
import numpy as np
import pandas as pd

from wordcloud import STOPWORDS as stopwords_wordcloud
from wordcloud import WordCloud
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
%%time

df_anchor_thumbnails = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')
df_description = pd.read_csv("../input/chai-time-data-science/Description.csv")
df_episodes = pd.read_csv("../input/chai-time-data-science/Episodes.csv", parse_dates=['recording_date', 'release_date'])
df_youtube_thumbnails = pd.read_csv("../input/chai-time-data-science/YouTube Thumbnail Types.csv")
print(f"Shape of df_anchor_thumbnails:\t{df_anchor_thumbnails.shape}")
print(f"Shape of df_description:\t{df_description.shape}")
print(f"Shape of df_episodes:\t\t{df_episodes.shape}")
print(f"Shape of df_youtube_thumbnails:\t{df_youtube_thumbnails.shape}")
def extract_timestamp_info(timestamp, get_hours=False, get_mins=False, get_secs=False):
    """ Extract timestamp data """
    timestamp = str(timestamp)
    if len(timestamp) <= 5:
        hours = 0
        mins, secs = timestamp.split(':')
    elif len(timestamp) == 7:
        hours, mins, secs = timestamp.split(':')
    if get_hours:
        return int(hours)
    if get_mins:
        return int(mins)
    if get_secs:
        return int(secs)
    return None


def get_timestamp_in_secs(timestamp):
    """
    Convert timestamp string into integer indicating number of seconds denoting said timestamp.
    Eg: If `timestamp` is '1:12:52', then `timestamp_in_secs` will be (1 x 3600) + (12 * 60) + (52)
    """
    timestamp = str(timestamp)
    timestamp_split = timestamp.split(':')
    if len(timestamp_split) == 2:
        hours = 0
        mins, secs = timestamp_split
    elif len(timestamp_split) == 3:
        hours, mins, secs = timestamp_split
    timestamp_in_secs = (int(hours) * 3600) + (int(mins) * 60) + int(secs)
    return timestamp_in_secs


def read_subtitle_data(filepath="../input/chai-time-data-science/Cleaned Subtitles/"):
    """
    Reads the cleaned subtitle files into one Pandas DataFrame, and adds some transformations to it.
    NOTE: Make sure you leave a trailing slash '/' for the filepath.
    """
    df_subtitles_data = pd.DataFrame()
    filenames = [filename for filename in os.listdir(filepath) if isfile(join(filepath, filename))]
    counter = 0
    for filename in filenames:
        episode_number = int(filename.split('.')[0][1:])
        df_temp = pd.read_csv(f"{filepath}{filename}")
        df_temp['Episode'] = episode_number
        df_temp['NumSpeakersForThisEpisode'] = df_temp['Speaker'].nunique()
        df_subtitles_data = pd.concat(objs=[df_subtitles_data, df_temp], ignore_index=True, sort=False)
        counter += 1
    
    # Sort by timestamps inside each episode
    df_subtitles_data['Hours'] = df_subtitles_data['Time'].apply(extract_timestamp_info, get_hours=True)
    df_subtitles_data['Mins'] = df_subtitles_data['Time'].apply(extract_timestamp_info, get_mins=True)
    df_subtitles_data['Secs'] = df_subtitles_data['Time'].apply(extract_timestamp_info, get_secs=True)
    
    df_subtitles_data.sort_values(by=['Episode', 'Hours', 'Mins', 'Secs'], ascending=[True, True, True, True], inplace=True)
    df_subtitles_data = df_subtitles_data.reset_index(drop=True)
    df_subtitles_data.drop(labels=['Hours', 'Mins', 'Secs'], axis=1, inplace=True)
    
    # Add TimestampInSecs (Convert string timestamp into interger denoting the number of seconds passed)
    df_subtitles_data['TimestampInSecs'] = df_subtitles_data['Time'].apply(get_timestamp_in_secs)
    
    # Add TalkTimeInSecs (Talk time of each person in the conversation, by episode)
    df_subtitles_data_altered = pd.DataFrame()
    episodes = df_subtitles_data['Episode'].unique().tolist()
    counter_talktime = 0
    for episode in episodes:
        df_temp = df_subtitles_data[df_subtitles_data['Episode'] == episode]
        df_temp['TalkTimeInSecs'] = (df_temp.iloc[::-1]['TimestampInSecs'].diff()).apply(round, args=[4]).abs().tolist()[::-1]
        df_subtitles_data_altered = pd.concat(objs=[df_subtitles_data_altered, df_temp], ignore_index=True, sort=False)
        counter_talktime += 1
    
    print(
        f"Successfully read {counter} files containing subtitles, and concatenated into one DataFrame.",
        f"Successfully added 'TalkTimeInSecs' column for {counter_talktime} episodes."
    )
    return df_subtitles_data_altered
%%time
df_subtitles = read_subtitle_data()
df_subtitles.info()
df_subtitles.head()
df_anchor_thumbnails
df_youtube_thumbnails
df_description.info()
df_description.head()
df_episodes.info()
df_episodes.head()
def generate_random_hex_code():
    """ Generates random 6-digit hexadecimal code """
    choices = '0123456789ABCDEF'
    random_hex_code = '#'
    for _ in range(6):
        random_hex_code += random.choice(choices)
    return random_hex_code


def plot_donut(title, labels, values):
    """
    Plots donut chart.
    Parameters:
        - title (str): Title of chart
        - labels (list): Categorical labels
        - values (list): Numerical values
    """
    colors = [generate_random_hex_code() for _ in range(len(labels))]
    _, ax1 = plt.subplots()
    ax1.pie(x=list(values), labels=list(labels), colors=colors, autopct='%1.1f%%', startangle=40)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')  
    plt.tight_layout()
    plt.title(f"{str(title)}", fontsize=15)
    plt.show()
    return None


def plot_bar_sns(title, x_column, y_column, data, orient, hue=None):
    """
    Plots vertical/horizontal bar chart using Seaborn.
    Parameters:
        - title (str): Title of chart
        - x_column (str): Name of column representing data for x-axis
        - y_column (str): Name of column representing data for y-axis
        - data (Pandas DataFrame): Dataset
        - orient (str): 'v' | 'h'
        - hue (str): Name of column by which you want to color hue (optional)
    """
    plt.figure(figsize=(12, 5))
    sns.barplot(data=data,
                x=x_column,
                y=y_column,
                orient=orient,
                hue=hue)
    plt.title(f"{str(title)}", fontsize=24)
    plt.xlabel(x_column, fontsize=16)
    plt.ylabel(y_column, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return None


def plot_bar(title, x_column, y_column, data=None, x_vals=None, y_vals=None, horizontal=False, annotate=True):
    """
    Plots bar chart or horizontal bar chart.
    Parameters:
        - title (str): Title of chart
        - x_column (str): Name of column representing data for x-axis
        - y_column (str): Name of column representing data for y-axis
        - data (Pandas DataFrame): Dataset (optional)
        - x_vals (list): Values/Labels for the x-axis (optional)
        - y_vals (list): Values/Labels for the y-axis (optional)
        - horizontal (bool): True for horizontal barchart; False for vertical barchart. Default: False
        - annotate (bool): True to use annotations; False otherwise. Default: True
    Usage:
        This function can be called by either of the following 2 ways.
        - By passing `data`, along with `x_column` and `y_column`
        - By passing `x_vals` and `y_vals`, along with `x_column` and `y_column`
        - Setting `horizontal` accordingly
    """
    plt.figure(figsize=(12, 5))
    if type(data) == pd.core.frame.DataFrame:
        colors = [generate_random_hex_code() for _ in range(len(data))]
        if horizontal:
            plt.barh(y=data[y_column], width=data[x_column], color=colors)
        else:
            plt.bar(x=data[x_column], height=data[y_column], color=colors)
        # Annotations
        if type(data[x_column].iloc[0]) == str:
            numerical_column = y_column
        else:
            numerical_column = x_column
        if annotate:
            for i, v in enumerate(data[numerical_column]):
                plt.text(x = v + 0.04, y = i, s = str(v), fontweight='bold', fontsize=12, color='black')
    else:
        x_vals, y_vals = list(x_vals), list(y_vals)
        colors = [generate_random_hex_code() for _ in range(len(x_vals))]
        if horizontal:
            plt.barh(y=y_vals, width=x_vals, color=colors)
        else:
            plt.bar(x=x_vals, height=y_vals, color=colors)
        # Annotations
        if type(x_vals[0]) == str:
            numerical_column = y_vals
        else:
            numerical_column = x_vals
        if annotate:
            for i, v in enumerate(numerical_column):
                plt.text(x = v + 0.04, y = i, s = str(v), fontweight='bold', fontsize=12, color='black')
    plt.title(f"{str(title)}", fontsize=24)
    plt.xlabel(x_column, fontsize=16)
    plt.ylabel(y_column, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return None


def plot_valuecounts(title, data, column, normalize=False):
    """
    Plots value-counts.
    Parameters:
        - title (str): Title of chart
        - data (Pandas DataFrame): Dataset
        - column (str): Name of column for which you want to plot value-counts
        - normalize (bool): True for percentages; False for absolutes. Default: False
    """
    if normalize:
        dictionary_valuecounts = data[column].value_counts(normalize=True).mul(100).apply(round, args=[2]).to_dict()
        x_column = 'Percentage of occurences'
    else:
        dictionary_valuecounts = data[column].value_counts().to_dict()
        x_column = 'Count of occurences'
    plot_bar(title=title,
             x_column=x_column,
             y_column=column,
             x_vals=dictionary_valuecounts.values(),
             y_vals=dictionary_valuecounts.keys(),
             horizontal=True)
    return None


def plot_timeseries(title, x_column, y_column, data=None, x_vals=None, y_vals=None, area=False):
    """
    Plots timeseries or timeseries-like line/area chart, given ascendingly sorted DataFrame and/or list-like data.
    Parameters:
        - title (str): Title of chart
        - x_column (str): Name of column representing data for x-axis, usually the element of time
        - y_column (str): Name of column representing data for y-axis
        - data (Pandas DataFrame): Dataset (optional)
        - x_vals (list): Values/Labels for the x-axis, usually the element of time (optional)
        - y_vals (list): Values/Labels for the y-axis (optional)
        - area (bool): True for area-chart; False for line-chart. Default: False
    Usage:
        This function can be called by either of the following 2 ways.
        - By passing `data`, along with `x_column` and `y_column`
        - By passing `x_vals` and `y_vals`, along with `x_column` and `y_column`
        - Setting `area` accordingly
    """
    color = '#135FB6'
    linewidth = 2.5
    if area:
        fill_color = 'skyblue'
        fill_alpha = 0.4
    
    plt.figure(figsize=(12, 5))
    if type(data) == pd.core.frame.DataFrame:
        if area:
            plt.fill_between(data[x_column], data[y_column], color=fill_color, alpha=fill_alpha)
        plt.plot(data[x_column], data[y_column], color=color, linewidth=linewidth)
        # Set 'n' equidistant points for xticks, instead of all points
        xticks = list(map(np.floor, list(np.linspace(start=1, stop=len(data), num=5))))
    else:
        if area:
            plt.fill_between(list(x_vals), list(y_vals), color=fill_color, alpha=fill_alpha)
        plt.plot(list(x_vals), list(y_vals), color=color, linewidth=linewidth)
        # Set 'n' equidistant points for xticks, instead of all points
        xticks = list(map(np.floor, list(np.linspace(start=1, stop=len(x_vals), num=5))))
    plt.title(f"{str(title)}", fontsize=24)
    plt.xlabel(x_column, fontsize=16)
    plt.ylabel(y_column, fontsize=16)
    plt.xticks(xticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()
    return None


def plot_scatter(title, x_column, y_column, data=None, x_vals=None, y_vals=None, alpha=None, size=80):
    """
    Plots scatter chart, given DataFrame and/or list-like data.
    Parameters:
        - title (str): Title of chart
        - x_column (str): Name of column representing data for x-axis, usually the element of time
        - y_column (str): Name of column representing data for y-axis
        - data (Pandas DataFrame): Dataset (optional)
        - x_vals (list): Values/Labels for the x-axis, usually the element of time (optional)
        - y_vals (list): Values/Labels for the y-axis (optional)
        - alpha (float): Values between 0 and 1 for transparency (optional)
        - size (int): Size of scatter points. Default: 80
    Usage:
        This function can be called by either of the following 2 ways.
        - By passing `data`, along with `x_column` and `y_column`
        - By passing `x_vals` and `y_vals`, along with `x_column` and `y_column`
    """
    color = 'g'
    
    plt.figure(figsize=(12, 5))
    if type(data) == pd.core.frame.DataFrame:
        plt.scatter(x=data[x_column], y=data[y_column], c=color, s=size, alpha=alpha)
    else:
        plt.scatter(x=list(x_vals), y=list(y_vals), c=color, s=size, alpha=alpha)
    plt.title(f"{str(title)}", fontsize=24)
    plt.xlabel(x_column, fontsize=16)
    plt.ylabel(y_column, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()
    return None


def plot_histogram(title, column, data=None, array=None, bins=10, grid=True, color='#2B88CA'):
    """
    Plots histogram, given DataFrame and/or list-like data.
    Parameters:
        - title (str): Title of chart
        - column (str): Name of column representing data for the histogram
        - data (Pandas DataFrame): Dataset (optional)
        - array (list): Values for the histogram to plot (optional)
        - bins (int): Number of bins for the histogram. Default: 10
        - grid (bool): True to plot grid, False otherwise. Default: True
        - color (str): Color of histogram bins. Hex-codes can be used. Default: '#2B88CA' (shade of blue)
    Usage:
        This function can be called by either of the following 2 ways.
        - By passing `data`, along with `column`
        - By passing `array`, along with `column`
    """
    plt.figure(figsize=(12, 5))
    if type(data) == pd.core.frame.DataFrame:
        plt.hist(x=data[column], bins=bins, color=color)
    else:
        plt.hist(x=array, bins=bins, color=color)
    plt.title(f"{str(title)}", fontsize=24)
    plt.xlabel(column, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if grid:
        plt.grid()
    plt.tight_layout()
    plt.show()
    return None
def clean_text(text):
    """
    Make text lowercase, remove text in square brackets, remove links, remove punctuations
    and remove words containing numbers.
    """
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# def preprocess_text(text):
#     """
#     Cleaning and parsing the text.
#     """
#     tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#     text_cleaned = clean_text(text)
#     tokenized_text = tokenizer.tokenize(text_cleaned)
#     # remove_stopwords = [word for word in tokenized_text if word not in stopwords.words('english')]
#     combined_text = ' '.join(tokenized_text)
#     return combined_text
def describe_missing_data(data, show_all=False):
    """
    Definition:
        Takes in raw DataFrame, and returns DataFrame with information about missing values (by variable)
    Parameters:
        - data (Pandas DataFrame): Pandas DataFrame of dataset
        - show_all (bool): True to show variables without missing values; False otherwise. Default: False
    Returns:
        Returns Pandas DataFrame with information about missing values (by variable).
        Columns include: ['Variable', 'NumMissingValues', 'PercentMissingValues', 'DataType']
    """
    # Datatype info
    df_dtypes = pd.DataFrame()
    data.dtypes.to_frame().reset_index()
    df_dtypes['Variable'] = data.dtypes.to_frame().reset_index()['index']
    df_dtypes['DataType'] = data.dtypes.to_frame().reset_index()[0]
    
    # Missing value info
    rename_dict = {
        'index': 'Variable',
        0: 'NumMissingValues',
        '0': 'NumMissingValues'
    }
    df_missing_values = data.isnull().sum().to_frame().reset_index()
    df_missing_values.rename(mapper=rename_dict, axis=1, inplace=True)
    df_missing_values.sort_values(by='NumMissingValues', ascending=False, inplace=True)
    df_missing_values = df_missing_values[df_missing_values['NumMissingValues'] > 0].reset_index(drop=True)
    percent_missing_values = (df_missing_values['NumMissingValues'] / len(data)).mul(100).apply(round, args=[3])
    df_missing_values['PercentMissingValues'] = percent_missing_values
    
    # Merge everything
    df_description = pd.merge(left=df_missing_values, right=df_dtypes, on='Variable', how='outer')
    df_description.fillna(value=0, inplace=True)
    if not show_all:
        df_description = df_description[df_description['NumMissingValues'] > 0]
    df_description['NumMissingValues'] = df_description['NumMissingValues'].astype(int)
    return df_description


def get_episode_number(episode_id):
    return int(str(episode_id)[1:])


# def is_podcast(episode_id):
#     return (str(episode_id).strip()[0].upper() == 'E')


# def split_data_by_episode_type(data):
#     """
#     Definition:
#         There are 2 types of episodes: Podcasts and Educational.
#         The `episode_id` of Podcasts are prefixed by 'E'.
#         The `episode_id` of Educational videos are prefixed by 'M'.
#         This function takes in the episodes DataFrame, splits by the two episode types
#         (Podcasts and Educational videos), and returns dictionary of split data.
#     """
#     data['is_podcast'] = data['episode_id'].apply(is_podcast)
#     data_podcast = data[data['is_podcast'] == True].reset_index(drop=True)
#     data_educational_video = data[data['is_podcast'] == False].reset_index(drop=True)
#     dictionary_data_by_episode_type = {
#         'podcast': data_podcast,
#         'educational': data_educational_video
#     }
#     return dictionary_data_by_episode_type
plot_bar(title="Percentage of missing values by variable",
         x_column='PercentMissingValues',
         y_column='Variable',
         data=describe_missing_data(data=df_episodes),
         horizontal=True)

plot_bar(title="Number of missing values by variable",
         x_column='NumMissingValues',
         y_column='Variable',
         data=describe_missing_data(data=df_episodes),
         horizontal=True)
df_episodes.head()
df_episodes_with_heroes = df_episodes[df_episodes['heroes'].notna()]
df_episodes_without_heroes = df_episodes[df_episodes['heroes'].isna()]
df_episodes_with_heroes['category'].value_counts()
df_episodes_without_heroes['category'].value_counts()
df_episodes_with_heroes['episode_number'] = df_episodes_with_heroes['episode_id'].apply(get_episode_number)
df_episodes_with_heroes = df_episodes_with_heroes.sort_values(by='episode_number', ascending=True).reset_index(drop=True)

df_episodes['episode_duration_mins'] = round((df_episodes['episode_duration'] / 60), 2)
df_episodes_with_heroes['episode_duration_mins'] = round((df_episodes_with_heroes['episode_duration'] / 60), 2)
df_episodes_without_heroes['episode_duration_mins'] = round((df_episodes_without_heroes['episode_duration'] / 60), 2)
categories_valuecount = df_episodes_with_heroes['category'].value_counts()
plot_donut(title="Podcasts by Category (With heroes)",
           labels=categories_valuecount.index,
           values=categories_valuecount.values)

categories_valuecount = df_episodes['category'].value_counts()
plot_donut(title="Podcasts by Category (All)",
           labels=categories_valuecount.index,
           values=categories_valuecount.values)

plot_valuecounts(title="Podcasts (having heroes) by Category",
                 data=df_episodes_with_heroes,
                 column='category',
                 normalize=False)

plot_valuecounts(title="Podcasts (all) by Category",
                 data=df_episodes,
                 column='category',
                 normalize=False)
plot_bar(title="Average podcast duration (in mins) with heroes vs without heroes",
         x_column='Average podcast duration (in mins)',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['episode_duration_mins'].mean(), 2),
                 round(df_episodes_without_heroes['episode_duration_mins'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=True)

plot_bar(title="Average YouTube CTR with heroes vs without heroes",
         x_column='Average YouTube CTR',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['youtube_ctr'].mean(), 2),
                 round(df_episodes_without_heroes['youtube_ctr'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=False)

plot_bar(title="Average YouTube views with heroes vs without heroes",
         x_column='Average YouTube views',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['youtube_views'].mean(), 2),
                 round(df_episodes_without_heroes['youtube_views'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=False)

plot_bar(title="Average YouTube impressions with heroes vs without heroes",
         x_column='Average YouTube impressions',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['youtube_impressions'].mean(), 2),
                 round(df_episodes_without_heroes['youtube_impressions'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=False)

plot_bar(title="Average YouTube likes with heroes vs without heroes",
         x_column='Average YouTube likes',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['youtube_likes'].mean(), 2),
                 round(df_episodes_without_heroes['youtube_likes'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=False)

plot_bar(title="Average Spotify streams with heroes vs without heroes",
         x_column='Average Spotify streams',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['spotify_streams'].mean(), 2),
                 round(df_episodes_without_heroes['spotify_streams'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=False)

plot_bar(title="Average Spotify listeners with heroes vs without heroes",
         x_column='Average Spotify listeners',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['spotify_listeners'].mean(), 2),
                 round(df_episodes_without_heroes['spotify_listeners'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=False)

plot_bar(title="Average Apple listeners with heroes vs without heroes",
         x_column='Average Apple listeners',
         y_column='Hero present?',
         x_vals=[round(df_episodes_with_heroes['apple_listeners'].mean(), 2),
                 round(df_episodes_without_heroes['apple_listeners'].mean(), 2)],
         y_vals=['With heroes',
                 'Without heroes'],
         horizontal=True,
         annotate=False)
dict_gender_valuecount = df_episodes_with_heroes['heroes_gender'].value_counts().to_dict()

plot_donut(title='Guest appearances by Gender',
           labels=dict_gender_valuecount.keys(),
           values=dict_gender_valuecount.values())
pct_guests_working_abroad = round((df_episodes_with_heroes['heroes_nationality'] != df_episodes_with_heroes['heroes_location']).mean() * 100, 3)

plot_donut(title="Is the guest working abroad? (Not in country of origin)",
           labels=['Yes', 'No'],
           values=[pct_guests_working_abroad, 100-pct_guests_working_abroad])
dict_nationality_valuecount = df_episodes_with_heroes['heroes_nationality'].value_counts().head(10).to_dict()
plot_donut(title='Guest appearances by nationality',
           labels=dict_nationality_valuecount.keys(),
           values=dict_nationality_valuecount.values())

plot_valuecounts(title="Count of guest-appearances by nationality",
                 data=df_episodes_with_heroes,
                 column='heroes_nationality')
plot_valuecounts(title="Tea flavour consumed (With heroes)",
                 data=df_episodes_with_heroes,
                 column='flavour_of_tea')

plot_valuecounts(title="Tea flavour consumed (Without heroes)",
                 data=df_episodes_without_heroes,
                 column='flavour_of_tea')

plot_valuecounts(title="Tea flavour consumed (Overall)",
                 data=df_episodes,
                 column='flavour_of_tea')
plot_valuecounts(title="Time of recording (With heroes)",
                 data=df_episodes_with_heroes,
                 column='recording_time',
                 normalize=False)

plot_valuecounts(title="Time of recording (Without heroes)",
                 data=df_episodes_without_heroes,
                 column='recording_time',
                 normalize=False)

plot_valuecounts(title="Time of recording (Overall)",
                 data=df_episodes,
                 column='recording_time',
                 normalize=False)
plot_timeseries(title="Duration of episodes (in mins) - With heroes",
                x_column='episode_number',
                y_column='episode_duration_mins',
                data=df_episodes_with_heroes,
                area=False)

plot_timeseries(title="Duration of episodes (in mins) - All episodes",
                x_column='episode_id',
                y_column='episode_duration_mins',
                data=df_episodes.sort_values(by='release_date', ascending=True),
                area=False)
columns_of_interest = ['episode_id', 'episode_name', 'heroes', 'heroes_nationality', 'category', 'youtube_views', 'episode_duration_mins']
df_by_episode_duration = df_episodes_with_heroes[df_episodes_with_heroes['episode_duration_mins'] >= 85].sort_values(by='episode_duration_mins', ascending=False)
df_by_episode_duration.reset_index(drop=True, inplace=True)
df_by_episode_duration[columns_of_interest]
# # Inspection
# sort_by = 'apple_avg_listen_duration'

# columns_of_interest = ['episode_id', 'heroes', 'heroes_twitter_handle', 'heroes_nationality', 'category',
#                        'youtube_watch_hours', 'youtube_views', 'youtube_impressions', 'spotify_streams',
#                        'episode_duration_mins']

# if sort_by not in columns_of_interest:
#     columns_of_interest.append(sort_by)

# df_most_popular_by = df_episodes_with_heroes.sort_values(by=sort_by, ascending=False)
# df_most_popular_by[columns_of_interest].head(10)
plot_bar(title="Podcast duration by hero (in mins) - Longest",
         x_column='episode_duration_mins',
         y_column='heroes',
         data=df_episodes_with_heroes.dropna().sort_values(by='episode_duration_mins', ascending=False).head(),
         horizontal=True,
         annotate=False)

plot_bar(title="Podcast duration by hero (in mins) - Shortest",
         x_column='episode_duration_mins',
         y_column='heroes',
         data=df_episodes_with_heroes.dropna().sort_values(by='episode_duration_mins', ascending=False).tail(),
         horizontal=True,
         annotate=False)
plot_bar(title="YouTube views by hero",
         x_column='youtube_views',
         y_column='heroes',
         data=df_episodes.dropna().sort_values(by='youtube_views', ascending=False).head(),
         horizontal=True,
         annotate=True)

plot_bar(title="Spotify views by hero",
         x_column='spotify_streams',
         y_column='heroes',
         data=df_episodes.dropna().sort_values(by='spotify_streams', ascending=False).head(),
         horizontal=True,
         annotate=True)
plot_bar(title="YouTube CTR (Click-through rate) by hero",
         x_column='youtube_ctr',
         y_column='heroes',
         data=df_episodes.dropna().sort_values(by='youtube_ctr', ascending=False).head(15),
         horizontal=True,
         annotate=True)
plot_bar(title="YouTube likes by hero",
         x_column='youtube_likes',
         y_column='heroes',
         data=df_episodes.dropna().sort_values(by='youtube_likes', ascending=False).head(15),
         horizontal=True,
         annotate=True)
title = "YouTube views by hero"

plt.figure(figsize=(12, 5))
plt.scatter(data=df_episodes.dropna().sort_values(by='youtube_views', ascending=False).head(20),
            x='heroes', y='youtube_views', s='youtube_views', alpha=0.5, c='g')

plt.title(f"{str(title)}", fontsize=24)
plt.xlabel('Hero', fontsize=16)
plt.ylabel('youtube_views', fontsize=16)
plt.xticks(rotation=70, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, df_episodes.dropna()['youtube_views'].max() * 1.1)
plt.grid()
plt.show()
plot_histogram(title="Episode duration (in mins)",
               column='episode_duration_mins',
               data=df_episodes)
plot_timeseries(title="Which episodes brought in YouTube views?",
                data=df_episodes_with_heroes,
                x_column='episode_number',
                y_column='youtube_views',
                area=False)

columns_of_interest = ['episode_id', 'episode_name', 'heroes', 'heroes_nationality', 'category', 'youtube_views']
df_yt_views = df_episodes_with_heroes[df_episodes_with_heroes['youtube_views'] >= 1000].sort_values(by='youtube_views', ascending=False)
df_yt_views.reset_index(drop=True, inplace=True)
df_yt_views[columns_of_interest].style.background_gradient(cmap='Greens')
plot_timeseries(title="Which episodes brought in subscribers?",
                data=df_episodes_with_heroes,
                x_column='episode_number',
                y_column='youtube_subscribers',
                area=False)

columns_of_interest = ['episode_id', 'episode_name', 'heroes', 'heroes_nationality', 'category', 'youtube_subscribers']
df_bringing_yt_subscribers = df_episodes[df_episodes['youtube_subscribers'] >= 40].sort_values(by='youtube_subscribers', ascending=False)
df_bringing_yt_subscribers.reset_index(drop=True, inplace=True)
df_bringing_yt_subscribers[columns_of_interest].style.background_gradient(cmap='Greens')
plot_timeseries(title="Which episodes brought in more YouTube watch hours?",
                data=df_episodes_with_heroes,
                x_column='episode_number',
                y_column='youtube_watch_hours',
                area=False)

columns_of_interest = ['episode_id', 'episode_name', 'heroes', 'heroes_nationality', 'category', 'youtube_watch_hours']
df_yt_watch_hours = df_episodes_with_heroes[df_episodes_with_heroes['youtube_watch_hours'] >= 100].sort_values(by='youtube_watch_hours', ascending=False)
df_yt_watch_hours.reset_index(drop=True, inplace=True)
df_yt_watch_hours[columns_of_interest].style.background_gradient(cmap='Greens')
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_episodes, x='youtube_thumbnail_type', y='youtube_ctr')
plt.title("YouTube CTR vs YouTube thumbnail type", fontsize=15)
plt.xlabel('youtube_thumbnail_type', fontsize=12)
plt.ylabel('youtube_ctr', fontsize=12)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df_episodes, x='youtube_thumbnail_type', y='youtube_views')
plt.title("YouTube views vs YouTube thumbnail type", fontsize=15)
plt.xlabel('youtube_thumbnail_type', fontsize=12)
plt.ylabel('youtube_views', fontsize=12)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df_episodes, x='youtube_thumbnail_type', y='youtube_avg_watch_duration')
plt.title("YouTube avg-watch-duration vs YouTube thumbnail type", fontsize=15)
plt.xlabel('youtube_thumbnail_type', fontsize=12)
plt.ylabel('youtube_avg_watch_duration', fontsize=12)
plt.show()
def get_talktimes_by_speaker(data):
    """
    Gets data about talktimes of host and guests, given the subtitles DataFrame.
    """
    host_name = 'Sanyam Bhutani'
    episodes = data['Episode'].unique().tolist()
    list_talktimes = []
    for episode in episodes:
        df_by_episode = data[data['Episode'] == episode]
        avg_host_talktime = df_by_episode[df_by_episode['Speaker'] == host_name]['TalkTimeInSecs'].mean()
        avg_guest_talktime = df_by_episode[df_by_episode['Speaker'] != host_name]['TalkTimeInSecs'].mean()
        dict_talktimes = {
            'Episode': episode,
            'AvgHostTalkTime': avg_host_talktime,
            'AvgGuestTalkTime': avg_guest_talktime
        }
        list_talktimes.append(dict_talktimes)
    return pd.DataFrame(data=list_talktimes)
df_subtitles.head()
df_talktimes = get_talktimes_by_speaker(data=df_subtitles)
df_talktimes.head()
linewidths = 3

plt.figure(figsize=(12, 5))
plt.plot(df_talktimes['Episode'], df_talktimes['AvgGuestTalkTime'], label='Guest', color='green', linewidth=linewidths)
plt.plot(df_talktimes['Episode'], df_talktimes['AvgHostTalkTime'], label='Host', color='blue', linewidth=linewidths)
plt.title("TalkTimes of Host vs Guest", fontsize=24)
plt.xlabel("Episode Number", fontsize=16)
plt.ylabel("AvgTalkTimeInMins", fontsize=16)
plt.legend(loc='best', fontsize=12)
plt.grid()
plt.show()
def generate_wordcloud(title, corpus):
    """
    Generates WordCloud from corpus of text.
    """
    corpus = str(corpus)
    stopwords = list(set(stopwords_wordcloud))
    corpus_cleaned = clean_text(text=corpus)
    tokens = corpus_cleaned.split(' ')
    comment_words = ""
    comment_words = comment_words + " ".join(tokens) + " "
    
    wordcloud = WordCloud(width=800, height=800,
                          background_color='black',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.title(str(title), fontsize=15)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    return None
# Let's learn about the following guests' word usage!
guests = ['Jeremy Howard', 'Abhishek Thakur', 'Parul Pandey', 'Shivam Bansal',
          'Edouard Harris', 'Robbert Bracco', 'Rohan Rao']

for guest in guests:
    dialogs = df_subtitles[df_subtitles['Speaker'] == guest]['Text'].tolist()
    corpus = ""
    for dialog in dialogs:
        corpus += str(dialog) + " "
    generate_wordcloud(title=f"WordCloud for {guest}", corpus=corpus)
def get_talking_speed(corpus, talktime_in_mins):
    """
    Calculates talking speed, given corpus of text and talk-time (in mins).
    Returns talking speed in words per minute.
    """
    words = str(corpus).split(' ')
    num_words_used = len(words)
    words_per_min = round(num_words_used / talktime_in_mins, 2)
    return words_per_min


def get_vocab_size(corpus, remove_stopwords=False):
    """
    Calculates size of vocabulary used from corpus of text.
    """
    corpus_cleaned = clean_text(text=corpus)
    words = corpus_cleaned.split(' ')
    vocabulary = list(set(words))
    if remove_stopwords:
        vocabulary = [word for word in vocabulary if word not in stopwords_wordcloud]
    return int(len(vocabulary))
%%time

guests = ['Jeremy Howard', 'Abhishek Thakur', 'Parul Pandey', 'Shivam Bansal',
          'Edouard Harris', 'Robbert Bracco', 'Rohan Rao']

df_speaker_stats = pd.DataFrame()
for guest in guests:
    df_subtitles_by_guest = df_subtitles[df_subtitles['Speaker'] == guest]
    dialogs = df_subtitles_by_guest['Text'].tolist()
    total_talktime_mins = round(df_subtitles_by_guest['TalkTimeInSecs'].sum() / 60, 2)
    corpus = ""
    for dialog in dialogs:
        corpus += str(dialog) + " "
    talking_speed = get_talking_speed(corpus=corpus, talktime_in_mins=total_talktime_mins)
    vocab_size = get_vocab_size(corpus=corpus, remove_stopwords=True)
    df_temp = pd.DataFrame(data={
        'Speaker': guest,
        'TalkingSpeedInWPM': round(talking_speed, 2),
        'VocabularySize': vocab_size
    }, index=[0])
    df_speaker_stats = pd.concat(objs=[df_speaker_stats, df_temp], ignore_index=True, sort=False)
df_speaker_stats.style.background_gradient(cmap='Greens')
%%time

guests = ['Jeremy Howard', 'Abhishek Thakur', 'Parul Pandey', 'Shivam Bansal',
          'Edouard Harris', 'Robbert Bracco', 'Rohan Rao']

df_sentiment = pd.DataFrame()

for guest in guests:
    df_subtitles_by_guest = df_subtitles[df_subtitles['Speaker'] == guest]
    dialogs = df_subtitles_by_guest['Text'].tolist()
    total_talktime_mins = round(df_subtitles_by_guest['TalkTimeInSecs'].sum() / 60, 2)
    corpus = ""
    for dialog in dialogs:
        corpus += str(dialog) + " "
    corpus_cleaned = clean_text(text=corpus)
    
    corpus_textblob_obj = TextBlob(corpus_cleaned)
    df_temp = pd.DataFrame(data={
        'Speaker': guest,
        'Corpus': corpus_cleaned,
        'Polarity': corpus_textblob_obj.sentiment.polarity,
        'Subjectivity': corpus_textblob_obj.sentiment.subjectivity
    }, index=[0])
    df_sentiment = pd.concat(objs=[df_sentiment, df_temp], ignore_index=True, sort=False)
df_sentiment
plt.figure(figsize=(10, 8))
for index, guest in enumerate(df_sentiment['Speaker']):
    df_sentiment_by_guest = df_sentiment[df_sentiment['Speaker'] == guest]
    polarities = df_sentiment_by_guest['Polarity'].iloc[0]
    subjectivities = df_sentiment_by_guest['Subjectivity'].iloc[0]
    plt.scatter(x=polarities, y=subjectivities, color='blue', s=100)
    plt.text(x=polarities+0.001, y=subjectivities+0.001, s=df_sentiment['Speaker'].iloc[index], fontsize=10)
plt.title("Sentiment Analysis", fontsize=24)
plt.xlabel('Polarity', fontsize=16)
plt.ylabel('Subjectivity', fontsize=16)
plt.grid()
plt.show()