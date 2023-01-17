from IPython.display import Image

import os

!ls ../input/image1

Image("../input/image1/COVID_TWITTER1200X600.jpg")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")
data.info()
data_shape =  data.shape

print(f"Shape of the dataset {data_shape}")
data.head(10).style.highlight_max(color ='red').highlight_min(color='green')
data.tail(5).style.highlight_max(color='blue').highlight_min(color='lightgreen')
data.describe().style.highlight_max(color="green").highlight_min(color="lightgreen")
data.columns
missing_graph = sns.heatmap(data.isnull(),cbar=False,yticklabels=False,cmap='viridis')

print(f"Graphically Representation of Missing values : \n {missing_graph}")
def most_frequent_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    items = []

    vals = []

    for col in data.columns:

        itm = data[col].value_counts().index[0]

        val = data[col].value_counts().values[0]

        items.append(itm)

        vals.append(val)

    tt['Most frequent item'] = items

    tt['Frequence'] = vals

    tt['Percent from total'] = np.round(vals / total * 100, 3)

    return(np.transpose(tt))



most_frequent_values(data)
def plot_features_distribution(features, title, df, isLog=False):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        if(isLog):

            sns.distplot(np.log1p(df[feature]),kde=True,hist=False, bins=120, label=feature)

        else:

            sns.distplot(df[feature],kde=True,hist=False, bins=120, label=feature)

    plt.xlabel('')

    plt.legend()

    plt.show()

data['hashtags'] = data['hashtags'].replace(np.nan, "['None']", regex=True)

data['hashtags'] = data['hashtags'].apply(lambda x: x.replace('\\N',''))

data['hashtags_count'] = data['hashtags'].apply(lambda x: len(x.split(',')))

plot_features_distribution(['hashtags_count'], 'Hashtags per tweet (all data)', data)
data['hashtags_individual'] = data['hashtags'].apply(lambda x: x.split(','))

from itertools import chain

all_hashtags = set(chain.from_iterable(list(data['hashtags_individual'])))

print(f"There are totally: {len(all_hashtags)}")
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(data['text'], title = 'Prevalent words in tweets')
india_df = data.loc[data.user_location=="India"]

show_wordcloud(india_df['text'], title = 'Prevalent words in tweets from India')
new_df = data.loc[data.user_location=="New Delhi"]

show_wordcloud(new_df['text'], title = 'Prevalent words in tweets from New Delhi')
mumbai_df = data.loc[data.user_location=="Mumbai"]

show_wordcloud(mumbai_df['text'], title = 'Prevalent words in tweets from Mumbai')
pune_df = data.loc[data.user_location=="Pune"]

show_wordcloud(pune_df['text'], title = 'Prevalent words in tweets from Pune')
hyderabad_df = data.loc[data.user_location=="Hyderabad"]

show_wordcloud(hyderabad_df['text'], title = 'Prevalent words in tweets from Hyderabad')
delhi_df = data.loc[data.user_location=="Delhi"]

show_wordcloud(delhi_df['text'], title = 'Prevalent words in tweets from Delhi')
punjab_df = data.loc[data.user_location=="Punjab"]

show_wordcloud(punjab_df['text'], title = 'Prevalent words in tweets from Punjab')
USA_df = data.loc[data.user_location=="USA"]

show_wordcloud(USA_df['text'], title = 'Prevalent words in tweets from USA')
london_df = data.loc[data.user_location=="London"]

show_wordcloud(london_df['text'], title = 'Prevalent words in tweets from London')
world_df = data.loc[data.user_location=="WORLDWIDE"]

show_wordcloud(world_df['text'], title = 'Prevalent words in tweets from WORLDWIDE')
global_df = data.loc[data.user_location=="Global"]

show_wordcloud(global_df['text'], title = 'Prevalent words in tweets from Global')
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count("user_location", "User location", data,4)
plot_count("user_name", "User name", data,4)
plot_count("source", "Source", data,4)