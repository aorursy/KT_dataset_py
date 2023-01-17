import numpy as np 

import pandas as pd

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS
data_df = pd.read_csv("/kaggle/input/bbc-youtube-videos-metadata/bbc.csv")
print(f"data shape: {data_df.shape}")
data_df.info()
data_df.describe()
data_df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(data_df)
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(data_df)
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
most_frequent_values(data_df)
def plot_count(feature, title, df, size=1, ordered=True):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    if ordered:

        g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    else:

        g = sns.countplot(df[feature], palette='Set3')

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
plot_count("video_category_id", "Video category", data_df,4)
plot_count("video_category_label", "Video category label", data_df,4)
plot_count("definition", "Video definition", data_df,2)
plot_count("licensed_content", "Licensed content", data_df,1)
plot_count("caption", "Caption", data_df,1)
plot_count("dimension", "Dimmension", data_df,1)
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
show_wordcloud(data_df['video_title'], title = 'Prevalent words in video title')
hd_df = data_df.loc[data_df.definition == 'hd']

show_wordcloud(hd_df['video_title'], title = 'Prevalent words in high definition video title')
sd_df = data_df.loc[data_df.definition == 'sd']

show_wordcloud(sd_df['video_title'], title = 'Prevalent words in simple definition video title')
d_df = data_df.loc[data_df.video_category_label == 'Entertainment']

show_wordcloud(d_df['video_title'], title = 'Prevalent words in entertainment video title')
d_df = data_df.loc[data_df.video_category_label == 'Comedy']

show_wordcloud(d_df['video_title'], title = 'Prevalent words in comedy video title')
d_df = data_df.loc[data_df.video_category_label == 'News & Politics']

show_wordcloud(d_df['video_title'], title = 'Prevalent words in News & Politics video title')
d_df = data_df.loc[data_df.video_category_label == 'Science & Technology']

show_wordcloud(d_df['video_title'], title = 'Prevalent words in Science & Technology video title')
show_wordcloud(data_df['video_description'], title = 'Prevalent words in video description')
def plot_features_distribution(features, title, df, isLog=False):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        if(isLog):

            sns.distplot(np.log1p(df[feature]),kde=True,hist=True, bins=120, label=feature)

        else:

            sns.distplot(df[feature],kde=True,hist=True, bins=120, label=feature)

    plt.xlabel('#')

    plt.legend()

    plt.show()
plot_features_distribution(['duration_sec'], 'Video duration distribution (sec.)', data_df)
plot_features_distribution(['duration_sec'], 'Video duration distribution (sec./logaritmic)', data_df, isLog=True)
plot_features_distribution(['view_count'], 'View count distribution (logaritmic)', data_df, isLog=True)
plot_features_distribution(['like_count'], 'Like count distribution (logaritmic)', data_df, isLog=True)
plot_features_distribution(['dislike_count'], 'Dislike count distribution (logaritmic)', data_df, isLog=True)
plot_features_distribution(['comment_count'], 'Comments count distribution (logaritmic)', data_df, isLog=True)
plot_features_distribution(['comment_count', 'dislike_count', 'like_count', 'view_count'],

                           'Feedback distribution - all (logaritmic)', data_df, isLog=True)
def plot_feature_distribution_grouped(feature, title, df, hue, size=4):

    plt.figure(figsize=(size*5,size*2))

    plt.title(title)

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    g = sns.countplot(df[feature], hue=df[hue], palette='Set3')

    plt.xlabel('#')

    plt.legend()

    plt.show()
plot_feature_distribution_grouped('video_category_label', 'Video category label grouped by video definition', data_df, 'definition', size=3)
plot_feature_distribution_grouped('caption', 'Video caption grouped by video category label', data_df, 'video_category_label', size=3)