import numpy as np

import pandas as pd

import os

import json

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS

from joblib import Parallel, delayed

import tqdm

import jieba

import time

import matplotlib.font_manager as fm
INPUT_PATH = "../input"

DATA_PATH = os.path.join(INPUT_PATH, os.listdir(INPUT_PATH)[0])

news_data_df = pd.read_csv(os.path.join(DATA_PATH, "news_collection.csv"), low_memory=False)
news_data_df.head()
news_data_df.info()
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
missing_data(news_data_df)
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
unique_values(news_data_df)
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
most_frequent_values(news_data_df)
news_data_df['year'] = news_data_df['date'].apply(lambda x: str(x)[0:4])

news_data_df['month'] = news_data_df['date'].apply(lambda x: str(x)[4:6])

news_data_df['day'] = news_data_df['date'].apply(lambda x: str(x)[6:8])
news_data_df[['title', 'date', 'year', 'month', 'day']].head()
def jieba_tokens(x, sep=' ', cut_all_flag=False):

    '''

    input: x - text in Chines to cut

    input: sep - separator to use in the output

    input: cut_all_flag - cut in individual ideograms rather than in concepts (groups of ideograms). 

    function: cut the text in Chinese in group of ideograms (or individual ideograms)

    output: the text cut in group of ideograms (or ideograms)

    '''

    try:

        return sep.join(jieba.cut(x, cut_all=cut_all_flag))

    except:

        return None
start_time = time.time()

news_data_df['proc_title'] = Parallel(n_jobs=4)(delayed(jieba_tokens)(x) for x in tqdm.tqdm_notebook(news_data_df['title'].values))

print(f"Total processing time: {round(time.time()-start_time,2)} sec.")
start_time = time.time()

news_data_df['proc_desc'] = Parallel(n_jobs=4)(delayed(jieba_tokens)(x) for x in tqdm.tqdm_notebook(news_data_df['desc'].values))

print(f"Total processing time: {round(time.time()-start_time,2)} sec.")
start_time = time.time()

news_data_df['main_url'] = news_data_df['url'].apply(lambda x: x.split('/')[2])

print(f"Total processing time: {round(time.time()-start_time,2)} sec.")
news_data_df[['main_url', 'url']].head()
def get_main_image(x):

    try:

        return x.split('/')[2]

    except:

        return x



start_time = time.time()

news_data_df['main_image'] = news_data_df['image'].apply(lambda x: get_main_image(x))

print(f"Total processing time: {round(time.time()-start_time,2)} sec.")
news_data_df[['main_image', 'image', 'main_url', 'url']].head()
!wget https://github.com/adobe-fonts/source-han-sans/raw/release/SubsetOTF/SourceHanSansCN.zip

!unzip -j "SourceHanSansCN.zip" "SourceHanSansCN/SourceHanSansCN-Regular.otf" -d "."

!rm SourceHanSansCN.zip

!ls
font_path = './SourceHanSansCN-Regular.otf'

font_prop = fm.FontProperties(fname=font_path)
def plot_count(feature, title, df, font_prop=font_prop, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:25], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop);

    plt.show()    
plot_count('main_image', 'Most frequent images sources (first 25 from all data)', news_data_df, size=4)
plot_count('main_url', 'Most frequent main pages of news sites (first 25 from all data)', news_data_df, size=4)
plot_count('source', 'Most frequent sources (first 25 from all data)', news_data_df, size=4)
plot_count('year', 'Year', news_data_df, size=1)
plot_count('month', 'Month', news_data_df, size=2)
plot_count('day', 'Day', news_data_df, size=4)
def most_frequent_texts(feature, df):

    total = float(len(df))

    dd = pd.DataFrame(df[feature].value_counts().index[:10], columns = ['Item'])

    dd['Frequency'] = df[feature].value_counts().values[:10]

    dd['Source'] = df['source']

    dd['Landing page'] = df['main_url']

    return dd
most_frequent_texts('title', news_data_df)
most_frequent_texts('desc', news_data_df)
prop = fm.FontProperties(fname=font_path, size=20)

stopwords = set(STOPWORDS)



def show_wordcloud(data, font_path=font_path, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        font_path=font_path,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        prop = fm.FontProperties(fname=font_path)

        fig.suptitle(title, fontsize=40, fontproperties=prop)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(news_data_df['proc_title'], font_path, title = 'Prevalent words in title, all data')
show_wordcloud(news_data_df['proc_desc'], font_path, title = 'Prevalent words in desc, all data')