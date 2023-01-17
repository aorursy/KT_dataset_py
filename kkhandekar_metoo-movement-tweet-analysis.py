#Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re,string,unicodedata

from string import punctuation

from math import pi

from PIL import Image



#Plotting Libraries

import matplotlib.pyplot as plt

import seaborn as sns



from bokeh.plotting import figure

from bokeh.io import output_notebook, show, output_file

from bokeh.models import ColumnDataSource, HoverTool, Panel

from bokeh.models.widgets import Tabs

from bokeh.palettes import Category20c

from bokeh.transform import cumsum

from bokeh.resources import INLINE



#NLTK Libraries

import nltk

from nltk.corpus import stopwords



#Warnings

import warnings

warnings.filterwarnings("ignore")



#Garbage Collection

import gc



#downloading wordnet/punkt dictionary

nltk.download('wordnet')

nltk.download('punkt')

nltk.download('stopwords')



#WordCloud Generator

from wordcloud import WordCloud,STOPWORDS



#Gensim Library for Text Processing

import gensim.parsing.preprocessing as gsp

from gensim import utils



from textblob import TextBlob, Word



#Keyword Extraction Libraries

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



#Tabulate

from tabulate import tabulate
data_url = '../input/hatred-on-twitter-during-metoo-movement/MeTooHate.csv'

data = pd.read_csv(data_url, header='infer')
data = data.drop(['status_id', 'created_at', 'favorite_count', 'retweet_count',

       'location', 'followers_count', 'friends_count', 'statuses_count',

       'category'], axis=1)
#Inspect

data.head()
#Shape

print("Total Tweets: ",data.shape[0])
#Check for null values

print("Number of records with null text columns: ",data['text'].isna().sum())
#Dropping records with null value

data = data.dropna()
#Taking Backup of prep'd data

data_bkp = data.copy()
#Number of Characters

data['chars'] = data['text'].str.len()
# Function to plot histogram

def plot_sumry(dataframe,column, title=''):

    plt.figure(figsize=(10,5))

    sns.set_palette('pastel')

    sns.set_color_codes()

    ax = sns.distplot(dataframe[column], color='midnightblue', bins=25)

    ax.set_title(title, fontsize=15)

    

    x_min = dataframe[column].min()

    x_max = dataframe[column].max()

    x_mean = dataframe[column].mean()

    

    print(f'Stat Summary of Tweet {column.capitalize()}:\n'

          f'Minimum Character Count   : {x_min}\n'

          f'Maximum Character Count   : {x_max}\n'

          f'Average Character Count   : {round(x_mean)}')



    

plot_sumry(data, 'chars', 'Characters Distribution')

    

# Creating a Sampler function



def sampler(dataframe,column):

    

    temp_df = dataframe[(dataframe[column]>=33) & (dataframe[column]<=150)]

    

    # sampling 15%

    df_sample = pd.DataFrame(temp_df['text'].sample(frac=0.10, replace=True, random_state=1))



    return df_sample.reset_index(drop=True)

#Creating a new sampled dataset

data_sample = sampler(data,'chars')

print("Number of Tweets in Sampled Dataset: ", data_sample.shape[0])
#Inspect new sampled dataset

data_sample.head()
# Create list of pre-processing func (gensim)

processes = [

               gsp.strip_tags, 

               gsp.strip_punctuation,

               gsp.strip_multiple_whitespaces,

               gsp.strip_numeric,

               gsp.remove_stopwords, 

               gsp.strip_short, 

               gsp.stem_text

            ]



# Create func to pre-process text

def proc_txt(txt):

    text = txt.lower()

    text = utils.to_unicode(text)

    for p in processes:

        text = p(text)

    return text

#Creating a new column with processed text

data_sample['text_proc'] = data_sample['text'].apply(lambda x: proc_txt(x))
#Taking a backup

data_proc_bkp = data_sample.copy()
# Creating a function to analyse the tweet sentiments



def sentiment_analyzer(text):

    #vad_sent = SentimentIntensityAnalyzer()

    #sentiment_dict = vad_sent.polarity_scores(text) 

    TB_sentiment_polarity = TextBlob(text).sentiment.polarity

    

    # decide sentiment as positive, negative and neutral 

    if TB_sentiment_polarity >= 0.00 : 

        return "Has No Hatred" 

  

    elif TB_sentiment_polarity <= 0.00 : 

        return "Has Hatred" 

  

    else : 

        return "Is Neutral"

    

#Analysing the sentiment

data_sample['sentiments'] = data_sample['text_proc'].apply(lambda x: sentiment_analyzer(x))
data_sample.head()
#Taking Backup

data_sentiments = data_sample.copy()
sentiment_count = data_sample.groupby('sentiments').size()



# Data to plot

labels = 'Has Hatred', 'Has No Hatred'

sizes = [sentiment_count[0], sentiment_count[1]]

colors = ['dimgrey', 'lightgray']

explode = (0.1, 0)  # explode 1st slice

fig = plt.figure(figsize=[8, 6])



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.title("Sentiment Distribution (sample data)", fontsize=16)

plt.show()

#Creating a stopword lists

stopword_list = set(stopwords.words('english'))



#get the text from text_proc columns

docs = data_sample['text_proc'].tolist()



#Create vocab of words & ignore words that appear in 80% of documents

cv = CountVectorizer(max_df=0.80,

                     stop_words=stopword_list,

                     max_features=40000

                    )



#Fitting the CountVector to the list created above

word_count_vector = cv.fit_transform(docs)
#Compute IDF Value

tf_transform = TfidfTransformer(smooth_idf=True, use_idf=True)

tf_transform.fit(word_count_vector)
# -- Extract Keywords Custom Function --



#Sort in Descending Order

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1],x[0]), reverse=True)

    

#Extract Keywords

def extraction(feature_names, sorted_items, n):

    sorted_items = sorted_items[:n]

    

    score_vals = []

    feature_vals = []

    

    for idx, score in sorted_items:

        

        score_vals.append(round(score,3))

        feature_vals.append(feature_names[idx])

        

    return feature_vals
#Extracting Keywords



feature_names = cv.get_feature_names()



#Creating seperate dataframe for different sentiments

df_Hate = data_sample[(data_sample['sentiments'] == 'Has Hatred')]

df_noHate = data_sample[(data_sample['sentiments'] == 'Has No Hatred')]



#Seperate docs for Hatred & No Hatred

docs_Hate = df_Hate['text'].tolist()

docs_noHate = df_noHate['text'].tolist()



for ht, nht in zip(docs_Hate,docs_noHate):

    tf_idf_vector_Hatred = tf_transform.transform(cv.transform([ ht ]))

    tf_idf_vector_noHatred = tf_transform.transform(cv.transform([ nht ]))

    

sorted_items_Hatred = sort_coo(tf_idf_vector_Hatred.tocoo())

sorted_items_noHatred = sort_coo(tf_idf_vector_noHatred.tocoo())



Hatred_keywords = extraction(feature_names,sorted_items_Hatred,100)

noHatred_keywords = extraction(feature_names,sorted_items_noHatred,100)
tab_data = [[Hatred_keywords]]

print(tabulate(tab_data, headers=['Keywords in Tweets with Hatred']))

print(" ")

tab_data_nht = [[noHatred_keywords]]

print(tabulate(tab_data_nht, headers=['Keywords in Tweets with no Hatred']))

# Function to plot word cloud

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(12.0,12.0), 

                   title = None, title_size=20, image_color=False):



    wordcloud = WordCloud(background_color='white',

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(text)

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'green', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

d = '../input/masks/masks-wordclouds/'
txt = str(df_Hate.text)

comments_mask = np.array(Image.open(d + 'comment.png'))

plot_wordcloud(txt, comments_mask, max_words=1000, max_font_size=100, 

               title = 'Common Words in Tweets with Hatred', title_size=30)
txt = str(df_noHate.text)

comments_mask = np.array(Image.open(d + 'comment.png'))

plot_wordcloud(txt, comments_mask, max_words=1000, max_font_size=100, 

               title = 'Common Words in Tweets with no Hatred', title_size=30)