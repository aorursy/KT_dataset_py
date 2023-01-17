import gc

import re

import string



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.express as px



from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer
input_folder = "/kaggle/input/tweet-sentiment-extraction/"

train = pd.read_csv(f'{input_folder}train.csv')

test = pd.read_csv(f'{input_folder}test.csv')

ss = pd.read_csv(f'{input_folder}sample_submission.csv')
print("Shape of train data",train.shape)

print("Shape of test data",test.shape)
train.head()
train = train.dropna()

test = test.dropna()

print("Number of missing value in the train data",train.isnull().sum())

print("Number of missing value in the test data",test.isnull().sum())

#We are calculating the meta features only for the test column as it is common in both train and test



#word_count

train['word_count'] = train['text'].apply(lambda x:len(str(x).split()))

test['word_count'] = test['text'].apply(lambda x:len(str(x).split()))



# unique_word_count

train['unique_word_count'] = train['text'].apply(lambda x: len(set(str(x).split())))

test['unique_word_count'] = test['text'].apply(lambda x: len(set(str(x).split())))



# stop_word_count

train['stop_word_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

test['stop_word_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



# url_count

train['url_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

test['url_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))



# mean_word_length

train['mean_word_length'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test['mean_word_length'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



# char_count

train['char_count'] = train['text'].apply(lambda x: len(str(x)))

test['char_count'] = test['text'].apply(lambda x: len(str(x)))



# punctuation_count

train['punctuation_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

test['punctuation_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))



# hashtag_count

train['hashtag_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

test['hashtag_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))



# mention_count

train['mention_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

test['mention_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))



#print the train and test data

train.head()
METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'mean_word_length',

                'char_count', 'punctuation_count']





fig, axes = plt.subplots(ncols=1, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)



for i,feat in enumerate(METAFEATURES):

    #Distribution for the training set w.r.t target category

    sns.distplot(train.loc[train['sentiment'] == 'positive'][feat], label="Positive", ax=axes[i], color='green')

    sns.distplot(train.loc[train['sentiment'] == 'neutral'][feat], label="Neutral", ax=axes[i], color='blue')

    sns.distplot(train.loc[train['sentiment'] == 'negative'][feat], label="Negative", ax=axes[i], color='red')

    

    axes[i].set_xlabel('')

    axes[i].tick_params(axis='x', labelsize=12)

    axes[i].tick_params(axis='y', labelsize=12)

    axes[i].legend()

    

    axes[i].set_title(f'{feat} Target Distribution in Training Set', fontsize=20)



plt.show()

    
METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'mean_word_length',

                'char_count', 'punctuation_count']



fig, axes = plt.subplots(ncols=1, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)



for i,feat in enumerate(METAFEATURES):

    #Distribution of train and test data set

    sns.distplot(train[feat], label='Training',ax=axes[i])

    sns.distplot(test[feat], label='Test',ax=axes[i])

    

    axes[i].set_xlabel('')

    axes[i].tick_params(axis='x', labelsize=12)

    axes[i].tick_params(axis='y', labelsize=12)

    axes[i].legend()

    

    axes[i].set_title(f'{feat} Training & Test set Distribution', fontsize=20)



plt.show()
sentiment_count = train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)

plt.figure(figsize=(12,6))

plt.title("Count of Target category ")

sns.countplot(x='sentiment',data=train)



fig = px.pie(sentiment_count, values='text', names='sentiment', title='Distribution of Target category', color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=600, 

                    height=300,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train.loc[train['sentiment'] == 'neutral', 'text'], title="Word Cloud of Neutral tweets on train data",color = 'white')
plot_wordcloud(train.loc[train['sentiment'] == 'positive', 'text'], title="Word Cloud of Positive tweets on train data",color = 'white')
plot_wordcloud(train.loc[train['sentiment'] == 'negative', 'text'], title="Word Cloud of Negative tweets on train data",color = 'white')
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train['clean_text'] = train['text'].apply(lambda x:clean_text(x))

test['clean_text'] = test['text'].apply(lambda x:clean_text(x))
def get_top_n_remove_stop_words(corpus, n=None,ngram=1):

    vec = CountVectorizer(ngram_range=(ngram,ngram), stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
#Distribution of ngrams in train and test set

#We are removing the stop words also

fig, axes = plt.subplots(ncols=3, figsize=(10, 15), dpi=100)

plt.tight_layout()



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'positive']['clean_text'], 20)

df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[0], color='green', data=df2)



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'neutral']['clean_text'], 20)

df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[1], color='blue', data=df1)



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'negative']['clean_text'], 20)

df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[2], color='red', data=df)



for i in range(3):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=10)

    axes[i].tick_params(axis='y', labelsize=10)



axes[0].set_title(f'Top {20} positive words', fontsize=10)

axes[1].set_title(f'Top {20} neutral words', fontsize=10)

axes[2].set_title(f'Top {20} negative words', fontsize=10)
#Distribution of ngrams in train and test set

#We are removing the stop words also

fig, axes = plt.subplots(ncols=3, figsize=(10, 15), dpi=100)

plt.tight_layout()



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'positive']['clean_text'], 20, 2)

df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[0], color='green', data=df2)



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'neutral']['clean_text'], 20, 2)

df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[1], color='blue', data=df1)



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'negative']['clean_text'], 20, 2)

df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[2], color='red', data=df)



for i in range(3):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=10)

    axes[i].tick_params(axis='y', labelsize=10)



axes[0].set_title(f'Top {20} positive words', fontsize=10)

axes[1].set_title(f'Top {20} neutral words', fontsize=10)

axes[2].set_title(f'Top {20} negative words', fontsize=10)
#Distribution of ngrams in train and test set

#We are removing the stop words also

fig, axes = plt.subplots(ncols=3, figsize=(10, 15), dpi=100)

plt.tight_layout()



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'positive']['clean_text'], 20, 3)

df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[0], color='green', data=df2)



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'neutral']['clean_text'], 20, 3)

df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[1], color='blue', data=df1)



common_words = get_top_n_remove_stop_words(train[train['sentiment'] == 'negative']['clean_text'], 20, 3)

df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

sns.barplot(y='ReviewText', x='count', ax=axes[2], color='red', data=df)



for i in range(3):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=10)

    axes[i].tick_params(axis='y', labelsize=10)



axes[0].set_title(f'Top {20} positive words', fontsize=10)

axes[1].set_title(f'Top {20} neutral words', fontsize=10)

axes[2].set_title(f'Top {20} negative words', fontsize=10)