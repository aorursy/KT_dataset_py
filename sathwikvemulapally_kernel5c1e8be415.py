# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install -q langdetect

!pip install -q textstat
import numpy as np

import pandas as pd



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)

pd.set_option('display.width', 1000)



from collections import defaultdict,Counter

from multiprocessing import Pool



import textstat

from statistics import *



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

import plotly.express as px

import plotly.offline as py



from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer, PorterStemmer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



from langdetect import detect



from wordcloud import WordCloud, STOPWORDS



from scipy.stats import norm, kurtosis, skew



from tqdm import tqdm

tqdm.pandas() 

import string, json, nltk, gc
stop = set(stopwords.words('english'))

plt.style.use('seaborn')
TRAIN_UNINTENDED_BIAS = "../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv"

TRAIN_TOXICITY = "../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv"



VALIDATION = "../input/jigsaw-multilingual-toxic-comment-classification/validation.csv"



TEST = "../input/jigsaw-multilingual-toxic-comment-classification/test.csv"
train_1_df = pd.read_csv(TRAIN_UNINTENDED_BIAS)

train_2_df = pd.read_csv(TRAIN_TOXICITY)



validation_df = pd.read_csv(VALIDATION)



test_df = pd.read_csv(TEST)
train_1_df.head()
train_2_df.head()
fig, ax = plt.subplots(1, 3, figsize=(20, 5))



sns.countplot(train_1_df['toxic'].astype(int), ax=ax[0])

ax[0].set_title('Unintended bias dataset')



sns.countplot(train_2_df['toxic'].astype(int), ax=ax[1])

ax[1].set_title('Toxicity dataset')



sns.countplot(validation_df['toxic'].astype(int), ax=ax[2])

ax[2].set_title('Validation dataset')



fig.suptitle('Toxicity distribution across datasets', fontweight='bold', fontsize=14)



fig.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))



sns.countplot(validation_df['lang'], ax=ax[0])

ax[0].set_title('Validation')



sns.countplot(test_df['lang'], ax=ax[1])

ax[1].set_title('Test')



fig.suptitle('Language distribution across datasets', fontweight="bold", fontsize=14)

fig.show()
fig, ax = plt.subplots(1,2, figsize=(15, 5))



sns.distplot(train_1_df[train_1_df['toxic']==0]['comment_text'].str.len(), axlabel="Non toxic", ax=ax[0])

sns.distplot(train_1_df[train_1_df['toxic']==1]['comment_text'].str.len(), axlabel="Toxic", ax=ax[1])



fig.show()



fig.suptitle("Distribution of number of No: Characters in Comments - Unintended bias dataset", fontsize=14)
fig, ax = plt.subplots(1,2, figsize=(15, 5))



sns.distplot(train_2_df[train_2_df['toxic']==0]['comment_text'].str.len(), axlabel="Non toxic", ax=ax[0])

sns.distplot(train_2_df[train_2_df['toxic']==1]['comment_text'].str.len(), axlabel="Toxic", ax=ax[1])



fig.show()



fig.suptitle("Distribution of number of No: Characters in Comments - Toxicity dataset", fontsize=14)
train_1_df[train_1_df['comment_text'].str.len() > 850][['comment_text', 'toxic']].sample(n=100).reset_index(drop=True)
train_2_df[train_2_df['comment_text'].str.len() > 2000][["comment_text", "toxic"]].sample(n=100).reset_index(drop=True)
fig, ax = plt.subplots(1,3, figsize=(15, 5))



sns.distplot(validation_df[validation_df['toxic']==0]['comment_text'].str.len(), axlabel="Validation - Non toxic", ax=ax[0])

sns.distplot(validation_df[validation_df['toxic']==1]['comment_text'].str.len(), axlabel="Validation - Toxic", ax=ax[1])

sns.distplot(test_df['content'].str.len(), axlabel="Test", ax=ax[2])



fig.show()



fig.suptitle("Distribution of number of No: Characters in Comments - Toxicity dataset", fontsize=14)
validation_df[validation_df['comment_text'].str.len() > 1000][["comment_text", 'lang', "toxic"]].sample(n=100).reset_index(drop=True)
test_df[test_df['content'].str.len() > 1000][["content", 'lang']].sample(n=100).reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))



validation_df["character_count"] = validation_df['comment_text'].apply(lambda x: len(x))

test_df['character_count'] = test_df['content'].apply(lambda x: len(x))



test_df['character_count'] = test_df['character_count'].apply(lambda x: 1000 if x > 1000 else x) # Nicer formatting 



sns.boxplot('lang', 'character_count', data=validation_df, ax=ax[0])

sns.boxplot('lang', 'character_count', data=test_df, ax=ax[1])



fig.show()



fig.suptitle('Distribution of # of characters for each language')
fig, ax = plt.subplots(1,2, figsize=(15, 5))



sns.distplot(train_1_df[train_1_df['toxic']==0]['comment_text'].str.split().str.len(), axlabel="Non toxic", ax=ax[0])

sns.distplot(train_1_df[train_1_df['toxic']==1]['comment_text'].str.split().str.len(), axlabel="Toxic", ax=ax[1])



fig.show()



fig.suptitle("Distribution of number of No: Words in Comments - Unintended bias", fontsize=14)
fig, ax = plt.subplots(1,2, figsize=(15, 5))



sns.distplot(train_2_df[train_2_df['toxic']==0]['comment_text'].str.split().str.len(), axlabel="Non toxic", ax=ax[0])

sns.distplot(train_2_df[train_2_df['toxic']==1]['comment_text'].str.split().str.len(), axlabel="Toxic", ax=ax[1])



fig.show()



fig.suptitle("Distribution of number of No: Words in Comments - Toxicity", fontsize=14)
def whisker_plot_stats(train):

    ## Number of words 

    train["num_words"] = train["comment_text"].progress_apply(lambda x: len(str(x).split()))



    ## Number of unique words 

    train["num_unique_words"] = train["comment_text"].progress_apply(lambda x: len(set(str(x).split())))



    ## Number of characters 

    train["num_chars"] = train["comment_text"].progress_apply(lambda x: len(str(x)))



    ## Number of stopwords 

    train["num_stopwords"] = train["comment_text"].progress_apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



    ## Number of punctuations 

    train["num_punctuations"] =train['comment_text'].progress_apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



    ## Number of title case words

    train["num_words_upper"] = train["comment_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



    # Number of title case words

    train["num_words_title"] = train["comment_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



    # Average length of the words

    train["mean_word_len"] = train["comment_text"].progress_apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    

    return train
print('Train 1...')

train_1_df = whisker_plot_stats(train_1_df)

print('Train 2...')

train_2_df = whisker_plot_stats(train_2_df)
train_1_df['num_words'].loc[train_1_df['num_words']>100] = 100

train_1_df['num_punctuations'].loc[train_1_df['num_punctuations']>10] = 10 

train_1_df['num_chars'].loc[train_1_df['num_chars']>350] = 350 

train_1_df['toxic'] = train_1_df['toxic'].apply(lambda x: 1 if x > 0.5 else 0)



train_2_df['num_words'].loc[train_2_df['num_words']>100] = 100

train_2_df['num_punctuations'].loc[train_2_df['num_punctuations']>10] = 10 

train_2_df['num_chars'].loc[train_2_df['num_chars']>350] = 350 





# figure related code

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

fig.suptitle('Distribution of # words in toxicity dataset', fontsize=14, fontweight='bold')



sns.boxplot(x='toxic', y='num_words', data=train_1_df, ax=ax[0])

ax[0].set_title('Unintended bias dataset')



sns.boxplot(x='toxic', y='num_words', data=train_2_df, ax=ax[1])

ax[1].set_title('Toxicity dataset')



fig.show()
train_1_df[train_1_df['num_words'] >= 100]['comment_text'].sample(n=100).reset_index(drop=True)
train_2_df[train_2_df['num_words'] >= 100]['comment_text'].sample(n=100).reset_index(drop=True)
def preprocess_comments(df, stop=stop, n=1, col='comment_text'):

    new_corpus=[]

    

    stem = PorterStemmer()

    lem = WordNetLemmatizer()

    

    for text in tqdm(df[col], total=len(df)):

        words = [w for w in word_tokenize(text) if (w not in stop)]

       

        words = [lem.lemmatize(w) for w in words if(len(w)>n)]

     

        new_corpus.append(words)

        

    new_corpus = [word for l in new_corpus for word in l]

    

    return new_corpus
fig,ax = plt.subplots(1, 2, figsize=(15,7))



for i in range(2):

    new = train_1_df[train_1_df['toxic']== i]

    corpus_train = preprocess_comments(new, {})

    

    dic = defaultdict(int)

    for word in corpus_train:

        if word in stop:

            dic[word]+=1

            

    top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    x, y = zip(*top)

    ax[i].bar(x,y)

    ax[i].set_title(str(i))



fig.suptitle("Common stopwords in unintented bias dataset")
fig, ax = plt.subplots(1,2,figsize=(20,12))



for i in range(2):

    new = train_2_df[train_2_df['toxic']==i]   

    corpus = corpus_train

    counter = Counter(corpus)

    most = counter.most_common()

    x = []

    y = []

    

    for word,count in most[:20]:

        if (word not in stop) :

            x.append(word)

            y.append(count)

            

    sns.barplot(x=y, y=x, ax=ax[i])

    ax[i].set_title(str(i))

    

fig.suptitle("Common words in toxicity dataset")
fig, ax = plt.subplots(1,2,figsize=(20,12))



for i in range(2):

    new = train_2_df[train_2_df['toxic']==i]   

    corpus = corpus_train_2

    counter = Counter(corpus)

    most = counter.most_common()

    x = []

    y = []

    

    for word,count in most[:20]:

        if (word not in stop) :

            x.append(word)

            y.append(count)

            

    sns.barplot(x=y, y=x, ax=ax[i])

    ax[i].set_title(str(i))

    

fig.suptitle("Common words in unintended bias dataset")
def get_top_ngram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(n, n),stop_words=stop).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:20]
fig, ax = plt.subplots(1, 2, figsize=(15, 10))



for i in range(2):

    new = train_1_df[train_1_df['toxic'] == i]['comment_text']

    top_n_bigrams = get_top_ngram(new, 2)[:20]

    x, y = map(list, zip(*top_n_bigrams))

    sns.barplot(x=y, y=x, ax=ax[i])

    ax[i].set_title(str(i))

    

fig.suptitle('Common bigrams in unintended bias dataset')
fig, ax = plt.subplots(1,2,figsize=(15,10))



for i in range(2):

    new = train_2_df[train_2_df['toxic'] == i]['comment_text']

    top_n_bigrams = get_top_ngram(new, 2)[:20]

    x, y = map(list,zip(*top_n_bigrams))

    sns.barplot(x=y,y=x,ax=ax[i])

    ax[i].set_title(str(i))

    

fig.suptitle("Common bigrams in toxicity dataset")
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None,ax=None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=100,

        max_font_size=30, 

        scale=3,

        random_state=1 

        )

    

    wordcloud = wordcloud.generate(str(data))

    ax.imshow(wordcloud,interpolation='nearest')

    ax.axis('off')
fig, ax = plt.subplots(1,2,figsize=(20,12))



for i in range(2):

    new = train_2_df[train_2_df['toxic'] == i]['comment_text']

    show_wordcloud(new, ax=ax[i])

    ax[i].set_title(str(i))

    

fig.suptitle('Wordcloud for toxicity dataset')
fig, ax = plt.subplots(1,2,figsize=(20,12))



for i in range(2):

    new = train_1_df[train_1_df['toxic'] == i]['comment_text']

    show_wordcloud(new, ax=ax[i])

    ax[i].set_title(str(i))

    

fig.suptitle('Wordcloud for unintended bias dataset')
def plot_readability(a, b, title, bins=0.4):

    

    # Setting limits

    a = a[a >= 0]

    a = a[a <= 100]

    b = b[b >= 0]

    b = b[b <= 100]

    

    trace1 = ff.create_distplot([a, b], ['non toxic', 'toxic'], bin_size=bins, show_rug=False)

    trace1['layout'].update(title=title)

    

    py.iplot(trace1, filename='Distplot')

    

    table_data= [["Statistical Measures","non toxic",'toxic'],

                 ["Mean",mean(a),mean(b)],

                 ["Standard Deviation",pstdev(a),pstdev(b)],

                 ["Variance",pvariance(a),pvariance(b)],

                 ["Median",median(a),median(b)],

                 ["Maximum value",max(a),max(b)],

                 ["Minimum value",min(a),min(b)]]

    

    trace2 = ff.create_table(table_data)

    py.iplot(trace2, filename='Table')
fre_non_toxic = np.array(train_1_df["comment_text"][train_1_df["toxic"].astype(int) == 0].sample(n=150000).apply(textstat.flesch_reading_ease))

fre_toxic = np.array(train_1_df["comment_text"][train_1_df["toxic"].astype(int) == 1].apply(textstat.flesch_reading_ease))



plot_readability(fre_non_toxic, fre_toxic, "Flesch Reading Ease - Unintended bias dataset", 1) 
fre_non_toxic = np.array(train_2_df['comment_text'][train_2_df['toxic'].astype(int) == 0].apply(textstat.flesch_reading_ease))

fre_toxic = np.array(train_2_df['comment_text'][train_2_df['toxic'].astype(int) == 1].apply(textstat.flesch_reading_ease))



plot_readability(fre_non_toxic, fre_toxic, "Flesch Reading Ease - Toxicity dataset", 1)
fre_non_toxic = np.array(validation_df['comment_text'][validation_df['toxic'].astype(int) == 0].apply(textstat.flesch_reading_ease))

fre_toxic = np.array(validation_df['comment_text'][validation_df['toxic'].astype(int) == 1].apply(textstat.flesch_reading_ease))



plot_readability(fre_non_toxic, fre_toxic, "Flesch Reading Ease - Validation set", 1)