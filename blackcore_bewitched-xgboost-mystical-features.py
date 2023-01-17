import base64

import numpy as np

import pandas as pd

import plotly.graph_objs as go

import plotly.tools as tls

from collections import Counter

from scipy.misc import imread

import xgboost as xgb

import seaborn as sns

import nltk

import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from sklearn import ensemble, metrics, model_selection, naive_bayes

from matplotlib import pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)



color = sns.color_palette()



pd.options.mode.chained_assignment = None
# Read training data with Pandas

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print("Number of rows in train dataset : ",df_train.shape[0])

print("Number of rows in test dataset : ",df_test.shape[0])
df_train.head()
df_train.describe()
df_train['text_len']=df_train["text"].apply(lambda x: len(str(x)))



plt.figure(figsize=(14,8))

sns.violinplot(x="text_len", y="author", data=df_train, scale="width")

plt.ylabel('Author Name', fontsize=14)

plt.xlabel('Text Length', fontsize=14)

plt.show()
plt.figure(figsize=(14,8))

sns.violinplot(x="text_len", y="author", data=df_train[df_train["text_len"] < 400], scale="width")

plt.ylabel('Author Name', fontsize=14)

plt.xlabel('Text Length', fontsize=14)

plt.show()
auth_cnt = df_train['author'].value_counts()

auth_cnt.values



plt.figure(figsize=(14,8))

sns.barplot(auth_cnt.index, auth_cnt.values)

plt.ylabel('Number of Occurrences', fontsize=14)

plt.xlabel('Author Name', fontsize=14)

plt.show()
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer



# Needed to get rid of punctuation

tokenizer = RegexpTokenizer(r'\w+')

# Searching a set is much faster than searching a list

eng_stopwords = set(stopwords.words("english") + 

                ['one','us','yet','could','would','need','even','might','like',

                 'must','every','never','go','thus','may','much','however'])



def cleanup_spooky_text( spooky_text ):

    # 1. Convert to lower case, and tokenize (split) into individual words

    spooky_words = tokenizer.tokenize(spooky_text.lower())

    # 2. Remove stop words

    meaningful_spooky_words = [w for w in spooky_words if not w in eng_stopwords]   

    # 3. Join the words back into one string separated by space

    return( " ".join(meaningful_spooky_words))



print(eng_stopwords)
df_train["text"].head()
df_train["clean_text"] = df_train["text"].apply(lambda x: cleanup_spooky_text(x))
df_train["clean_text"].head()
df_train['clean_text_len']=df_train["clean_text"].apply(lambda x: len(str(x)))



plt.figure(figsize=(14,8))

sns.violinplot(x="clean_text_len", y="author", data=df_train[df_train["clean_text_len"] < 300], scale="width")

plt.ylabel('Author Name', fontsize=14)

plt.xlabel('Text Length', fontsize=14)

plt.show()
all_words = df_train['clean_text'].str.split(expand=True).unstack().value_counts()



data = [go.Bar(

            x = all_words.index.values[2:40],

            y = all_words.values[2:40],

            marker= dict(colorscale='Viridis',color = all_words.values[2:80]),

            text='Word counts'

    )]



layout = go.Layout(title='Top 40 Word frequencies in the cleansed training dataset')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
EAP_text = df_train[df_train.author=="EAP"]["clean_text"].values

HPL_text = df_train[df_train.author=="HPL"]["clean_text"].values

MWS_text = df_train[df_train.author=="MWS"]["clean_text"].values
from wordcloud import WordCloud
plt.figure(figsize=(20,20))

plt.subplot(211)

wc = WordCloud(background_color="black", max_words=100, 

               stopwords=eng_stopwords, max_font_size= 40)

wc.generate(" ".join(EAP_text))

plt.title("Edgar Allan Poe\n", fontsize=30)

plt.imshow(wc.recolor(colormap= 'viridis', random_state=17))

plt.axis('off')
plt.figure(figsize=(20,20))

wc = WordCloud(background_color="black", max_words=100, 

               stopwords=eng_stopwords, max_font_size= 40)

wc.generate(" ".join(HPL_text))

plt.title("HP Lovecraft\n", fontsize=30)

plt.imshow(wc.recolor(colormap= 'viridis', random_state=17))

plt.axis('off')
plt.figure(figsize=(20,20))

wc = WordCloud(background_color="black", max_words=100, 

               stopwords=eng_stopwords, max_font_size= 40)

wc.generate(" ".join(MWS_text))

plt.title("Mary Shelley\n", fontsize= 30)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17))

plt.axis('off')
## Number of words in the text ##

df_train["num_words"] = df_train["clean_text"].apply(lambda x: len(str(x).split()))



plt.figure(figsize=(14,8))

sns.violinplot(x='num_words', y='author', data=df_train[df_train["num_words"] <= 30])

plt.xlabel('Number of words in text', fontsize=14)

plt.ylabel('Author Name', fontsize=14)

plt.title('Number of words by author', fontsize=14)

plt.show()
df_train["num_puncts"] = df_train['text'].apply(lambda x: len([c for c in str(x) \

                                                                     if c in string.punctuation]) )

plt.figure(figsize=(14,8))

sns.violinplot(x='num_puncts', y='author', data=df_train[df_train['num_puncts'] <= 10])

plt.xlabel('Author Name', fontsize=14)

plt.ylabel('Number of punctuations by author', fontsize=14)

plt.title('Number of puntuations in text', fontsize=14)

plt.show()
from nltk.tokenize import word_tokenize, RegexpTokenizer



sample_text = df_train.text[0]



print('Word Tokenizer output: ' + str(word_tokenize(sample_text)) + "\n")



tokenizer = RegexpTokenizer(r'\w+') # Keep only words by removing punctuation

print('RegexpTokenizer output: ' + str(tokenizer.tokenize(sample_text)))
stemmer = nltk.stem.PorterStemmer()

print("The stemmed form of thinking is: {}".format(stemmer.stem("thinking")))
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

print("The lemmatized form of believes is: {}".format(lemm.lemmatize("believes")))
sentence = ["This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall."]

vectorizer = CountVectorizer(min_df=0)

sentence_transform = vectorizer.fit_transform(sentence)



print("The features are:\n {}".format(vectorizer.get_feature_names()))

print("\nThe vectorized array looks like:\n {}".format(sentence_transform.toarray()))
df_test["clean_text"] = df_test["text"].apply(lambda x: cleanup_spooky_text(x))



## TEXT-BASED FEATURES - TODO

## Based on the links from this topic: https://www.kaggle.com/c/spooky-author-identification/discussion/42925



## META-FEATURES

## Number of characters in the text

df_train["nb_chars"] = df_train["text"].apply(lambda x: len(str(x)))

df_test["nb_chars"] = df_test["text"].apply(lambda x: len(str(x)))



## Number of words in the text

df_train["nb_words"] = df_train["text"].apply(lambda x: len(str(x).split()))

df_test["nb_words"] = df_test["text"].apply(lambda x: len(str(x).split()))



## Number of relevant words in the text (stop words removed)

df_train["nb_rel_words"] = df_train["clean_text"].apply(lambda x: len(str(x).split()))

df_test["nb_rel_words"] = df_test["clean_text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text (stop words removed)

df_train["nb_uniq_words"] = df_train["clean_text"].apply(lambda x: len(set(str(x).split())))

df_test["nb_uniq_words"] = df_test["clean_text"].apply(lambda x: len(set(str(x).split())))



## Number of stopwords in the text

df_train["nb_stopwords"] = df_train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

df_test["nb_stopwords"] = df_test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text

df_train["nb_punct"] =df_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

df_test["nb_punct"] =df_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
## Prepare the data for modeling

## TODO
## TODO

def bewitched_XGB(train_X, train_y, test_X, test_y):

    

    xgb_params = {

    'seed': 20171106,

    'colsample_bytree': 0.8,

    'silent': 1,

    'subsample': .85,

    'eta': 0.04,

    'objective': 'multi:softprob',

    'num_parallel_tree': 7,

    'max_depth': 5,

    'min_child_weight': 10,

    'nthread': 22,

    'num_class': 3,

    'eval_metric': 'mlogloss',

    }

    

    num_rounds = 1000

    xgtrain = xgb.DMatrix(train_X, label=train_y)