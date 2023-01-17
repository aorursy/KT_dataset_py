import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from itertools import islice

import textwrap

from sklearn.model_selection import train_test_split





wrapper = textwrap.TextWrapper(initial_indent='', width=70,

                               subsequent_indent=' '*3)



import nltk

nltk.download('wordnet')

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')

nltk.download('vader_lexicon')
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



text_column = 'text'

label = 'author'
train_df.head()
import string



from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD



import xgboost as xgb

from sklearn.metrics import log_loss

from sklearn.model_selection import KFold

from sklearn.naive_bayes import MultinomialNB

from nltk.sentiment.vader import SentimentIntensityAnalyzer



english_stopwords = set(stopwords.words("english"))
# import stemmer and lemmatizer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer



# define LemmaCountVectorizer which will find all unique word and their occurrences

porter_stemmer = PorterStemmer()

lemm = WordNetLemmatizer()



class LemmaCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(LemmaCountVectorizer, self).build_analyzer()

        return lambda doc: (porter_stemmer.stem(lemm.lemmatize(w)) for w in analyzer(doc))



# Seperate text by author

eap_text = list(train_df[train_df['author'] == 'EAP'][text_column].values)

hpl_text = list(train_df[train_df['author'] == 'HPL'][text_column].values)

mws_text = list(train_df[train_df['author'] == 'MWS'][text_column].values)



author_text_dict = dict(zip([0,1,2], [eap_text,hpl_text, mws_text]))



# apply LemmaCountVectorizer to the full text

full_text = eap_text + mws_text + hpl_text



full_tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 

                                       min_df=2,

                                       stop_words='english',

                                       decode_error='ignore')

full_tf = full_tf_vectorizer.fit_transform(full_text)

full_feature_names = full_tf_vectorizer.get_feature_names()

# full_count_vec = np.asarray(full_tf.sum(axis=0)).ravel()

# full_zipped = list(zip(full_feature_names, full_count_vec))



# create dataframe to store the word frequency for each author (initialized with zeros):

# rows - represents the authors

# columns represents each unique word (after stemming and lemmatizing)

# so each cell in our new dataframe, represents how many occurrences each author has for each word in all of his lines

author_word_freq_df = pd.DataFrame(0.0, index=[0,1,2], columns=full_feature_names)



# dictionary contains each word count for each author

author_wordcount_dict = {}



for author, text in author_text_dict.items():

  tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 

                                       min_df=2,

                                       stop_words='english',

                                       decode_error='ignore')

  tf = tf_vectorizer.fit_transform(text)

  feature_names = tf_vectorizer.get_feature_names()

  count_vec = np.asarray(tf.sum(axis=0)).ravel()

  zipped = list(zip(feature_names, count_vec))

  author_wordcount_dict[author] = zipped
# fill the word frequency dataframe by each author word count

for author, zipped in author_wordcount_dict.items():

  for word, count in zipped:

    author_word_freq_df[word.lower()][author] = count



# transpose the dataframe, now the rows contains the unique words, columns contains authors

transposed_freq_df = author_word_freq_df.T



# Create new columns:

# 1. 0_count,1_count, 2_count - represeting the word count difference between the authors

# 2. 0_ratio,1_ratio, 2_ratio - represeting the word count ratio between the authors



transposed_freq_df['0_count'] = transposed_freq_df[0] - transposed_freq_df[1] - transposed_freq_df[2]

transposed_freq_df['1_count'] = transposed_freq_df[1] - transposed_freq_df[0] - transposed_freq_df[2]

transposed_freq_df['2_count'] = transposed_freq_df[2] - transposed_freq_df[0] - transposed_freq_df[1]



# epsilon is used to prevent division by zero, when a certain word is used by only one author

epsilon = 1 

transposed_freq_df['0_ratio'] = (transposed_freq_df[0] + epsilon) /(transposed_freq_df[1] + transposed_freq_df[2] + epsilon)

transposed_freq_df['1_ratio'] = (transposed_freq_df[1] + epsilon) /(transposed_freq_df[0] + transposed_freq_df[2] + epsilon)

transposed_freq_df['2_ratio'] = (transposed_freq_df[2] + epsilon) /(transposed_freq_df[0] + transposed_freq_df[1] + epsilon)



transposed_freq_df.sort_values(by='0_ratio', ascending=False)



def calc_count_score(text, author):

  word_list = word_tokenize(text)

  score = 0

    

  for word in word_list:

    lemm_word = porter_stemmer.stem(lemm.lemmatize(word))    

    

    if lemm_word in transposed_freq_df.index:

      score = score + transposed_freq_df[str(author)+'_count'][lemm_word]

    

  score = score / len(word_list)

  return score



def calc_ratio_score(text, author):

  word_list = word_tokenize(text)

  score = 1

    

  for word in word_list:

    lemm_word = porter_stemmer.stem(lemm.lemmatize(word))    

    

    if lemm_word in transposed_freq_df.index:

      

      score = score * transposed_freq_df[str(author)+'_ratio'][lemm_word]

    

  return score / len(word_list)


train_df['eap_freq_count_score'] = train_df[text_column].apply(lambda row: calc_count_score(row, 0))

train_df['hpl_freq_count_score'] = train_df[text_column].apply(lambda row: calc_count_score(row, 1))

train_df['mws_freq_count_score'] = train_df[text_column].apply(lambda row: calc_count_score(row, 2))



train_df['eap_freq_ratio_score'] = train_df[text_column].apply(lambda row: calc_ratio_score(row, 0))

train_df['hpl_freq_ratio_score'] = train_df[text_column].apply(lambda row: calc_ratio_score(row, 1))

train_df['mws_freq_ratio_score'] = train_df[text_column].apply(lambda row: calc_ratio_score(row, 2))



test_df['eap_freq_count_score'] = test_df[text_column].apply(lambda row: calc_count_score(row, 0))

test_df['hpl_freq_count_score'] = test_df[text_column].apply(lambda row: calc_count_score(row, 1))

test_df['mws_freq_count_score'] = test_df[text_column].apply(lambda row: calc_count_score(row, 2))



test_df['eap_freq_ratio_score'] = test_df[text_column].apply(lambda row: calc_ratio_score(row, 0))

test_df['hpl_freq_ratio_score'] = test_df[text_column].apply(lambda row: calc_ratio_score(row, 1))

test_df['mws_freq_ratio_score'] = test_df[text_column].apply(lambda row: calc_ratio_score(row, 2))
import string

def unique_word_fraction(text):

    """function to calculate the fraction of unique words on total words of the text"""

    text_splited = text.split(' ')

    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

    text_splited = [s for s in text_splited if s]

    word_count = text_splited.__len__()

    unique_count = list(set(text_splited)).__len__()

    return (unique_count/word_count)





eng_stopwords = set(stopwords.words("english"))

def stopwords_count(text):

    """ Number of stopwords fraction in a text"""

    text = text.lower()

    text_splited = text.split(' ')

    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

    text_splited = [s for s in text_splited if s]

    word_count = text_splited.__len__()

    stopwords_count = len([w for w in text_splited if w in eng_stopwords])

    return (stopwords_count/word_count)





def punctuations_fraction(text):

    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """

    char_count = len(text)

    punctuation_count = len([c for c in text if c in string.punctuation])

    return (punctuation_count/char_count)





def char_count(text):

    """function to return number of chracters """

    return len(text)



def fraction_noun(text):

    """function to give us fraction of noun over total words """

    text_splited = text.split(' ')

    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

    text_splited = [s for s in text_splited if s]

    word_count = text_splited.__len__()

    pos_list = nltk.pos_tag(text_splited)

    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])

    return (noun_count/word_count)



def fraction_adj(text):

    """function to give us fraction of adjectives over total words in given text"""

    text_splited = text.split(' ')

    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

    text_splited = [s for s in text_splited if s]

    word_count = text_splited.__len__()

    pos_list = nltk.pos_tag(text_splited)

    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])

    return (adj_count/word_count)



def fraction_verbs(text):

    """function to give us fraction of verbs over total words in given text"""

    text_splited = text.split(' ')

    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]

    text_splited = [s for s in text_splited if s]

    word_count = text_splited.__len__()

    pos_list = nltk.pos_tag(text_splited)

    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])

    return (verbs_count/word_count)
# create new features for the traing and test dataframes

for df in [train_df, test_df]:  

  

  # Number of characters in the text

  df['char_count'] = df[text_column].apply(lambda text: len(text))



  # Number of words in the text

  df['word_count'] = df[text_column].apply(lambda text: len(word_tokenize(text)))



  # Number of unique words in the text

  df['unique_word_count'] = df[text_column].apply(lambda text: len(set(word_tokenize(text))))



  # Number of stopwords

  df['stopwords_count'] = df[text_column].apply(lambda text: len([word for word in word_tokenize(str(text).lower()) if word in english_stopwords]))

  

  # Number of punctuations

  df['punctuations_count'] = df[text_column].apply(lambda text: len([word for word in word_tokenize(text) if word in string.punctuation]))



  # Number of capitalized words

  df['capitalized_count'] = df[text_column].apply(lambda text: len([word for word in word_tokenize(text) if word.istitle()]))

  

  # Number of uppercase words

  df["upper_words_count"] = df[text_column].apply(lambda text: len([word for word in word_tokenize(text) if word.isupper()]))



  # mean word length

  df['mean_word_len'] = df[text_column].apply(lambda text: np.mean([len(word) for word in word_tokenize(text)]))



  # the ratio of the punctuation

  df['punctuations_fraction'] = df[text_column].apply(lambda row: punctuations_fraction(row))

  

  # Ratio of nouns

  df['fraction_noun'] = df[text_column].apply(lambda row: fraction_noun(row))

  

  # Ratio of adjective

  df['fraction_adj'] = df[text_column].apply(lambda row: fraction_adj(row))

  

  # Ratio of verbs

  df['fraction_verbs'] = df[text_column].apply(lambda row: fraction_verbs(row))

  

  # Add feature for each stopword in the nltk English stopwords list, representing how many instances each stopword has in the current sentence

  for idx, curr_stopword in enumerate(english_stopwords):

    df[curr_stopword] = df[text_column].apply(lambda text: len([word for word in word_tokenize(str(text).lower()) if word == curr_stopword]))





new_features = ['char_count', 'word_count', 'unique_word_count', 'stopwords_count', 'punctuations_count'

                , 'capitalized_count', 'mean_word_len', 'upper_words_count', 'punctuations_fraction', 'fraction_noun', 'fraction_adj', 'fraction_verbs']
# train_id = train_df['id'].values

test_id = test_df['id'].values



author_mapping_dict = {'EAP': 0, 'HPL': 1, 'MWS': 2}

cols_to_drop = ['id', 'text']

X_train = train_df.drop(cols_to_drop+['author'], axis=1)

X_test = test_df.drop(cols_to_drop, axis=1)



y_train = train_df['author'].map(author_mapping_dict)
import xgboost as xgb



xgb_clf = xgb.XGBClassifier(objective='multi:softprob',

                            colsample_bytree = 0.3,

                            learning_rate = 0.1,

                            max_depth = 3, 

                            alpha = 10,

                            n_estimators = 10, num_round=2000)

xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict_proba(X_test)
y_pred
out_df = pd.DataFrame(y_pred)

out_df.columns = ['EAP', 'HPL', 'MWS']

out_df.insert(0, 'id', test_id)

out_df.to_csv("sub_fe.csv", index=False)