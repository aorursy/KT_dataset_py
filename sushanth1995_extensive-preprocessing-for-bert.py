!pip install symspellpy

!pip install pycontractions

!pip install keras-bert
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
pd.set_option('display.max_colwidth', -1)

import matplotlib.pyplot as plt

from tqdm import tqdm

tqdm.pandas()



import re

import string



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.util import ngrams

from nltk.stem import PorterStemmer





import pkg_resources

from symspellpy.symspellpy import SymSpell

from symspellpy import SymSpell, Verbosity



#Contraction Import

from pycontractions import Contractions
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")

bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")



symspell_segmenter = SymSpell(max_dictionary_edit_distance=2, prefix_length=8)

symspell_segmenter.load_dictionary(dictionary_path, term_index=0, count_index=1)



sym_spell_misspelled = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)

sym_spell_misspelled.load_dictionary(dictionary_path, term_index=0, count_index=1)

sym_spell_misspelled.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
cont = Contractions(api_key="glove-twitter-100")

cont.load_models()
"""Let's load the data files"""

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train.head()
from keras_bert import load_vocabulary, Tokenizer, get_checkpoint_paths

from keras_bert.datasets import get_pretrained, PretrainedList

model_path = get_pretrained(PretrainedList.wwm_uncased_large)

paths = get_checkpoint_paths(model_path)

token_dict = load_vocabulary(paths.vocab)

tokenizer = Tokenizer(token_dict)
def to_lower(text):

    text = text.lower()

    return text





def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+|pic.twitter.com\S+')

    return url.sub('[url]',text)





def remove_punct(text):

    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    return text





def remove_special_ucchar(text):

    text = re.sub('&.*?;', ' ', text)

    return text





def remove_numbers(text):

    text = re.sub(r'\d+', ' ', text)

    return text





def remove_mentions(text):

    text = re.sub(r'@\w*', ' ', text)

    return text





def handle_unicode(text):

    text = text.encode('ascii', 'replace').decode('utf-8')

    return text





def remove_punctuations(text):

    text = re.sub(r'([^A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)

    return text





def remove_square_bracket(text):

    text = re.sub('\[.*?\]', ' ', text)

    return text





def remove_angular_bracket(text):

    text = re.sub('\<.*?\>+', ' ', text)

    return text





def remove_newline(text):

    text = re.sub('\n', ' ', text)

    return text





def remove_words_with_numbers(text):

    text = re.sub('\w*\d\w*', ' ', text)

    return text

    



def hashtag_to_words(text):

    hashtag_list = re.findall(r"#\w+",text)

    for hashtag in hashtag_list:

        hashtag = re.sub(r'#', '', hashtag)

        text = re.sub(hashtag, symspell_segmenter.word_segmentation(hashtag).segmented_string, text)

    text = re.sub(r'#', ' ', text)

    return text





def extra_spaces(text):

    text = text.strip()

    text = re.sub('\s+|\t+', ' ', text)

    return text



def remove_stopwords(text):

    text_tokens=word_tokenize(text)

    textop = ''

    for token in text_tokens:

        if token not in stopwords.words('english'):

            textop = textop + token + ' '

    return textop





def correct_misspelled_with_context(text):

    suggestions = sym_spell_misspelled.lookup_compound(text, max_edit_distance=2)

    text = str(suggestions[0])

    text = re.sub(r', \d', ' ', text)

    return text





def stemming_text(text):

    stemmer= PorterStemmer()

    text_tokens=word_tokenize(text)

    textop = ''

    for token in text_tokens:

        textop = textop + stemmer.stem(token) + ' '

    return textop





def lemmatization(text):

    lemmatizer=WordNetLemmatizer()

    text_tokens=word_tokenize(text)

    textop = ''

    for token in text_tokens:

        textop = textop + lemmatizer.lemmatize(token) + ' '

    return textop





def removeRepeated(tweet):

    prev = ''

    tweet_new = ''

    for c in tweet:

        caps = False

        if c.isdigit():

            tweet_new += c

            continue

        if c.isalpha() == True:

            if ord(c) >= 65 and ord(c)<=90:

                caps = True

            c = c.lower()

            if c == prev:

                count += 1

            else:

                count = 1

                prev = c

            if count >= 3:

                continue

            if caps == True:

                tweet_new += c.upper()

            else:

                tweet_new += c

        else:

            tweet_new += c

    return tweet_new





def Expand_Contractions(text):

    return list(cont.expand_texts([text]))[0]

def build_vocab(text, tokenizer=word_tokenize):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = text.apply(lambda x: tokenizer(x)).explode().value_counts().to_dict()

    return vocab
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in vocab:

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
def count_chars(text):

    new_text = text.apply(lambda x : list(x)).explode()

    return new_text.unique().shape[0]



def count_words(text):

    new_text = text.apply(lambda x : x.split(' ')).explode()

    return new_text.unique().shape[0]



def preprocess_pipeline(steps, col, df):

    new_col = df[col]

    char_count_before = 0

    word_count_before = 0

    char_count_after = 0

    word_count_after = 0

    for each_step in steps:

        char_count_before = count_chars(new_col)

        word_count_before = count_words(new_col)

        new_col = new_col.apply(each_step)

        char_count_after = count_chars(new_col)

        word_count_after = count_words(new_col)

        print("Preprocessing step: ",each_step.__name__)

        print("Unique Char Count ---> Before: %d | After: %d"%(char_count_before, char_count_after))

        print("Unique Word Count ---> Before: %d | After: %d"%(word_count_before, word_count_after))

        vocab = build_vocab(new_col,word_tokenize)

        check_coverage(vocab,token_dict)

        print()

    

    return new_col
### Define pipeline

pipeline = []



pipeline.append(handle_unicode)

pipeline.append(to_lower)

pipeline.append(remove_newline)

pipeline.append(remove_url)

pipeline.append(remove_special_ucchar)

pipeline.append(hashtag_to_words)

pipeline.append(remove_mentions)

# pipeline.append(remove_square_bracket)

# pipeline.append(remove_angular_bracket)

pipeline.append(Expand_Contractions)

# pipeline.append(remove_words_with_numbers)

# pipeline.append(remove_punctuations)

# pipeline.append(remove_punct)

pipeline.append(extra_spaces)

# pipeline.append(remove_numbers)

# pipeline.append(removeRepeated)

pipeline.append(correct_misspelled_with_context)

# pipeline.append(remove_stopwords)

# pipeline.append(stemming_text)

# pipeline.append(lemmatization)



# sentences = train["text"].progress_apply(lambda x: x.split()).values

vocab = build_vocab(train["text"],word_tokenize)

oov = check_coverage(vocab,token_dict)
keywords = train.keyword.dropna().apply(lambda x: re.sub('%20',' ',x))
vocab = build_vocab(keywords,word_tokenize)

oov = check_coverage(vocab,token_dict)
%%time

train = pd.read_csv('../input/nlp-getting-started/train.csv')





print("For Training data:")

train['processed_text'] = preprocess_pipeline(pipeline, 'text', train)

train.head()
print(train.loc[12])
%%time

test = pd.read_csv('../input/nlp-getting-started/test.csv')



print("For Testing data:")

test['processed_text'] = preprocess_pipeline(pipeline, 'text', test)

test.head()
chars = train['processed_text'].apply(lambda x : list(x)).explode()

chars.unique()
print(train['text'].iloc[1031])

print(train['processed_text'].iloc[1031])
ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train.loc[train['id'].isin(ids_with_target_error),'target'] = 0
u, idx = np.unique(train['processed_text'], return_index=True)

train = train.iloc[idx]
tweet_len = train['processed_text'].apply(len)

print(tweet_len.max())
tweet_len = test['processed_text'].apply(len)

print(tweet_len.max())
train.to_csv('processed train.csv', index=False)

test.to_csv('processed test.csv', index=False)