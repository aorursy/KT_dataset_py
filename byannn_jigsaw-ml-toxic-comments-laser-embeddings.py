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
import os, warnings, pickle, gc, re, string



from tqdm.notebook import tqdm

tqdm.pandas()
# HYPERPARAMETERS



MAX_LEN = 220

MAX_FEATURES = 100000

EMBED_SIZE = 600



BATCH_SIZE = 128

N_EPOCHS = 5



LEARNING_RATE = 8e-4



# We will concatenate Crawl and GloVe embeddings



CRAWL_EMB_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

GLOVE_EMB_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
print('Loading train sets...')

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")



train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)

])



del train1, train2



print('Loading validation sets...')

valid = pd.read_csv('/kaggle/input/val-en-df/validation_en.csv')



print('Loading test sets...')

test = pd.read_csv('/kaggle/input/test-en-df/test_en.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",

                 "didn't": "did not", "doesn't": "does not", "don't": "do not",

                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                 "he'd": "he would", "he'll": "he will", "he's": "he is",

                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",

                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",

                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",

                 "she'd": "she would", "she'll": "she will", "she's": "she is",

                 "shouldn't": "should not", "that's": "that is", "there's": "there is",

                 "they'd": "they would", "they'll": "they will", "they're": "they are",

                 "they've": "they have", "we'd": "we would", "we're": "we are",

                 "weren't": "were not", "we've": "we have", "what'll": "what will",

                 "what're": "what are", "what's": "what is", "what've": "what have",

                 "where's": "where is", "who'd": "who would", "who'll": "who will",

                 "who're": "who are", "who's": "who is", "who've": "who have",

                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",

                 "you'll": "you will", "you're": "you are", "you've": "you have",

                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}



def _get_misspell(misspell_dict):

    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    return misspell_dict, misspell_re



def replace_typical_misspell(text):

    misspellings, misspellings_re = _get_misspell(misspell_dict)



    def replace(match):

        return misspellings[match.group(0)]



    return misspellings_re.sub(replace, text)



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',

          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',

          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',

          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',

          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',

          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',

          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']



def clean_text(x):

    x = str(x)

    for punct in puncts + list(string.punctuation):

        if punct in x:

            x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    return re.sub(r'\d+', ' ', x)



def preprocess(train, valid, test, tfms):

    for tfm in tfms:

        print(tfm.__name__)

        train['comment_text'] = train['comment_text'].progress_apply(tfm)

        valid['comment_text_en'] = valid['comment_text_en'].progress_apply(tfm)

        test['content'] = test['content'].progress_apply(tfm)

    

    return train, valid, test
tfms = [replace_typical_misspell, clean_text, clean_numbers]

train, valid, test = preprocess(train, valid, test, tfms)
comments = list(train['comment_text']) + list(valid['comment_text_en']) + list(test['content_en'])



print(comments[0])

print(len(comments))
!pip install laserembeddings
!python -m laserembeddings download-models
from laserembeddings import Laser



laser = Laser()



embedding_matrix = laser.embed_sentences(comments, lang='en')
np.save("jigsaw-ml_laser.npy", embedding_matrix)