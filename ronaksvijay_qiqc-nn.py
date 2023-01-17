%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



import os

import re

import csv

import string

import emoji

import regex

import eli5

import pickle

import gensim

import spacy

import gc

import random

from tqdm import tqdm



import numpy as np

import pandas as pd

import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from scipy.sparse import hstack

from IPython.display import Image

from prettytable import PrettyTable



from tqdm import tqdm_notebook

tqdm_notebook().pandas()



from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

from nltk.stem.lancaster import LancasterStemmer

from nltk.util import ngrams



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import f1_score, classification_report

from sklearn.calibration import CalibratedClassifierCV



import xgboost as xgb



from keras import backend as K

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential

from keras.layers import Dense, Input, Embedding, Dropout, Conv1D, Bidirectional, GlobalMaxPool1D, Concatenate

from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, BatchNormalization, SpatialDropout1D

from keras.layers import CuDNNLSTM, CuDNNGRU, Lambda

from keras import initializers, regularizers, constraints

from keras.callbacks import ModelCheckpoint

from keras.utils import to_categorical

from keras.optimizers import *



from keras.layers import *

from keras.models import *

from keras.callbacks import *

from keras.optimizers import *



from keras.engine.topology import Layer

from keras.callbacks import *
import nltk

# nltk.download('wordnet')

# nltk.download('punkt')
def seed_everything(seed=1234):

  random.seed(seed)

  os.environ['PYTHONHASHSEED'] = str(seed)

  np.random.seed(seed)



seed_everything()
path = '../input/'



df_train = pd.read_csv(path+'train.csv')

df_test = pd.read_csv(path+'test.csv')



print("Number of data points in training data:", df_train.shape[0])

print("Number of data points in test data:", df_test.shape[0])
df_train.head()
# https://www.kaggle.com/canming/ensemble-mean-iii-64-36

def clean_tag(text):

  if '[math]' in text:

    text = re.sub('\[math\].*?math\]', '[formula]', text) #replacing with [formuala]

    

  if 'http' in text or 'www' in text:

    text = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '[url]', text) #replacing with [url]

  return text
## https://www.kaggle.com/theoviel/improve-your-score-with-text-preprocessing-v2 

# clean word contractions



contraction_mapping = {"We'd": "We had", "That'd": "That had", "AREN'T": "Are not", "HADN'T": "Had not", "Could've": "Could have", "LeT's": "Let us", "How'll": "How will", "They'll": "They will", "DOESN'T": "Does not", "HE'S": "He has", "O'Clock": "Of the clock", "Who'll": "Who will", "What'S": "What is", "Ain't": "Am not", "WEREN'T": "Were not", "Y'all": "You all", "Y'ALL": "You all", "Here's": "Here is", "It'd": "It had", "Should've": "Should have", "I'M": "I am", "ISN'T": "Is not", "Would've": "Would have", "He'll": "He will", "DON'T": "Do not", "She'd": "She had", "WOULDN'T": "Would not", "She'll": "She will", "IT's": "It is", "There'd": "There had", "It'll": "It will", "You'll": "You will", "He'd": "He had", "What'll": "What will", "Ma'am": "Madam", "CAN'T": "Can not", "THAT'S": "That is", "You've": "You have", "She's": "She is", "Weren't": "Were not", "They've": "They have", "Couldn't": "Could not", "When's": "When is", "Haven't": "Have not", "We'll": "We will", "That's": "That is", "We're": "We are", "They're": "They' are", "You'd": "You would", "How'd": "How did", "What're": "What are", "Hasn't": "Has not", "Wasn't": "Was not", "Won't": "Will not", "There's": "There is", "Didn't": "Did not", "Doesn't": "Does not", "You're": "You are", "He's": "He is", "SO's": "So is", "We've": "We have", "Who's": "Who is", "Wouldn't": "Would not", "Why's": "Why is", "WHO's": "Who is", "Let's": "Let us", "How's": "How is", "Can't": "Can not", "Where's": "Where is", "They'd": "They had", "Don't": "Do not", "Shouldn't":"Should not", "Aren't":"Are not", "ain't": "is not", "What's": "What is", "It's": "It is", "Isn't":"Is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }



def clean_contractions(text):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    

    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])

    return text
# https://www.kaggle.com/canming/ensemble-mean-iii-64-36



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', 

        '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 

        '█', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '¬', '░', '¡', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', 

        '—', '‹', '─', '▒', '：', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '⋅', '‘', '∞', 

        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', '≤', '‡', '√', '◄', '━', 

        '⇒', '▶', '≥', '╝', '♡', '◊', '。', '✈', '≡', '☺', '✔', '↵', '≈', '✓', '♣', '☎', '℃', '◦', '└', '‟', '～', '！', '○', 

        '◆', '№', '♠', '▌', '✿', '▸', '⁄', '□', '❖', '✦', '．', '÷', '｜', '┃', '／', '￥', '╠', '↩', '✭', '▐', '☼', '☻', '┐', 

        '├', '«', '∼', '┌', '℉', '☮', '฿', '≦', '♬', '✧', '〉', '－', '⌂', '✖', '･', '◕', '※', '‖', '◀', '‰', '\x97', '↺', 

        '∆', '┘', '┬', '╬', '،', '⌘', '⊂', '＞', '〈', '⎙', '？', '☠', '⇐', '▫', '∗', '∈', '≠', '♀', '♔', '˚', '℗', '┗', '＊', 

        '┼', '❀', '＆', '∩', '♂', '‿', '∑', '‣', '➜', '┛', '⇓', '☯', '⊖', '☀', '┳', '；', '∇', '⇑', '✰', '◇', '♯', '☞', '´', 

        '↔', '┏', '｡', '◘', '∂', '✌', '♭', '┣', '┴', '┓', '✨', '\xa0', '˜', '❥', '┫', '℠', '✒', '［', '∫', '\x93', '≧', '］', 

        '\x94', '∀', '♛', '\x96', '∨', '◎', '↻', '⇩', '＜', '≫', '✩', '✪', '♕', '؟', '₤', '☛', '╮', '␊', '＋', '┈', '％', 

        '╋', '▽', '⇨', '┻', '⊗', '￡', '।', '▂', '✯', '▇', '＿', '➤', '✞', '＝', '▷', '△', '◙', '▅', '✝', '∧', '␉', '☭', 

        '┊', '╯', '☾', '➔', '∴', '\x92', '▃', '↳', '＾', '׳', '➢', '╭', '➡', '＠', '⊙', '☢', '˝', '∏', '„', '∥', '❝', '☐', 

        '▆', '╱', '⋙', '๏', '☁', '⇔', '▔', '\x91', '➚', '◡', '╰', '\x85', '♢', '˙', '۞', '✘', '✮', '☑', '⋆', 'ⓘ', '❒', 

        '☣', '✉', '⌊', '➠', '∣', '❑', '◢', 'ⓒ', '\x80', '〒', '∕', '▮', '⦿', '✫', '✚', '⋯', '♩', '☂', '❞', '‗', '܂', '☜', 

        '‾', '✜', '╲', '∘', '⟩', '＼', '⟨', '·', '✗', '♚', '∅', 'ⓔ', '◣', '͡', '‛', '❦', '◠', '✄', '❄', '∃', '␣', '≪', '｢', 

        '≅', '◯', '☽', '∎', '｣', '❧', '̅', 'ⓐ', '↘', '⚓', '▣', '˘', '∪', '⇢', '✍', '⊥', '＃', '⎯', '↠', '۩', '☰', '◥', 

        '⊆', '✽', '⚡', '↪', '❁', '☹', '◼', '☃', '◤', '❏', 'ⓢ', '⊱', '➝', '̣', '✡', '∠', '｀', '▴', '┤', '∝', '♏', 'ⓐ', 

        '✎', ';', '␤', '＇', '❣', '✂', '✤', 'ⓞ', '☪', '✴', '⌒', '˛', '♒', '＄', '✶', '▻', 'ⓔ', '◌', '◈', '❚', '❂', '￦', 

        '◉', '╜', '̃', '✱', '╖', '❉', 'ⓡ', '↗', 'ⓣ', '♻', '➽', '׀', '✲', '✬', '☉', '▉', '≒', '☥', '⌐', '♨', '✕', 'ⓝ', 

        '⊰', '❘', '＂', '⇧', '̵', '➪', '▁', '▏', '⊃', 'ⓛ', '‚', '♰', '́', '✏', '⏑', '̶', 'ⓢ', '⩾', '￠', '❍', '≃', '⋰', '♋', 

        '､', '̂', '❋', '✳', 'ⓤ', '╤', '▕', '⌣', '✸', '℮', '⁺', '▨', '╨', 'ⓥ', '♈', '❃', '☝', '✻', '⊇', '≻', '♘', '♞', 

        '◂', '✟', '⌠', '✠', '☚', '✥', '❊', 'ⓒ', '⌈', '❅', 'ⓡ', '♧', 'ⓞ', '▭', '❱', 'ⓣ', '∟', '☕', '♺', '∵', '⍝', 'ⓑ', 

        '✵', '✣', '٭', '♆', 'ⓘ', '∶', '⚜', '◞', '்', '✹', '➥', '↕', '̳', '∷', '✋', '➧', '∋', '̿', 'ͧ', '┅', '⥤', '⬆', '⋱', 

        '☄', '↖', '⋮', '۔', '♌', 'ⓛ', '╕', '♓', '❯', '♍', '▋', '✺', '⭐', '✾', '♊', '➣', '▿', 'ⓑ', '♉', '⏠', '◾', '▹', 

        '⩽', '↦', '╥', '⍵', '⌋', '։', '➨', '∮', '⇥', 'ⓗ', 'ⓓ', '⁻', '⎝', '⌥', '⌉', '◔', '◑', '✼', '♎', '♐', '╪', '⊚', 

        '☒', '⇤', 'ⓜ', '⎠', '◐', '⚠', '╞', '◗', '⎕', 'ⓨ', '☟', 'ⓟ', '♟', '❈', '↬', 'ⓓ', '◻', '♮', '❙', '♤', '∉', '؛', 

        '⁂', 'ⓝ', '־', '♑', '╫', '╓', '╳', '⬅', '☔', '☸', '┄', '╧', '׃', '⎢', '❆', '⋄', '⚫', '̏', '☏', '➞', '͂', '␙', 

        'ⓤ', '◟', '̊', '⚐', '✙', '↙', '̾', '℘', '✷', '⍺', '❌', '⊢', '▵', '✅', 'ⓖ', '☨', '▰', '╡', 'ⓜ', '☤', '∽', '╘', 

        '˹', '↨', '♙', '⬇', '♱', '⌡', '⠀', '╛', '❕', '┉', 'ⓟ', '̀', '♖', 'ⓚ', '┆', '⎜', '◜', '⚾', '⤴', '✇', '╟', '⎛', 

        '☩', '➲', '➟', 'ⓥ', 'ⓗ', '⏝', '◃', '╢', '↯', '✆', '˃', '⍴', '❇', '⚽', '╒', '̸', '♜', '☓', '➳', '⇄', '☬', '⚑', 

        '✐', '⌃', '◅', '▢', '❐', '∊', '☈', '॥', '⎮', '▩', 'ு', '⊹', '‵', '␔', '☊', '➸', '̌', '☿', '⇉', '⊳', '╙', 'ⓦ', 

        '⇣', '｛', '̄', '↝', '⎟', '▍', '❗', '״', '΄', '▞', '◁', '⛄', '⇝', '⎪', '♁', '⇠', '☇', '✊', 'ி', '｝', '⭕', '➘', 

        '⁀', '☙', '❛', '❓', '⟲', '⇀', '≲', 'ⓕ', '⎥', '\u06dd', 'ͤ', '₋', '̱', '̎', '♝', '≳', '▙', '➭', '܀', 'ⓖ', '⇛', '▊', 

        '⇗', '̷', '⇱', '℅', 'ⓧ', '⚛', '̐', '̕', '⇌', '␀', '≌', 'ⓦ', '⊤', '̓', '☦', 'ⓕ', '▜', '➙', 'ⓨ', '⌨', '◮', '☷', 

        '◍', 'ⓚ', '≔', '⏩', '⍳', '℞', '┋', '˻', '▚', '≺', 'ْ', '▟', '➻', '̪', '⏪', '̉', '⎞', '┇', '⍟', '⇪', '▎', '⇦', '␝', 

        '⤷', '≖', '⟶', '♗', '̴', '♄', 'ͨ', '̈', '❜', '̡', '▛', '✁', '➩', 'ா', '˂', '↥', '⏎', '⎷', '̲', '➖', '↲', '⩵', '̗', '❢', 

        '≎', '⚔', '⇇', '̑', '⊿', '̖', '☍', '➹', '⥊', '⁁', '✢']



def clean_punct(x):

  for punct in puncts:

    if punct in x:

      x = x.replace(punct, f' {punct} ')

  return x
def data_cleaning(x):

  x = clean_tag(x)

  x = clean_contractions(x)

  x = clean_punct(x)

  return x
df_train['preprocessed_question_text'] = df_train['question_text'].progress_map(lambda x: data_cleaning(x))

df_test['preprocessed_question_text'] = df_test['question_text'].progress_map(lambda x: data_cleaning(x))
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec

# Spelling correction for words.



spell_model = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')

words = spell_model.index2word

w_rank = {}

for i, word in enumerate(words): 

  w_rank[word] = i # Dictionary with key: word, value: idx

WORDS = w_rank



def words(text): return re.findall(r'\w+', text.lower())



def P(word):

  "Probability of `word`"

  # use inverse of rank as proxy

  # returns 0 if the word isn't in the dictionary

  return - WORDS.get(word, 0)



def correction(word):

  "Most probable spelling correction for word."

  return max(candidates(word), key=P)



def candidates(word):

  "Generate possible spelling corrections for word."

  return (known([word]) or known(edits1(word)) or [word])



def known(words):

  "The subset of `words` that appear in the dictionary of WORDS."

  return set(w for w in words if w in WORDS)



def edits1(word):

  "All edits that are one edit away from `word`."

  letters = 'abcdefghijklmnopqrstuvwxyz'

  splits = [(word[:i], word[i:])        for i in range(len(word) + 1)]

  deletes = [L + R[1:]                  for L, R in splits if R]  

  transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]  

  replaces = [L + c + R[1:]             for L, R in splits if R for c in letters]

  inserts = [L + c + R                  for L, R in splits for c in letters]

  return set(deletes + transposes + replaces + inserts)



def edits2(word):

  "All edits that are two edits away from `word`"

  return (e2 for e1 in edits1(word) for e2 in edits1(e1))



def singlify(word):

  return "".join([letter for i, letter in enumerate(word) if i == 0 or letter != word[i-1]])
# Initializing

ps = PorterStemmer()

lc = LancasterStemmer()

sb = SnowballStemmer('english')
##

# https://www.kaggle.com/wowfattie/3rd-place

# Method for loading word vectors of given embedding file.



def load_embeddings(word_dict, lemma_dict, embedding):

    

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    

    if embedding == 'glove':

      EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt' #../input/embeddings/

      embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    else: #para

      EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt' #../input/embeddings/

      embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



    embed_size = 300

    nb_words = len(word_dict)+1

    embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)

    unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.

    

    # Trying all combinations(Lower, upper, stemmer, lemm, spell corrector) for finding more word embeddings.

    for key in tqdm_notebook(word_dict):

        word = key

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.lower()         #Lower

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.upper()         #Upper

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = key.capitalize()    #Capitalize 

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = ps.stem(key)        #PorterStemmer

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lc.stem(key)        #LancasterStemmer

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = sb.stem(key)        #SnowballStemmer

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        word = lemma_dict[key]     #Lemmatizer

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[word_dict[key]] = embedding_vector

            continue

        if len(key) > 1:

            word = correction(key) #Spell Corrector

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:

                embedding_matrix[word_dict[key]] = embedding_vector

                continue

        embedding_matrix[word_dict[key]] = unknown_vector                    

    return embedding_matrix, nb_words 
text_list = pd.concat([df_train['preprocessed_question_text'], df_test['preprocessed_question_text']])

num_train_data = df_train.shape[0]
# Using Spacy tokenizer to tokenize words in data.

from tqdm import tqdm_notebook



start_time = time.time()

print("Spacy NLP ...")

nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])

nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

word_dict = {}

word_index = 1

lemma_dict = {}

docs = nlp.pipe(text_list, n_threads = 2)

word_sequences = []

for doc in tqdm_notebook(docs):

    word_seq = []

    for token in doc:

        if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):

            word_dict[token.text] = word_index

            word_index += 1

            lemma_dict[token.text] = token.lemma_

        if token.pos_ is not "PUNCT":

            word_seq.append(word_dict[token.text])

    word_sequences.append(word_seq)

del docs

gc.collect()

train_word_sequences = word_sequences[:num_train_data]

test_word_sequences = word_sequences[num_train_data:]

print("--- %s seconds ---" % (time.time() - start_time))
# config

embed_size = 300

max_features = len(word_dict)+1

maxlen = 72
# pad the sequences

X = pad_sequences(train_word_sequences, maxlen=maxlen)

X_test = pad_sequences(test_word_sequences, maxlen=maxlen)



# target values

Y = df_train['target'].values
print("Loading embedding matrix ...")

embedding_matrix_glove, nb_words = load_embeddings(word_dict, lemma_dict, 'glove')

embedding_matrix_para, nb_words = load_embeddings(word_dict, lemma_dict, 'para')

embedding_matrix = np.mean((1.28*embedding_matrix_glove, 0.72*embedding_matrix_para), axis=0)



del embedding_matrix_glove, embedding_matrix_para

gc.collect()

embedding_matrix.shape
class AttentionWeightedAverage(Layer):

    """

    Computes a weighted average of the different channels across timesteps.

    Uses 1 parameter pr. channel to compute the attention value for a single timestep.

    """



    def __init__(self, return_attention=False, **kwargs):

        self.init = initializers.RandomUniform(seed=10000)

        self.supports_masking = True

        self.return_attention = return_attention

        super(AttentionWeightedAverage, self).__init__(** kwargs)



    def build(self, input_shape):

        self.input_spec = [InputSpec(ndim=3)]

        assert len(input_shape) == 3



        self.W = self.add_weight(shape=(input_shape[2], 1),

                                 name='{}_W'.format(self.name),

                                 initializer=self.init)

        self.trainable_weights = [self.W]

        super(AttentionWeightedAverage, self).build(input_shape)



    def call(self, x, mask=None):

        # computes a probability distribution over the timesteps

        # uses 'max trick' for numerical stability

        # reshape is done to avoid issue with Tensorflow

        # and 1-dimensional weights

        logits = K.dot(x, self.W)

        x_shape = K.shape(x)

        logits = K.reshape(logits, (x_shape[0], x_shape[1]))

        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))



        # masked timesteps have zero weight

        if mask is not None:

            mask = K.cast(mask, K.floatx())

            ai = ai * mask

        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())

        weighted_input = x * K.expand_dims(att_weights)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:

            return [result, att_weights]

        return result



    def get_output_shape_for(self, input_shape):

        return self.compute_output_shape(input_shape)



    def compute_output_shape(self, input_shape):

        output_len = input_shape[2]

        if self.return_attention:

            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]

        return (input_shape[0], output_len)



    def compute_mask(self, input, input_mask=None):

        if isinstance(input_mask, list):

            return [None] * len(input_mask)

        else:

            return None

class AdamW(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)

                 epsilon=1e-8, decay=0., **kwargs):

        super(AdamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (2/4)

        self.epsilon = epsilon

        self.initial_decay = decay



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]

        wd = self.wd # decoupled weight decay (3/4)



        lr = self.lr

        if self.initial_decay > 0:

            lr *= (1. / (1. + self.decay * K.cast(self.iterations,

                                                  K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t)))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p # decoupled weight decay (4/4)



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'weight_decay': float(K.get_value(self.wd)),

                  'epsilon': self.epsilon}

        base_config = super(AdamW, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
# Model 1

def LSTM_GRU(spatialdropout=0.2, rnn_units=64, weight_decay=0.07):

  K.clear_session()

  x_input = Input(shape=(maxlen,))

  

  emb = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False, name='Embedding')(x_input)



  x = SpatialDropout1D(spatialdropout, seed=1024)(emb)



  rnn1 = Bidirectional(CuDNNLSTM(rnn_units, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed=111100), recurrent_initializer=initializers.Orthogonal(gain=1.0, seed=123000)))(x)



  rnn2 = Bidirectional(CuDNNGRU(rnn_units, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed=111100), recurrent_initializer=initializers.Orthogonal(gain=1.0, seed=123000)))(rnn1)



  x = concatenate([rnn1, rnn2])

  x = GlobalMaxPooling1D()(x)

  x_output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=111100))(x)



  model = Model(inputs=x_input, outputs=x_output)

  model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=weight_decay),)

  return model
# Model 2

def poolRNN(spatialdropout=0.2, gru_units=128, weight_decay=0.04):

  K.clear_session()

  x_input = Input(shape=(maxlen,))

  

  emb = Embedding(max_features, embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False,)(x_input)



  x = SpatialDropout1D(spatialdropout, seed=1024)(emb)



  rnn1 = Bidirectional(CuDNNGRU(gru_units, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed=111100), recurrent_initializer=initializers.Orthogonal(gain=1.0, seed=123000)))(x)



  last = Lambda(lambda t: t[:, -1], name='last')(rnn1)

  maxpool = GlobalMaxPooling1D()(rnn1)

  attn  = AttentionWeightedAverage()(rnn1)

  average = GlobalAveragePooling1D()(rnn1)



  c = concatenate([last, maxpool, attn], axis=1)

  c = Reshape((3, -1))(c)

  c = Lambda(lambda x: K.sum(x, axis=1))(c)

  x = BatchNormalization()(c)

  x = Dense(200, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=111000))(x)

  x = Dropout(0.2, seed=1024)(x)

  x = BatchNormalization()(x)

  x_output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=111000))(x)



  model = Model(inputs=x_input, outputs=x_output)

  model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=weight_decay))

  return model
# Model 3

def BiLSTM_CNN(spatialdropout=0.2, rnn_units=128, filters=[100, 80, 30, 12], weight_decay=0.10):

  K.clear_session()

  x_input = Input(shape=(maxlen,))

  

  emb = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False, name='Embedding')(x_input)



  x = SpatialDropout1D(rate=spatialdropout, seed=10000)(emb)



  rnn = Bidirectional(CuDNNLSTM(rnn_units, return_sequences=True, kernel_initializer=initializers.glorot_uniform(seed=123000), recurrent_initializer=initializers.Orthogonal(gain=1.0, seed=123000)))(x)

  

  x1 = Conv1D(filters=filters[0], activation='relu', kernel_size=1, padding='same', kernel_initializer=initializers.glorot_uniform(seed=110000))(rnn)



  x2 = Conv1D(filters=filters[1], activation='relu', kernel_size=1, padding='same', kernel_initializer=initializers.glorot_uniform(seed=120000))(rnn)



  x3 = Conv1D(filters=filters[2], activation='relu', kernel_size=1, padding='same', kernel_initializer=initializers.glorot_uniform(seed=130000))(rnn)



  x4 = Conv1D(filters=filters[3], activation='relu', kernel_size=1, padding='same', kernel_initializer=initializers.glorot_uniform(seed=140000))(rnn)



  x1 = GlobalMaxPooling1D()(x1)

  x2 = GlobalMaxPooling1D()(x2)

  x3 = GlobalMaxPooling1D()(x3)

  x4 = GlobalMaxPooling1D()(x4)



  c = concatenate([x1, x2, x3, x4])

  x = Dense(200, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=111000))(c)

  x = Dropout(0.2, seed=10000)(x)

  x = BatchNormalization()(x)

  x_output = Dense(1, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=110000))(x)

 

  model = Model(inputs=x_input, outputs=x_output)

  model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=weight_decay))

  return model
def f1_smart(y_true, y_pred):

  args = np.argsort(y_pred)

  tp = y_true.sum()

  fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)

  res_idx = np.argmax(fs)

  return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2
%%time

# KFold - Model Ensemble

kfold = StratifiedKFold(n_splits=7, random_state=10, shuffle=True)

bestscore = []

bestloss = []

y_test = np.zeros((X_test.shape[0], ))

oof = np.zeros((X.shape[0], ))

epochs = [8, 8, 7, 6]

val_list = []

for i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):

  val_list += list(valid_index)

  print('FOLD%s'%(i+1))

  X_train, X_val, Y_train, Y_val = X[train_index], X[valid_index], Y[train_index], Y[valid_index]

  filepath = "weights_best.h5"

  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0, verbose=0)

  callbacks = [checkpoint, reduce_lr]

  if i == 0:

    model = LSTM_GRU(spatialdropout=0.20, rnn_units=64, weight_decay=0.07)

    print('LSTM_GRU(spatialdropout=0.20, rnn_units=64, weight_decay=0.07)')

  elif i == 1:

    model = poolRNN(spatialdropout=0.20, gru_units=128, weight_decay=0.04)

    print('poolRNN(spatialdropout=0.20, gru_units=64, weight_decay=0.04)')

  elif i == 2:

    model = BiLSTM_CNN(spatialdropout=0.2, rnn_units=128, filters=[100, 90, 30, 12], weight_decay=0.10)

    print('BiLSTM_CNN(spatialdropout=0.2, rnn_units=128, filters=[100, 90, 30, 12], weight_decay=0.10)')



  model.fit(X_train, Y_train, batch_size=512, epochs=epochs[i], validation_data = (X_val, Y_val), verbose=0, callbacks=callbacks)

  print("Train log_loss:%s"%model.history.history['loss'])

  print("val log_loss:%s"%model.history.history['val_loss'])

  y_pred = model.predict([X_val], batch_size=1024, verbose=2)

  y_test += np.squeeze(model.predict([X_test], batch_size=1024, verbose=2))/3

  oof[valid_index] = np.squeeze(y_pred)

  f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))

  print('Optimal F1: {:.4f} at threshold: {:.4f}\n'.format(f1, threshold))

  bestscore.append(threshold)

  if i ==2: break
f1, threshold = f1_smart(np.squeeze(Y[val_list]), np.squeeze(oof[val_list]))

print('Optimal F1: {:.4f} at threshold: {:.4f}'.format(f1, threshold))
# kaggle submision

y_test = y_test.reshape((-1, 1))

submission = df_test[['qid']].copy()

submission['prediction'] = (y_test>threshold).astype(int)

submission.to_csv('submission.csv', index=False)