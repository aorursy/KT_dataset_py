# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import io

import re

import os

import nltk

import time

import math

import scipy

import string

import zipfile

import operator

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm

import matplotlib.pyplot as pt

from collections import defaultdict

from gensim.models import KeyedVectors, Word2Vec, fasttext

import warnings

warnings.filterwarnings('ignore')
quora_train = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

quora_test = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")

quora_train.head(1)
file_path = "/kaggle/input/quora-insincere-questions-classification/embeddings.zip"



def Embeddings(file_path,file):

    '''

    parameter : file_path(embedding file), 

                file = name of the file

    return : embedding_matrix(dictionary)

    ''' 

    embeddings_glove = dict()

    with zipfile.ZipFile(file_path,'r') as zf:

        if file == "glove":

            with io.TextIOWrapper(zf.open("glove.840B.300d/glove.840B.300d.txt"), encoding="utf-8") as f:

                for line in tqdm(f):

                    values=line.split(' ') # ".split(' ')" only for glove-840b-300d; for all other files, ".split()" works

                    word=values[0]

                    vectors=np.asarray(values[1:],'float32')

                    embeddings_glove[word]=vectors

            return embeddings_glove

        

        elif file == "word2vec":

            embeddings_glove = KeyedVectors.load_word2vec_format(zf.open("GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"),binary=True)

            return embeddings_glove



        elif file == "paragram":

            path = zf.extract("paragram_300_sl999/paragram_300_sl999.txt")

            def get_coefs(word,*arr): 

                return word, np.asarray(arr, dtype='float32')

            embeddings_glove = dict(get_coefs(*w.split(" ")) for w in open(path, encoding='latin'))

            return embeddings_glove



        elif file=="fasttext":

            path = zf.extract("wiki-news-300d-1M/wiki-news-300d-1M.vec")

            def get_coefs(word,*arr): 

                return word, np.asarray(arr, dtype='float32')

            embeddings_glove = dict(get_coefs(*w.split(" ")) for w in open(path, encoding='latin'))

            return embeddings_glove
word2vecModel = Embeddings(file_path,file="word2vec")
#Function1

symbols = dict({"√":" sqrt ","π":" pi ","α":" alpha ","θ":" theta ","∞":" infinity ","∝":" proportional to ","sinx":" sin x ",

           "cosx":" cos x ", "tanx":" tan x ","cotx":" cot x ", "secx":" sec x ", "cosecx":" cosec x ", "£":" pound ", "β":" beta ", 

           "σ": " theta ", "∆":" delta ","μ":" mu ",'∫': " integration ", "ρ":" rho ", "λ":" lambda ","∩":" intersection ",

           "Δ":" delta ", "φ":" phi ", "℃":" centigrade ","≠":" does not equal to ","Ω":" omega ","∑":" summation ","∪":" union ",

           "ψ":" psi ", "Γ":" gamma ","⇒":" implies ","∈":" is an element of ", "≡":" is congruent to ",

           "≈":" is approximately equal to ", "~":" is distributed as ","≅":" is isomorphic to ","⩽":" is less than or equal to ",

           "≥":" is greater than or equal to ","⇐":" is implied by ","⇔":" is equivalent to ", "∉":" is not an element of ",

           "∅" : " empty set ", "∛":" cbrt ","÷":" division ","㏒":" log ","∇":" del ","⊆":" is a subset of ","±":" plus–minus ",

           "⊂":" is a proper subset of ","€":" euro ","㏑":" ln ","₹":" rupee ","∀":" there exists "})



def special_chars(text,symbols):

    for p in symbols:

        if p in text:

            text = text.replace(p, symbols[p])

            text = re.sub("\s{2}"," ",text)

    return text



quora_train['question_text_word2vec'] = quora_train['question_text'].apply(lambda x: special_chars(x,symbols))

quora_test['question_text_word2vec'] = quora_test['question_text'].apply(lambda x: special_chars(x,symbols))

print("special_chars : Done")





#Function2

word2vecRem = "ುِೋ̡̿∠្▒ௌ…″̲・̢ూ⃗（̙ి̒؟？ि∈̻͒。਼☉⊥☁ா(ಿ̵ّ̥ു́‏✅ូ̌ோ”‐̂︡ੁ͔ี̋∆̾/ॣ̮⊨़̍̊∪َ͑☝ุ℅ौ〇ी̚ःំ̸்͚͛‑《ँీ̫ാֿ「❓＝’ા∡ٌី＾ుั̖?̅͡ះ∇<:〖}ា̈́͋\ਂ♏✌，्|ਾ≅㏒;ం／⚧＄ْ⧽̳ி❤[ு!̼ិো➡⦁♨﻿­）̴᠌⁡่ಾ；⎠∴͉ை：−∅͊˜⊂̎⎞,⟩„∼♭≱▾-ে̔‛̟̀ాొ̞∨》​.ើ̷̱¦∝⁠“്̣̓⬇ੀ⌚⋯͐́͂ं͝∩̐͗˂⁻ెী'̀া્̜͌⧼̩͈͕̆–ં্ৃਿ)̪ो◌้̹̉」‌।️∗⟨͎ੰ∘̛、′—∛͘̦ॄ㏑়\়়⎛̰̄∑♀͠＞ি̘´̯̈ͅ✏︠☹े≡₱ूി͇！{‎－ुॢ⎝್̭ៃ₊̬،̗̽∖̺₹్ृॉ̧̝ाู͖̤⊆‘͜]ै̃̑̓̕ା∧͆〗"



def remove_punct(text,punToBeRemove):

    translate_table = dict((ord(char), None) for char in punToBeRemove) 

    #Loop to iterate 

    for idx,val in enumerate(text.values):

        val = val.translate(translate_table)

        text.values[idx] = val.strip()

    return text

quora_train['question_text_word2vec'] = remove_punct(quora_train['question_text_word2vec'], word2vecRem)

quora_test['question_text_word2vec'] = remove_punct(quora_test['question_text_word2vec'], word2vecRem)

print("remove_punct : Done")



#Function3

contractions = {"'aight": 'alright', "ain't": 'am not', "amn't": 'am not', "aren't": 'are not', "can't": 'can not',

"'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "daren't": 

'dare not', "daresn't": 'dare not', "dasn't": 'dare not', "didn't": 'did not', "doesn't": 'does not', 

"don't": 'do not', 'dunno': "don't know", "d'ye": 'do you', "e'er": 'ever', "everybody's": 'everybody is', 

"everyone's": 'everyone is', 'finna': 'fixing to', "g'day": 'good day', 'gimme': 'give me', "giv'n": 'given', 

'gonna': 'going to', "gon't": 'go not', 'gotta': 'got to', "hadn't": 'had not', "had've": 'had have', 

"hasn't": 'has not', "haven't": 'have not', "he'd": 'he had', "he'll": 'he will', "he's": 'he is', 

"he've": 'he have', "how'd": 'how did', 'howdy': 'how do you do', "how'll": 'how will', "how're": 'how are', 

"how's": 'how is', "I'd": 'I had', "I'd've": 'I would have', "I'll": 'I will', "I'm": 'I am', 

"I'm'a": 'I am about to', "I'm'o": 'I am going to', 'innit': 'is it not', "I've": 'I have', "isn't": 'is not', 

"it'd ": 'it would', "it'll": 'it will', "it's ": 'it is', 'iunno': "I don't know", "let's": 'let us', 

"ma'am": 'madam', "mayn't": 'may not', "may've": 'may have', 'methinks': 'me thinks', "mightn't": 'might not', 

"might've": 'might have', "mustn't": 'must not', "mustn't've": 'must not have', "must've": 'must have', 

"needn't": 'need not', 'nal': 'and all', "ne'er": 'never', "o'clock": 'of the clock', "o'er": 'over',

"ol'": 'old', "oughtn't": 'ought not', "'s": 'is', "shalln't": 'shall not', "shan't": 'shall not', 

"she'd": 'she would', "she'll": 'she will', "she's": 'she is', "should've": 'should have', 

"shouldn't": 'should not', "shouldn't've": 'should not have', "somebody's": 'somebody has', 

"someone's": 'someone has', "something's": 'something has', "so're": 'so are', "that'll": 'that will', 

"that're": 'that are', "that's": 'that is', "that'd": 'that would', "there'd": 'there would', 

"there'll": 'there will', "there're": 'there are', "there's": 'there is', "these're": 'these are', 

"they've": 'they have', "this's": 'this is', "those're": 'those are', "those've": 'those have', "'tis": 'it is', 

"to've": 'to have', "'twas": 'it was', 'wanna': 'want to', "wasn't": 'was not', "we'd": 'we would', 

"we'd've": 'we would have', "we'll": 'we will', "we're": 'we are', "we've": 'we have', "weren't": 'were not', 

"what'd": 'what did', "what'll": 'what will', "what're": 'what are', "what's": 'what does', "what've": 'what have',

"when's": 'when is', "where'd": 'where did', "where'll": 'where will', "where're": 'where are',

"where's": 'where is',"where've": 'where have', "which'd": 'which would', "which'll": 'which will', 

"which're": 'which are',"which's": 'which is', "which've": 'which have', "who'd": 'who would',

"who'd've": 'who would have', "who'll": 'who will', "who're": 'who are', "who'ves": 'who is', "who'": 'who have',

"why'd": 'why did', "why're": 'why are', "why's": 'why does', "willn't": 'will not', "won't": 'will not',

'wonnot': 'will not', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have',

"y'all": 'you all', "y'all'd've": 'you all would have', "y'all'd'n've": 'you all would not have',

"y'all're": 'you all are', "cause":"because","have't":"have not","cann't":"can not","ain't":"am not",

"you'd": 'you would', "you'll": 'you will', "you're": 'you are', "you've": 'you have', 'cannot': 'can not', 

'wont': 'will not', "You'": 'Am not', "Ain'": 'Am not', "Amn'": 'Am not', "Aren'": 'Are not',

"Can'": 'Because', "Could'": 'Could have', "Couldn'": 'Could not have', "Daren'": 'Dare not', 

"Daresn'": 'Dare not', "Dasn'": 'Dare not', "Didn'": 'Did not', "Doesn'": 'Does not', "Don'": "Don't know", 

"D'": 'Do you', "E'": 'Ever', "Everybody'": 'Everybody is', "Everyone'": 'Fixing to', "G'": 'Give me', 

"Giv'": 'Going to', "Gon'": 'Got to', "Hadn'": 'Had not', "Had'": 'Had have', "Hasn'": 'Has not', 

"Haven'": 'Have not', "He'": 'He have', "How'": 'How is', "I'": 'I have', "Isn'": 'Is not', "It'": "I don't know", 

"Let'": 'Let us', "Ma'": 'Madam', "Mayn'": 'May not', "May'": 'Me thinks', "Mightn'": 'Might not', 

"Might'": 'Might have', "Mustn'": 'Must not have', "Must'": 'Must have', "Needn'": 'And all', "Ne'": 'Never',

"O'": 'Old', "Oughtn'": 'Is', "Shalln'": 'Shall not', "Shan'": 'Shall not', "She'": 'She is', 

"Should'": 'Should have', "Shouldn'": 'Should not have', "Somebody'": 'Somebody has', "Someone'": 'Someone has', 

"Something'": 'Something has', "So'": 'So are', "That'": 'That would', "There'": 'There is',

"They'": 'They have', "This'": 'This is', "Those'": 'It is', "To'": 'Want to', "Wasn'": 'Was not',

"Weren'": 'Were not', "What'": 'What have', "When'": 'When is', "Where'": 'Where have', "Which'": 'Which have', 

"Who'": 'Who have', "Why'": 'Why does', "Willn'": 'Will not', "Won'": 'Will not', "Would'": 'Would have',

"Wouldn'": 'Would not have', "Y'": 'You all are',"What's":"What is","What're":"What are","what's":"what is",

"what're":"what are", "Who're":"Who are", "your're":"you are","you're":"you are", "You're":"You are",

"We're":"We are", "These'": 'These have', "we're":"we are","Why're":"Why are","How're":"How are ",

"how're ":"how are ","they're ":"they are ", "befo're":"before","'re ":" are ",'don"t ':"do not", 

"Won't ":"Will not ","could't":"could not", "would't":"would not", "We'": 'We have',"Hasn't":"Has not",

"n't":"not", 'who"s':"who is"}



def decontraction(text,contractions):

    #Loop to iterate 

    for idx,val in enumerate(text.values):

        val = ' '.join(word.replace(word,contractions[word]) if word in contractions

                    else word for word in val.split())

        #generic one

        val = re.sub(r"\'s", " ", val);val = re.sub(r"\''s", " ", val);val = re.sub(r"\"s", " ", val)

        val = re.sub(r"n\'t", " not ", val);val = re.sub(r"n\''t", " not ", val);val = re.sub(r"n\"t", " not ", val)

        val = re.sub(r"\'re ", " are ", val);val = re.sub(r"\'d ", " would", val);val = re.sub(r"\''d ", " would", val)

        val = re.sub(r"\"d ", " would", val);val = re.sub(r"\'ll ", " will", val);val = re.sub(r"\''ll ", " will", val)

        val = re.sub(r"\"ll ", " will", val);val = re.sub(r"\'ve ", " have", val);val = re.sub(r"\''ve ", " have", val)

        val = re.sub(r"\"ve ", " have", val);val = re.sub(r"\'m ", " am", val);val = re.sub(r"\''m "," am", val)

        val = re.sub(r"\"m "," am", val);val = re.sub("\s{2}"," ",val)

        text.values[idx] = val.strip() 

    return text

quora_train['question_text_word2vec'] = decontraction(quora_train['question_text_word2vec'],contractions)

quora_test['question_text_word2vec'] = decontraction(quora_test['question_text_word2vec'],contractions)

print("decontraction : Done")





#Function4

word2vecKeep = ['☺','"', '§', '≈', '¯', '@', '│', '±', '⅓', '†', '\x9d', '=', '¥', '√', '$', '>', '¾', '¶', '^', 

                '∫', '\uf02d', '£', '≠', '≤', '△', '™', '⋅', '₩', '×', '\x8f', '✔', '◦', '#', '∞', '®', '«', '⅔', '%',

                '¢', '≥', '♣', '\u202a', '`', '↓', '½', '~', '•', '°', '₦', '∀', '¬', '÷', '·', '*', '↑', '¨', '»',

                '_', '&', '℃', '¼', '\x8d', '⇒', '∂', '✓', '˚', '￼', '♡', '‰', '\uf0d8', '©', '\u202c', '¡',

                '+', '¸', '̶', '¿', '→', '\x92', '€']



def spacing_of_chars(text,characters_list):

    for char in characters_list:

        if char in text:

            text = text.replace(char," "+char+" ")

            text = re.sub("\s+"," ",text)

    return text

quora_train['question_text_word2vec'] = quora_train['question_text_word2vec'].apply(lambda x: spacing_of_chars(x,word2vecKeep))

quora_test['question_text_word2vec'] = quora_test['question_text_word2vec'].apply(lambda x: spacing_of_chars(x,word2vecKeep))

print("spacing_of_chars : Done")





#Function5

replace_word = dict({"Quorans":"Quora", "Brexit":"Britain exit", "cryptocurrencies":"cryptocurrency", "Blockchain":"blockchain", 

                "demonetisation":"demonetization", "Pokémon":"Pokemon", "Qoura":"Quora", "fiancé":"fiance",

                "Cryptocurrency":"cryptocurrency", "x²":"x squeare", "Quoras":"Quora","Whst":"What", "²":"square", 

                "Demonetization":"demonetization", "brexit":"Britain exit", "São":"Sao","genderfluid":"Gender fluid", 

                "Howcan":"How can", "undergraduation":"under graduation", "Whydo":"Why do", "à":"a",

                "chapterwise":"chapter wise", "Cryptocurrencies":"cryptocurrency", "fiancée":"fiance", 

                "wouldwin":"would win", "Nanodegree":"nano degree","nanodegree":"nano degree", "blockchains":"blockchain", 

                "clichés":"cliche", "Erdoğan":"Erdogan", "Beyoncé":"Beyonce", "fullform":"full form",

                "Atatürk":"Ataturk", "Whyis":"Why is","amfrom":"am from", "2k17":"2017", "demonitization":"demonetization",

                "cliché":"cliche", "Montréal":"Montreal", "thé":"the", "am17":"am 17", "willhappen":"will happen",

                "³":"cube", "whatapp":"whatsapp", "ε":"epsilon", "whatsaap":"whatsapp",'Σ':"summation","Quorians":"Quora users",

                "cryptocurreny":"cryptocurrency", "mastuburation":"masturbation","Whatre":"What are", "Whatdo":"What do",

                "δ":"delta","oversmart":"over smart","¹":"one"})



def word_correction(text,set_words):

    for idx,val in enumerate(text.values):

        val = ' '.join(word.replace(word,set_words[word]) if word in set_words else word for word in val.split())

        text.values[idx] = val

    return text

quora_train['question_text_word2vec'] = word_correction(quora_train['question_text_word2vec'], replace_word)

quora_test['question_text_word2vec'] = word_correction(quora_test['question_text_word2vec'], replace_word)

print("word_correction : Done")





#Function6

def question_text_vocab(text):

    freq_dict = defaultdict(int)

    total_sent = text.apply(lambda x: x.split()).values

    for sent in total_sent:

        for token in sent:

            freq_dict[token] += 1

    return freq_dict



text_vocab_tr = question_text_vocab(quora_train['question_text_word2vec'])

text_vocab_te = question_text_vocab(quora_test['question_text_word2vec'])

print("question_text_vocab : Done")





#Function7

def coverage(vocab, embeddings_index,print_statement=False):

    #Initializing values

    known_words = defaultdict(int)

    unknown_words = defaultdict(int)

    knownWordsVal = 0

    unknownWordsVal = 0

    #iterating words

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            knownWordsVal += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            unknownWordsVal += vocab[word]

            pass

    

    if print_statement == True:

        print('Found {:.2%} of words in the embedding of the question text vocab'

           .format(len(known_words) / len(vocab)))

        print('Found {:.2%} of the words in the question text vocab'.format(knownWordsVal / (knownWordsVal + unknownWordsVal)))

    else:

        pass

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words



oov_glove_tr = coverage(text_vocab_tr, word2vecModel)

oov_glove_te = coverage(text_vocab_te, word2vecModel)

print("coverage : Done")





#Function8

def oob_vocab(vocab, embeddings):

    freq_dict = defaultdict()

    for word in vocab:

        if word[0].istitle() == True:

            if word[0].lower() in embeddings:

                freq_dict[word[0]]= word[0].lower()

        elif word[0].islower() == True:

            if word[0].title() in embeddings:

                freq_dict[word[0]]= word[0].title()

    return freq_dict



word_dict_tr = oob_vocab(oov_glove_tr,word2vecModel)

word_dict_te = oob_vocab(oov_glove_te,word2vecModel)

print("coverage : Done")





#Function9

def numbers(text):

    text = re.sub('[0-9]{6}', '######', text)

    text = re.sub('[0-9]{5}', '#####', text)

    text = re.sub('[0-9]{4}', '####', text)

    text = re.sub('[0-9]{3}', '###', text)

    text = re.sub('[0-9]{2}', '##', text)

    return text

quora_train['question_text_word2vec'] = quora_train['question_text_word2vec'].apply(lambda x: numbers(x))

quora_test['question_text_word2vec'] = quora_test['question_text_word2vec'].apply(lambda x: numbers(x))

print("numbers : Done")





#Function10

correct_word_word2vec = {"doesnt":"does not", "didnt":"did not", "isnt":"is not","shouldnt":"should not","hasnt":"has not",

                    "wasnt":"was not", "btech":"bachelor in technology", "Isnt":"Is not", "Quorans":"Quora users",

                    "Shouldnt":"Should not", "Doesnt":"Does not", "Quoras":"Quora", "Qoura":"Quora",

                    "Btech":"bachelor in technology", "nonMuslims":"non Muslims", "Pokémon":"Pokemon", "Didnt":"Did not",

                    "nonMuslim":"not Muslim", "demonetisation": "demonetization","Wasnt":"Was not","sinx":"sin x", "cosx":"cos x",

                    "tanx":"tan x", "cotx":"cot x", "secx":"sec x","cosecx":"cosec x","demonetisation":"demonetization",

                    "infty":"infinity", "AfricanAmericans":"African - Americans","cryptocurrency":"crypto currency",

                    "cryptocurrencies":"crypto currency"}



quora_train['question_text_word2vec'] = word_correction(quora_train['question_text_word2vec'], correct_word_word2vec)

quora_test['question_text_word2vec'] = word_correction(quora_test['question_text_word2vec'], correct_word_word2vec)

print("word_correction : Done")
TextVocabTrain = question_text_vocab(quora_train['question_text_word2vec'])

print("Training Dataset")

OOVGloveTrain = coverage(TextVocabTrain, word2vecModel, print_statement=True)
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.backend import clear_session, maximum

from tensorflow.keras.callbacks import EarlyStopping 

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras import Model, initializers, regularizers, constraints, optimizers, layers

from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding

from tensorflow.keras.layers import Concatenate, LSTM, Activation, GRU, Reshape, Lambda, Multiply
from sklearn.model_selection import train_test_split

y = quora_train['target']

X = quora_train.drop(columns = ['target'])



X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.20, stratify=y)

print("The shape of train,cv & test dataset before conversion into vector")

print(X_train.shape, y_train.shape)

print(X_cv.shape, y_cv.shape)

print(quora_test.shape)
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train['question_text_word2vec'])

encoded_docs_train = tokenizer.texts_to_sequences(X_train['question_text_word2vec'])

encoded_docs_cv = tokenizer.texts_to_sequences(X_cv['question_text_word2vec'])

encoded_docs_test = tokenizer.texts_to_sequences(quora_test['question_text_word2vec'])
maxlength = 75 #maximum length of a sentence to be padded



Xtrain = pad_sequences(encoded_docs_train, maxlen = maxlength, padding='post')

Xcv = pad_sequences(encoded_docs_cv, maxlen = maxlength, padding='post')

Xtest = pad_sequences(encoded_docs_test, maxlen = maxlength, padding='post')
vocab_size = len(tokenizer.word_index) + 1



#getting embedding matrix of 300 dim

glove_words = set(word2vecModel.index2word)

embedding_matrix_word2vec= np.zeros((vocab_size, 300))

for word, idx in tokenizer.word_index.items():

    if word in glove_words:

        embedding_vector = word2vecModel[word]

        embedding_matrix_word2vec[idx] = embedding_vector



print('The shape of emdedding matrix is: ',embedding_matrix_word2vec.shape)
from tensorflow.keras.utils import to_categorical

ytrain = to_categorical(y_train, 2)

ycv = to_categorical(y_cv, 2)
#Callback function

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix, f1_score

from tensorflow.keras.callbacks import Callback



class accuracy_value(Callback):



    def __init__(self,training_data,validation_data):

        self.X_train = training_data[0]

        self.y_train = training_data[1]

        self.X_val = validation_data[0]

        self.y_val = validation_data[1]



    def on_train_begin(self, logs = {}):

        self.f1_scores = []

        self.precisions = []

        self.recalls = []



    def on_epoch_end(self, epoch, logs = {}):

        #F1 Score

        y_predicted = np.asarray(self.model.predict(self.X_val)).round()

        f1_val = f1_score(self.y_val,y_predicted,average=None)

        self.f1_scores.append(f1_val)



        print(" - f1 score : {}".format(np.round(f1_val,4)))



f1Score = accuracy_value(training_data=(Xtrain, ytrain), validation_data=(Xcv, ycv))
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
clear_session()



inputs = Input(shape=(maxlength,), dtype='int32', name='Input_Text') #input layer

Embedding_Layer = Embedding(vocab_size, 300, weights = [embedding_matrix_word2vec], input_length = maxlength, trainable=False)(inputs) #embedding layer

#convolution layer

ConvLayer1 = Conv1D(30,5,padding = 'same',activation='relu',strides=2)(Embedding_Layer)

ConvLayer2 = Conv1D(30,5,padding = 'same',activation='relu',strides=2)(Embedding_Layer) 

ConvLayer3 = Conv1D(30,5,padding = 'same',activation='relu',strides=2)(Embedding_Layer)

convLayer4 = Conv1D(30,5,padding = 'same',activation='relu',strides=2)(Embedding_Layer)

mergedLayer = Concatenate()([ConvLayer1, ConvLayer2, ConvLayer3, convLayer4])



LSTMLayer = LSTM(64, kernel_initializer='glorot_normal', return_sequences=True)(mergedLayer) #lstm layer

flat = Flatten()(LSTMLayer) #flattening the features

dense = Dense(4096, activation='relu')(flat)

output = Dense(2, activation='sigmoid')(dense)

model = Model(inputs,output)



model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"]) #compile the model

model.fit(Xtrain, ytrain, batch_size=512, verbose=1, epochs=10, validation_data=(Xcv,ycv), shuffle=True, callbacks=[f1Score, earlyStopping]) #fitting the model



threshold = dict()

ypred = model.predict(Xcv, batch_size=512,verbose=1)

from sklearn import metrics

for thresh in np.arange(0.1, 0.501, 0.05):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, round(list(metrics.f1_score(ycv, (ypred>thresh).astype(int), average=None))[1],3)))

    threshold[thresh] = round(list(metrics.f1_score(ycv, (ypred>thresh).astype(int), average=None))[1],3)



print("\nThe best threshold is:",max(threshold, key=threshold.get))



#printing classification report using the best threshold

ypredicted = (ypred>max(threshold, key=threshold.get)).astype(int)

print("Classification Report:\n",metrics.classification_report(ycv,ypredicted))



#predicting test data

ypredict = list()

ypred = model.predict(Xtest, batch_size=512,verbose=1)

for i in ypred:

    ypredict.append((i[1]>max(threshold, key=threshold.get)).astype(int))



#creating dataframe

df_test = pd.DataFrame({"qid":quora_test["qid"].values})

df_test['prediction'] = ypredict

print("Quora Test Output:\n",df_test['prediction'].value_counts())
df_test.to_csv('submission.csv', index=False)