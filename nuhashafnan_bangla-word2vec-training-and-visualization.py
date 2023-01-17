import pandas as pd
import numpy as np
import re
from re import sub
import multiprocessing
from unidecode import unidecode
import os
import glob
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from time import time 
from collections import defaultdict
#turn on internet option in kernel
!pip install bangla-stemmer
from bangla_stemmer.stemmer import stemmer

def text_to_word_list(text):
    text = text.split()
    return text

def replace_strings(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\u00C0-\u017F"          #latin
                           u"\u2000-\u206F"          #generalPunctuations
                               
                           "]+", flags=re.UNICODE)
    english_pattern=re.compile('[a-zA-Z0-9]+', flags=re.I)
    #latin_pattern=re.compile('[A-Za-z\u00C0-\u00D6\u00D8-\u00f6\u00f8-\u00ff\s]*',)
    
    text=emoji_pattern.sub(r'', text)
    text=english_pattern.sub(r'', text)

    return text

def stopwordRemoval(text):    
    x=str(text)
    l=x.split()

    stm=[elem for elem in l if elem not in stop]
    
    out=' '.join(stm)
    
    return str(out)

def remove_punctuations(my_str):
    # define punctuation
    punctuations = '''````£|¢|Ñ+-*/=EROero৳০১২৩৪৫৬৭৮৯012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰⚽️✌�￰৷￰'''
    
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char

    # display the unpunctuated string
    return no_punct



def joining(text):
    out=' '.join(text)
    return out

def preprocessing(text):
    out=remove_punctuations(replace_strings(text))
    return out


def Stemming(text):
    
    x=str(text)
    l=x.split()

    stmr = stemmer.BanglaStemmer()
    stm = stmr.stem(l)

    out=' '.join(stm)
    
    return str(out)

df =pd.read_csv('/kaggle/input/corpus/main_dataset_v3.csv')
data1 =pd.read_excel('/kaggle/input/bangla-stopwords/stopwords_bangla.xlsx')
stop = data1['words'].tolist()

df = df[(df['Sentence'].str.len()<100)&(df['Sentence'].str.len()>1)]
df['Sentence'].apply(lambda x: len(str(x))).plot(kind='hist');
df['Sentence'] = df.Sentence.apply(lambda x: preprocessing(str(x)))
#df['Sentence'] = df.Sentence.apply(lambda x: Stemming(str(x)))
df['Sentence'] = df.Sentence.apply(lambda x: stopwordRemoval(str(x)))
df['Sentence'].apply(lambda x: len(str(x))).plot(kind='hist');
df.reset_index(drop=True, inplace=True)
df['Sentence'] = df.Sentence.apply(lambda x: text_to_word_list(str(x)))
word2vecinput = [row for row in df.Sentence]

model = Word2Vec(word2vecinput, size=400, window=20, min_count=5,sg=0,negative=3,workers=multiprocessing.cpu_count()-1)

print(model.wv.most_similar("মা", topn=5))
print(model.wv.most_similar("খুলনা", topn=5))
print(model.wv.most_similar("রোজা", topn=5))
print(model.wv.most_similar("অপরাধ", topn=5))
print(model.wv.similarity('ঋণ', 'ব্যাংক'))
print(model.wv.doesnt_match("বার্সেলোনা ফুটবল গোলকিপার রাজনীতি".split()))

model.wv.save_word2vec_format('corpus')
#python -m gensim.scripts.word2vec2tensor --input model_name --output model_name
#python -m gensim.scripts.word2vec2tensor --input model_name --output model_name
!python -m gensim.scripts.word2vec2tensor --input corpus --output corpus