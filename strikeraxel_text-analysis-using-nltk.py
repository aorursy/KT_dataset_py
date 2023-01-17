# Input data files are available in the "../input/" directory.

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# install nltk lib for processing text

!pip install nltk

import nltk
# Load txt corpus to expirement

from nltk.corpus import gutenberg, webtext

txt_corpus = gutenberg.raw('austen-emma.txt')

txt_web = webtext.raw('overheard.txt')
# Convert raw txt into sentences



# Using in-built english tokenizer

from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer

txt_sentences = sent_tokenize(txt_corpus)

txt_sentences[:5]



# Loading custom sentence tokenizer

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

cus_sentences = tokenizer.tokenize(txt_corpus)

cus_sentences[:5]



# train with whole corpus so that it can detect boundries

ps_t = PunktSentenceTokenizer(txt_web)

ps_sentences = ps_t.tokenize(txt_web)

ps_sentences[:5]



#NOTE: Other tokenizer are also avialable like line_tokenize
# Convert sentence to words



## Using available tokenizer

from nltk.tokenize import SpaceTokenizer, TreebankWordTokenizer, WhitespaceTokenizer

tw_t = TreebankWordTokenizer()

tw_t.tokenize(txt_sentences[0])



# Creating custom tokenizer

from nltk.tokenize.api import TokenizerI

class MyWordTokenizer(TokenizerI): 

    

    def tokenize(self,string):

        return string.split(" ")

    

cs_t = MyWordTokenizer()

cs_t.tokenize(txt_sentences[0])
# Example of stemming

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

print(stemmer.stem('cooking'))

print(stemmer.stem('believes'))

# NOTE: more classes of stemmer are available LancasterStemmer, RegexpStemmer and SnowballStemmer. Do use whats necessary.



# Example of lemmatizing

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('cookbooks'))

print(lemmatizer.lemmatize('believes'))

# NOTE: default POS lemma is noun if you want cooking to be cook specify the lemma for verb

print(lemmatizer.lemmatize('cooking'))

print(lemmatizer.lemmatize('cooking', pos='v'))



# Replace won't -> will not, can't -> can not, etc.

import re

replacement_patterns = [

    (r'won\'t', 'will not'),

    (r'can\'t', 'cannot'),

    (r'i\'m', 'i am'),

    (r'ain\'t', 'is not')

 ]

class RegexpReplacer(object):

    def __init__(self, patterns=replacement_patterns):

        self.patterns = [ ( re.compile(regx), repl ) for regx, repl in patterns]

    

    def replace(self, text):

        tmp = text

        for regx, repl in self.patterns:

            tmp = re.sub(regx, repl, tmp)

        return tmp

    

replacer = RegexpReplacer()

replacer.replace("i can't do this")
# Text or Corpus operations

from nltk.text import Text

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

my_string = "Two plus two is four, minus one that's three — quick maths. Every day man's on the block. Smoke trees. See your girl in the park, that girl is an uckers. When the thing went quack quack quack, your men were ducking! Hold tight Asznee, my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."



# Step 1: Lowercase the string

my_string = my_string.lower()



# Step 2: Convert sentence into words 

words = word_tokenize(my_string)



# Step 3: Remove stop words from corpus

stop_w = set(stopwords.words('english'))

words = [ word for word in words if word not in stop_w]



# Step 4: Understand word collection

txt = Text(words)

## Get Vocabulary with word frequency

txt.vocab()

txt.plot(20) # Frequent 20 words