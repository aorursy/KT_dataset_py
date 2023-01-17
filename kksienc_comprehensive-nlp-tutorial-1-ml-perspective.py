text ="Welcome to NLP Tutorial"
text = text.lower()
print(text)
!pip install pyspellchecker
from spellchecker import SpellChecker
def correct_spellings(text):
    spell = SpellChecker()
    corrected_words = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)
        


text = "Spelling correctin is proprly perfrmed"
text = correct_spellings(text)
print(text)
import re
text = "Correcting   double  space  text "
text = re.sub(' +', ' ', text)
print(text)
import string

text = "This! scententance, has so: many- punctuations."
text = text.translate(str.maketrans('', '', string.punctuation))
print(text)
text = 'Shall I refer this answer in www.google.com ?'
text  = re.sub(r"https?://\S+|www\.\S+", "", text )
print(text)
text ="Being no 1 team is more important or being no 3 but with fair play "
text= re.sub(r'[0-9]',' ',text)
print (text)
from nltk.corpus import stopwords 

text = "This is not the most important topic"

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    # stop_words will contain  set all english stopwords
    filtered_sentence = []   
    for word in text.split(): 
        if word not in stop_words: 
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence)

text = remove_stopwords(text)
print(text) 
 

import emoji
text = 'Python is 👍'
print(emoji.demojize(text))
text = "He was not happy with the score of team"
from nltk.corpus import wordnet
import nltk
def convert_to_antonym(sentence):
    words = nltk.word_tokenize(sentence)
    new_words = []
    temp_word = ''
    for word in words:
        antonyms = []
        if word == 'not':
            temp_word = 'not_'
        elif temp_word == 'not_':
            for syn in wordnet.synsets(word):
                for s in syn.lemmas():
                    for a in s.antonyms():
                        antonyms.append(a.name())
            if len(antonyms) >= 1:
                word = antonyms[0]
            else:
                word = temp_word + word # when antonym is not found, it will
                                    # remain not_happy
            
            temp_word = ''
        if word != 'not':
            new_words.append(word)
    return ' '.join(new_words)
    
text = convert_to_antonym(text)
print(text)   
from nltk.stem.porter import PorterStemmer
text = " David wanted to go with Alfa but Alfa went with Charli so David is going with Bravo"
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

text = stem_words(text)
print(text)
import nltk
text = "This is very good observation by you."
nltk.pos_tag(text.split()) 

import nltk
nltk.download('tagsets')
nltk.help.upenn_tagset('DT')
# import these modules 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
import nltk 
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV }

# without wordnet map it takes evey word as noun
text = "David wanted to go with Alfa but Alfa went with Charli so David is going with Bravo "
 
def lemma_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word ,wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


lemma_words(text) 

text = """
<html><head><title>The NLP story</title></head>
<body>
<p class="title"><b>The NLP story</b></p>
<p class="story">Once upon a time there were three little  techniques; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived till the next conference.</p>
<p class="story">...</p>
"""

print(text)
from bs4 import BeautifulSoup
text = BeautifulSoup(text, "html").text# HTML decoding
# for lxml  decodinf
#text = BeautifulSoup(text, "lxml").text
print(text)
def clean_text(text):
    """
        text: a string
        
        return: modified initial string
  """
    text = text.lower() # lowercase text
    text= re.sub(r'[^\w\s#]',' ',text) #Removing every thing other than space, word and hash
    text  = re.sub(r"https?://\S+|www\.\S+", "", text )
    text= re.sub(r'[0-9]',' ',text)
    #text = correct_spellings(text)
    text = convert_to_antonym(text)
    text = re.sub(' +', ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text    
    return text
import numpy as np  
import pandas as pd 
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
from IPython.display import Image
Image("/kaggle/input/images/texttonumber.jpg",  width=400)
corpus= pd.DataFrame(columns=['text'])
corpus['text']= pd.concat([train_df["text"], test_df["text"]])
from IPython.display import Image
Image("/kaggle/input/images/tokenization.png",  width=400)

from IPython.display import Image
Image("/kaggle/input/images/ngram.png",  width=400)
Image("/kaggle/input/images/countvectorization.png",  width=400)
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
# analyzer should be word/ character
#ngram_range lower and upper boundary of the range of n-values

## let's get counts for the first 5 tweets in the data
train_countvectors = count_vectorizer.fit_transform(train_df["text"])

# generating test CountVectorizer matrix
test_countvectors = count_vectorizer.transform(test_df["text"])

# converting sparce to dense vector 
print(train_countvectors.shape)
print(train_countvectors[0])
print(train_countvectors[0].todense())
Image("/kaggle/input/images/tfidf.png",  width=450)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', ngram_range=(1, 1))
# training tfidf on corpus
tfidf_vectorizer.fit(corpus['text'])
train_tfidfvectors = tfidf_vectorizer.transform(train_df['text'])
test_tfidfvectors = tfidf_vectorizer.transform(test_df['text'])

print(train_tfidfvectors.shape)
print(test_tfidfvectors.shape)
print(train_tfidfvectors.todense().shape)
print(train_tfidfvectors[0].todense())
train_countvectors.shape
from sklearn.decomposition import TruncatedSVD   
tsv = TruncatedSVD(n_components=100)
train_countvectors_svd = tsv.fit_transform(train_countvectors) 
train_tfidfvectors_svd = tsv.fit_transform(train_tfidfvectors)

print(train_countvectors_svd.shape)
print(train_tfidfvectors_svd.shape)
# base function to train madel against datasets
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn import model_selection

def text_modeling( model):
    print( "Model :"+ str(model))
    print('***** F1 Scores *******')
    scores=model_selection.cross_val_score(model, train_countvectors.toarray(), train_df["target"], cv=3, scoring="f1")
    print("CountVectorized dataset :"+str(scores.mean()))
    scores= model_selection.cross_val_score(model, train_tfidfvectors.toarray(), train_df["target"], cv=3, scoring="f1")
    print("TF-IDF Vectorized dataset :"+str(scores.mean()))
    scores = model_selection.cross_val_score(model, train_tfidfvectors_svd, train_df["target"], cv=3, scoring="f1")
    print("TF-IDF Vectorized + SVD dataset "+str(scores.mean()))
    scores = model_selection.cross_val_score(model, train_countvectors_svd, train_df["target"], cv=3, scoring="f1")
    print("CountVectorized + SVD dataset "+str(scores.mean()))

from sklearn.naive_bayes import GaussianNB
gn = GaussianNB()
text_modeling( gn)
from sklearn.naive_bayes import BernoulliNB
#Create a Gaussian Classifier
 
br = BernoulliNB()
text_modeling( br)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')
text_modeling( lr)
# from sklearn.svm import SVC
# svm = SVC()
# text_modeling(svm)
# import xgboost as xgb
# xgb_clf = xgb.XGBClassifier()
# text_modeling(xgb_clf)