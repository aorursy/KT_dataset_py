%ls -l
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import string
import re    
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD

stoplist = set(stopwords.words("english"))
%matplotlib inline
train = pd.read_csv('../input/WK7525train.csv')
test = pd.read_csv('../input/WK7525test.csv')
train.head()
train['X_input'][1]
replacement_patterns = [  
    (r'won\'t', 'will not'),  
    (r'can\'t', 'cannot'),  
    (r'i\'m', 'i am'),  
    (r'ain\'t', 'is not'),  
    (r'(\w+)\'ll', '\g<1> will'),  
    (r'(\w+)n\'t', '\g<1> not'),  
    (r'(\w+)\'ve', '\g<1> have'),  
    (r'(\w+)\'s', '\g<1> is'),  
    (r'(\w+)\'re', '\g<1> are'),  
    (r'(\w+)\'d', '\g<1> would')
]

class RegexpReplacer(object):  
    def __init__(self, patterns=replacement_patterns):    
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]  
        
    def replace(self, text):    
        s = text    
        for (pattern, repl) in self.patterns:      
            s = re.sub(pattern, repl, s)    
        return s

from nltk.corpus import wordnet

class AntonymReplacer(object):
    
    def replace(self, word, pos=None):
        antonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            return None
        
    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words

from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn

stoplist = set(stopwords.words("english"))

class Remove_Noise(object):
    
    def __init__(self,stop_word = stoplist):
        self.stop_word = stoplist
    
    def noise_rm(self,doc):
        doc = re.sub('[#$%^&\',:()*+/<=>@[\\]^_``{|}~]',' ',doc)
        doc = re.sub('[0-9]+',' ',doc)
        doc = re.sub('\n','',doc)
        doc = re.sub(' +',' ',doc)
        doc = doc.lower()
        return doc
    
    def lemmatize(self,token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(token, tag)
    
    def tokenize(self,document): 
        #document = unicode(document,'utf-8')
        lemmy = []
        for sent in sent_tokenize(document):
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                if token in self.stop_word:
                    continue
                lemma = self.lemmatize(token, tag)
                lemmy.append(lemma)
        return lemmy
def join_tokens(data):
    ans = ' '.join(data)
    return ans


replacer = RegexpReplacer()
remover = Remove_Noise()
AntoRep = AntonymReplacer()
train['X_input'].fillna(' ', inplace=True)
test['X_input'].fillna(' ', inplace=True)
train['comment_full'] = train['X_input'].apply(replacer.replace)
test['comment_full'] = test['X_input'].apply(replacer.replace)
train['Remove_noise'] = train['comment_full'].apply(remover.noise_rm)
test['Remove_noise'] = test['comment_full'].apply(remover.noise_rm)
train['TokenandLemma'] = train['Remove_noise'].apply(remover.tokenize)
test['TokenandLemma'] = test['Remove_noise'].apply(remover.tokenize)
train["Processed"] = train['TokenandLemma'].apply(AntoRep.replace_negations)
test["Processed"] = test['TokenandLemma'].apply(AntoRep.replace_negations)
train["Sentence"] = train["Processed"].apply(join_tokens)
test["Sentence"] = test["Processed"].apply(join_tokens)
train.to_pickle('train_processed.pkl')
test.to_pickle('test_processed.pkl')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
train = pd.read_pickle('train_processed.pkl')
test = pd.read_pickle('test_processed.pkl')
test.head()
train_text = train['Sentence']
test_text = test['Sentence']
!ls -lah
word_vec = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='word',ngram_range=(1, 2),max_features=20000)
char_vec = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='char',ngram_range=(1, 6),max_features=20000)
train_word_features = word_vec.fit_transform(train_text)
test_word_features = word_vec.transform(test_text)
train_char_features = char_vec.fit_transform(train_text)
test_char_features = char_vec.transform(test_text)
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score , precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve,precision_score,recall_score,classification_report
classifierEtree = ExtraTreesClassifier(n_estimators=200,n_jobs=-1)
classifierLR = LogisticRegression(solver='sag')
train_target = train['Y']
test_target = test['Y']
classifierEtree.fit(train_features, train_target)
classifierLR.fit(train_features, train_target)
test_predicted = classifierEtree.predict(test_features)
    
accuracy = accuracy_score(test_target,test_predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(test_target,test_predicted))
test_predicted = classifierLR.predict(test_features)
    
accuracy = accuracy_score(test_target,test_predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(test_target,test_predicted))
from sklearn.externals import joblib
filename = 'final_modelET.pkl'
joblib.dump(classifierEtree,filename)
filename = 'final_modelLR.pkl'
joblib.dump(classifierLR,filename)
filename = 'char_vectorizer.pkl'
joblib.dump(char_vec,filename)
filename = 'word_vectorizer.pkl'
joblib.dump(word_vec,filename)
classifier = joblib.load('final_modelLR.pkl')
classifier = joblib.load('final_modelET.pkl')
classifier = joblib.load('char_vectorizer.pkl')
