# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from scipy import sparse
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   

#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

#settings
color = sns.color_palette()
sns.set_style("dark")

stopword_list = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
# Any results you write to the current directory are saved as output.
# Load Dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("DIMENSION OF DATABASE : ")
print(">>> Dimension du train :", train.shape) # (159571, 8)
print(">>> Dimension du train :", test.shape) # (153164, 2)

# Unique ID ? 
g=train['id'].value_counts()
g.where(g>1).dropna()

g=test['id'].value_counts()
g.where(g>1).dropna()

# Traitement des valeurs manquantes
print("\nMISSING VALUES : ")
print(">>> Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print(">>> Check for missing values in Test dataset")
null_check=test.isnull().sum()
print(null_check)
print(">>> Filling NA with \"unknown\"")
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)

# Repartition des différentes classes : 
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for col in list_classes:
    print("\nRépartition pour la variable ", col, " : \n", collections.Counter(train[col]))
    
#marking comments without any tags as "clean"
rowsums=train.iloc[:,2:].sum(axis=1)
train['total_toxicity'] = rowsums
train['clean']=(rowsums==0)
train.head()

print('\nDistribution of Total Toxicity Labels (important for validation)')
print('On train set : ',pd.value_counts(train.total_toxicity))
# Packages require 
import emoji

"""
Description de la fonction : 
1. Identifier les emoticones dans les texts comments
2. Lister et compter les emoticones 
3. Supprimer les emoticones dans la phrase
"""

# Fonction 1 : Extraire tous les emoji de la phrase
def extract_emojis(str):
  return ' '.join(c for c in str if c in emoji.UNICODE_EMOJI)

# Definition des pattern unicode pour les emojis 
'''
TO DO:
pour bien nettoyer la liste, il va falloir faire un dico des unicodes pour les emoticons
'''
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\u2122"
                           u'\u260E'
                           "]+", flags=re.UNICODE)
# Fonction 2 : 
def pipeline_emoji(df):
    # Lister tous les emojis présents dans la phrase
    df['list_emoji'] = df["comment_text"].apply(lambda x: extract_emojis(x))
    # Compter le nombre d'emoji
    df['count_emoji'] = df["comment_text"].apply(lambda x: len(extract_emojis(x)))
    # Supprimer les emojis - on remplace
    df["comment_text"] = df["comment_text"].apply(lambda x: emoji_pattern.sub(r'', x)) 
    
"""
pipeline_emoji(train)
print(list(train.columns))

# Recuperation des obs qui ont un emoji
m = np.array(train['nb_emoji'])
idx = np.where(m == 1)
train.iloc[idx]
train.comment_text.iloc[599]
# 137 et 143 et 599
print(train.comment_text.iloc[599].encode('ascii', 'backslashreplace'))
"""
def indirect_features(df):
    
    # ========================================================================================
    # >>> Retreat IP Adress
    # ========================================================================================
    print(">>>> Retreat IP Address ----------------- ")
    ip_pattern = re.compile("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", flags=re.UNICODE)
    df['ip']=df["comment_text"].apply(lambda x: 
                                      re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}",str(x)))
    # Count of ip addresses
    df['count_ip']=df["ip"].apply(lambda x: len(x))
    # Delete IP Adress
    df["comment_text"] = df["comment_text"].apply(lambda x: ip_pattern.sub(r'', x)) 

    # ========================================================================================
    # >>> Retreat Link
    # ========================================================================================
    # Links
    df['complete_link']=df["comment_text"].apply(lambda x: 
                                                 " ".join(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                                             str(x))))

    """
    TO DO : A REVOIR >>> >
    df['domain_name']=df['complete_link'].apply(lambda x:
                                               urlparse(x))
    # ValueError: Invalid IPv6 URL
    """

    # Count of links
    df['count_links']=df["complete_link"].apply(lambda x: len(x))

    """
    TO DO : delete links
    """
    
    # ========================================================================================
    # >>> Retreat time and date
    # ========================================================================================
    
    """ 
    
    TO DO : A REVOIR CAR D'AUTRES TYPES DE REGEX SUR LES DATES 
    Exemple : L612 : 5-Mar 15 
    
    """
    
    time_pattern = re.compile("\d{1,2}:\d{1,2}", 
                              flags=re.UNICODE)
    date_pattern = re.compile(r'\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|may|june|july|august|september|october|november|december)\s\d{4}', 
                              flags=re.UNICODE)

    df['time']=df["comment_text"].apply(lambda x: re.findall("\d{1,2}:\d{1,2}",str(x)))
    df['time_flag']=df.time.apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: time_pattern.sub(r'', x)) 

    df['date']=df["comment_text"].apply(lambda x: 
                                        re.findall(r'\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}',
                                                   str(x)))
    df['date_flag'] = df.date.apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: date_pattern.sub(r'', x)) 

    # ========================================================================================
    # >>> Retreat ID User 
    # ========================================================================================
    # TO DO : Delete username >>> Not very interesting
    user_pattern = re.compile("\[\[User(.*)", flags=re.UNICODE)

    df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)",str(x)))
    df['count_usernames']=df["username"].apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: user_pattern.sub(r'', x)) 

    # ========================================================================================
    # >>> Retreat divers pattern 
    # ========================================================================================
    divers_pattern = re.compile("\(UTC\)|\(utc\)", flags=re.UNICODE)
    text = divers_pattern.sub(r'', text)

    # ========================================================================================
    # >>> Retreat emoji
    # ========================================================================================
    pipeline_emoji(df)
     
    # ========================================================================================
    # Retraitement si case vide
    # ========================================================================================
    df.comment_text = df.comment_text.replace(r'^\s*$', "NAN", regex=True)
    
    # ========================================================================================
    # >>> Retreat words
    # ========================================================================================
    # 1- '\n' can be used to count the number of sentences in each comment
    df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
    # 2- Word count in each comment:
    df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
    # 3- Unique word count
    df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
    # 4- Letter count
    df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
    # 5- upper case words count
    df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    # 6- title case words count
    df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    # 7- Number of stopwords
    df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopword_list]))
    # 8- Average length of the words
    df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment 
                                                                  if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                    axis=1)
    
    # ========================================================================================
    # >>> Symbols, punctuation and emoticon
    # ========================================================================================
    # 1- punctuation count
    df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) 
                                                                      if c in string.punctuation]))
    # 2- Exclamation marks count
    df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
    # 3- Question marks count
    df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
    # 4- Symbols counts
    df['num_symbols'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in '*&$%'))
    # 5- Smiley count
    df['num_smilies'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

    # ========================================================================================
    # >>> Derived features
    # ========================================================================================
    # 1- Word count percent in each comment:
    df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
    # 2- Punct percent in each comment:
    df['punct_percent']=df['count_punctuations']*100/df['count_word']
    
indirect_features(train)
indirect_features(test)
print("FIN >>> > ")
CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

def expand_contractions(sentence, contraction_mapping): 
     
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  
                                      flags=re.IGNORECASE|re.DOTALL) 
    def expand_match(contraction): 
        match = contraction.group(0) 
        first_char = match[0] 
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                        
        expanded_contraction = first_char+expanded_contraction[1:] 
        return expanded_contraction 
         
    expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
    return expanded_sentence 

# Remove accent and diacritics 
import unidecode
def remove_accent_before_tokens(sentences):
    res = unidecode.unidecode(sentences)
    return(res)

# Remove accent and punctuation
def remove_before_token(sentence, keep_apostrophe = False):
    sentence = sentence.strip()
    if keep_apostrophe:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    else :
        PATTERN = r'[^a-zA-Z0-9]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    return(filtered_sentence)

print("Fin")
"""

TO DO >>> Dans les extracts de features indirectes, on a supprimer des éléments dans les commentaires.

"""
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup

def preprocessing_clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    
    # >>> Remove html 
    comment=BeautifulSoup(comment).get_text()
    
    # >>> Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    
    # >>> Suppression des leak elements 
    # remove \n
    comment=re.sub("\\n","",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)    
    words=[CONTRACTION_MAP[word] if word in CONTRACTION_MAP else word for word in words]

    # Stopwords
    words = [w for w in words if not w in stopword_list]
    
    # Lemmatization
    # words= [lem.lemmatize(word, "v") for word in words]

    clean_sent=" ".join(words)

    # remove any non alphanum,digit character
    #clean_sent=re.sub("\W+"," ",clean_sent)
    #clean_sent=re.sub("  "," ",clean_sent)
    return(clean_sent)

print(">>> Before cleaning")
print(train.comment_text.iloc[23])

print("\n>>> After cleaning")
preprocessing_clean(train.comment_text.iloc[23])
"""

TO DO  :

- Faire une colonne text lemma
- Faire une colonne texte stem 

Essayer les modeles sur ces deux versions 

"""
# Application sur tout le corpus : 
# type(train.comment_text) = pandas.core.series.Series
# type(corpus) = pandas.core.series.Series

# Clean des comments sur le train
clean_corpus=train.comment_text.apply(lambda x :preprocessing_clean(x))
print("Not cleaned : ", clean_corpus[42])
print("\nCleaned : ", clean_corpus[42])

print("FIN")

"""
# Clean des comments sur le test
clean_corpus=test.comment_text.apply(lambda x :preprocessing_clean(x))
print("Not cleaned : ", clean_corpus[42])
print("Cleaned : ", clean_corpus[42])
"""
print("Not cleaned : ", clean_corpus[23])
print("\nCleaned : ", clean_corpus[23])
"""
Create final dataset 
"""

"""

# Transform series into dataframe
df_clean_corpus = clean_corpus.reset_index()

# Merge
df_final = pd.concat([df.drop("comment_text",axis = 1),
                      df_clean_corpus.drop("index",axis = 1)], 
                     axis =1 )
df_final.head()
"""

# TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
# Exemple de lancement : 
# tfidf_word = TfidfVectorizer()
# X_tfidf_word = tfidf_word.fit_transform(X[:, 1])

# TF-IDF sur les mots :
tfidf_word = TfidfVectorizer()
X_tfidf_word = tfidf_word.fit_transform(clean_corpus)

# TF-IDF sur les character n-grams :
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
X_tfidf_char = tfidf_char.fit_transform(clean_corpus)

# Concatenation des deux bases 
X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])

# Affichages des features dans les deux trucs 
features_tfidf_word = np.array(tfidf_word.get_feature_names())
print(list(features_tfidf_word))
features_tfidf_word = np.array(tfidf_char.get_feature_names())
print(list(features_tfidf_word))
clean_corpus
barack-obama-mother.jpg
159559
# Avant de lancer une régression logistique ou autre : 
X_tfidf_word = tfidf_word.transform(clean_corpus)
type(X_tfidf_word)
X_tfidf_char = tfidf_char.transform(clean_corpus)
type(X_tfidf_char)
X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])

# Après on peut : LogisticRegression().fit(X_tfidf, y[:, i])
# Pour ajouter les features indirectes dans la base TF IDF: 
features = ['count_sent',
 'count_word',
 'count_unique_word',
 'count_letters',
 'count_words_upper',
 'count_words_title',
 'count_stopwords',
 'mean_word_len',
 'total_length',
 'capitals',
 'caps_vs_length',
 'count_punctuations',
 'num_exclamation_marks',
 'num_question_marks',
 'num_symbols',
 'num_smilies',
 'word_unique_percent',
 'punct_percent']

x_feat_indirect = train[features]
from scipy.sparse import hstack
X_train_dtm = hstack([X_tfidf,np.array(x_feat_indirect)])
X_train_dtm.shape
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
lst_drop = TARGET_COLS
print(TARGET_COLS)
lst_drop.append("id")
lst_drop.append("comment_text")
lst_drop.append("total_toxicity")
lst_drop.append("clean")
print(lst_drop)
# Sur la base de train
train_x  = train.drop(lst_drop, axis = 1)
print(list(train_x.columns))
target_y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
print(list(target_y.columns))

#Sur la base de test
test_x = test.drop(["id","comment_text"],axis = 1)
print(list(test_x.columns))
test_x.fillna(0, inplace=True)
# Fitting a simple Logistic Regression indirect feature
print("Using only Indirect features")
model = LogisticRegression(C=3)
X_train, X_valid, y_train, y_valid = train_test_split(train_x, 
                                                      target_y, 
                                                      test_size=0.25,
                                                      random_state=42)
train_loss = []
valid_loss = []
importance=[]

submission_binary = pd.read_csv('../input/sample_submission.csv')

for i, j in enumerate(TARGET_COLS):
    print('Class:= '+j)
    model.fit(X_train,y_train[j])
    preds_valid = model.predict_proba(X_valid)
    preds_train = model.predict_proba(X_train)
    train_loss_class=log_loss(y_train[j],preds_train)
    valid_loss_class=log_loss(y_valid[j],preds_valid)
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    """
    importance.append(model.coef_)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
    """
    
    # Prediction in same time in order to create submission file
    test_y_prob = model.predict_proba(test_x)
    submission_binary[j] = test_y_prob
    
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))

# generate submission file
# submission_binary.to_csv('submission_binary.csv',index=False)
"""
# Create a train and test set
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# Train the model 
from nltk.tag.perceptron import PerceptronTagger
pct_tag = PerceptronTagger(load=False)
pct_tag.train(train_sents)

# Check the performance 
print ("Evaluation Own PerceptronTagger on train set ", pct_tag.evaluate(train_sents))
print ("Evaluation Own PerceptronTagger on test set ", pct_tag.evaluate(test_sents))
"""
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')



df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]


vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
data = vectorizer.fit_transform(df)
X = MaxAbsScaler().fit_transform(data)

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((test.shape[0], len(col)))



loss = []

for i, j in enumerate(col):
    print('===Fit '+j)
    model = LogisticRegression()
    model.fit(X[:nrow_train], train[j])
    preds[:,i] = model.predict_proba(X[nrow_train:])[:,1]
    
    pred_train = model.predict_proba(X[:nrow_train])[:,1]
    print('log loss:', log_loss(train[j], pred_train))
    loss.append(log_loss(train[j], pred_train))
    
print('mean column-wise log loss:', np.mean(loss))
    
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)
