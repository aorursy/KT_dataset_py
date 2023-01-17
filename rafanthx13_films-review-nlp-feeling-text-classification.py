import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
import re

from sklearn.metrics import classification_report

from bs4 import BeautifulSoup             


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.4f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
seed = 42
np.random.seed(seed)
# DataSet of PopCorn
df_train_labeled = pd.read_csv(
    '/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', sep='\t', usecols=['sentiment','review'])

# dataset de treino do imdb_master
df_train_add = pd.read_csv(
    '/kaggle/input/imdb-review-dataset/imdb_master.csv', encoding="latin-1", usecols=['review', 'label'])
# Prepare imdb to join with labeldTrain: remove unsup label elements and convert pos/neg to 1/0
df_train_add = df_train_add[df_train_add['label'] != 'unsup']
df_train_add['label'] = df_train_add['label'].map({'pos': 1, 'neg': 0})
df_train_add = df_train_add.rename({'label': 'sentiment'}, axis=1)

# join datasets
df_train = pd.concat([df_train_labeled, df_train_add]).reset_index(drop=True)
n_pos, n_neg = df_train.sentiment.value_counts()[0], df_train.sentiment.value_counts()[0]
print("Balanced DataSet:\n\t=> {} rows are negative (0) and {} rows are positive (1)".format(n_pos, n_neg))
print('Final Train DataSet Shape\n\t=> {} rows and {} columns'.format(df_train.shape[0], df_train.shape[1]))
df_train.head(3)
df_submission = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip', sep='\t')
print('Test Data To Submit: {} rows and {} columns'.format(df_submission.shape[0], df_submission.shape[1]))
df_submission.head(2)
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
   "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",
   "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
   "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
   "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
   "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
   "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
   "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is",
   "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
   "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",
   "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
   "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
   "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
   "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have",
   "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
   "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will",
   "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
   "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
   "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
   "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",
   "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
   "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
   "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
   "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
   "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",
   "you've": "you have" 
}


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
    'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
    'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many',
    'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
    'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
    'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota',
    'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp',
    'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization',
    'pokémon': 'pokemon'
}

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

def time_spent(time0):
    t = time.time() - time0
    t_int = int(t) // 60
    t_min = t % 60
    if(t_int != 0):
        return '{}min {:.3f}s'.format(t_int, t_min)
    else:
        return '{:.3f}s'.format(t_min)
from sklearn.metrics import confusion_matrix, classification_report

this_labels = ['Negative','Positive']

def class_report(y_real, y_my_preds, name="", labels=this_labels):
    if(name != ''):
        print(name,"\n")
    print(confusion_matrix(y_real, y_my_preds), '\n')
    print(classification_report(y_real, y_my_preds, target_names=labels))
def plot_nn_loss_acc(history):
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(14,5))

    # summarize history for accuracy
    axis1.plot(history.history['accuracy'], label='Train', linewidth=3)
    axis1.plot(history.history['val_accuracy'], label='Validation', linewidth=3)
    axis1.set_title('Model accuracy', fontsize=16)
    axis1.set_ylabel('accuracy')
    axis1.set_xlabel('epoch')
    axis1.legend(loc='upper left')

    # summarize history for loss
    axis2.plot(history.history['loss'], label='Train', linewidth=3)
    axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)
    axis2.set_title('Model loss', fontsize=16)
    axis2.set_ylabel('loss')
    axis2.set_xlabel('epoch')
    axis2.legend(loc='upper right')
    plt.show()

from nltk.stem import WordNetLemmatizer

t0 = time.time()

corpus = []

set_stop_words = set(stopwords.words('english'))
lematizator = WordNetLemmatizer()

for i in range(df_train.shape[0]):
    soup = BeautifulSoup(df_train.iloc[i]['review'], "html.parser")
    review = soup.get_text()
    review = re.sub('\[[^]]*\]', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set_stop_words]
    review = [lematizator.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)
    
print(time_spent(t0)) # 1min and 30s
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert to tdidf
tfidf_vec = TfidfVectorizer(ngram_range=(1, 3))
tfidf_vector = tfidf_vec.fit_transform(corpus)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    tfidf_vector, df_train['sentiment'], test_size=0.20, random_state=42)
from sklearn.svm import LinearSVC

t0 = time.time()
linear_svc = LinearSVC(C=0.5, random_state=42)
linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_test)

class_report(y_test, y_pred, 'Test Strategie 3: TD-IDF (1,3) LinearSVC')

print(time_spent(t0))
corpus_submission = []

for j in range(df_submission.shape[0]):
    soup = BeautifulSoup(df_submission.iloc[j]['review'], "html.parser")
    review = soup.get_text()
    review = re.sub('\[[^]]*\]', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set_stop_words]
    review = [lematizator.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus_submission.append(review)
    
tfidf_vec_sub = tfidf_vec.transform(corpus_submission)

predict = linear_svc.predict(tfidf_vec_sub)

df_submission['sentiment'] = predict
df_submission = df_submission[['id','sentiment']]
df_submission.to_csv("submission_linear_svc.csv",index=False) 
print("Kaggle Score: 0.98108")