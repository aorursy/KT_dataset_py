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
!pip install bert-for-tf2 

!pip install sentencepiece
!pip install swifter
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
import numpy as np
#Training data
df_train = pd.read_csv('../input/topic-modeling-for-research-articles-20/Train.csv')
print('Training data shape: ', df_train.shape)

df_test = pd.read_csv("../input/topic-modeling-for-research-articles-20/Test.csv")
print('Testing data shape:',df_test.shape)
#Training data
df_train = pd.read_csv('../input/df-train/df_train.csv')
print('Training data shape: ', df_train.shape)

df_test = pd.read_csv("../input/df-test/df_test.csv")
print('Testing data shape:',df_test.shape)

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import swifter

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

# set of stopwords
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #number_free = ''.join([i for i in stop_free if not i.isdigit()])
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def count_digits(line):
    digits_cnt = 0
    for i in [x.lower().strip() for x in line]:
        if(i.isdigit()):
            digits_cnt+=1
    return digits_cnt

#def count_punct(line):
    
def count_punct(line):
    punct_cnt = 0
    for i in [x.lower().strip() for x in line]:
        if(i in exclude):
            punct_cnt+=1
    return punct_cnt
    
    
df_train['ABSTRACT_clean'] = df_train['ABSTRACT'].swifter.set_npartitions(64).apply(lambda x: clean(x))

df_train['totalwords_text'] = df_train['ABSTRACT_clean'].str.count(' ') + 1

df_train['digit_cnts'] = df_train['ABSTRACT'].swifter.set_npartitions(32).apply(lambda x: count_digits(x))

df_train['punct_cnts'] = df_train['ABSTRACT'].swifter.set_npartitions(32).apply(lambda x: count_punct(x))

df_test['ABSTRACT_clean'] = df_test['ABSTRACT'].swifter.set_npartitions(64).apply(lambda x: clean(x))

df_test['totalwords_text'] = df_test['ABSTRACT_clean'].str.count(' ') + 1

df_test['digit_cnts'] = df_test['ABSTRACT'].swifter.set_npartitions(32).apply(lambda x: count_digits(x))

df_test['punct_cnts'] = df_test['ABSTRACT'].swifter.set_npartitions(32).apply(lambda x: count_punct(x))

df_train.head()
ID_COL = 'id'

TARGET_COLS = ['Analysis of PDEs', 'Applications',
               'Artificial Intelligence', 'Astrophysics of Galaxies',
               'Computation and Language', 'Computer Vision and Pattern Recognition',
               'Cosmology and Nongalactic Astrophysics',
               'Data Structures and Algorithms', 'Differential Geometry',
               'Earth and Planetary Astrophysics', 'Fluid Dynamics',
               'Information Theory', 'Instrumentation and Methods for Astrophysics',
               'Machine Learning', 'Materials Science', 'Methodology', 'Number Theory',
               'Optimization and Control', 'Representation Theory', 'Robotics',
               'Social and Information Networks', 'Statistics Theory',
               'Strongly Correlated Electrons', 'Superconductivity',
               'Systems and Control']

TOPIC_COLS = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
df_train['total_presence'] = df_train['Computer Science'] + df_train['Mathematics'] + df_train['Physics'] + df_train['Statistics']
df_test['total_presence'] = df_test['Computer Science'] + df_test['Mathematics'] + df_test['Physics'] + df_test['Statistics']


physics_abs = df_train['ABSTRACT_clean'][df_train['Physics'] == 1]

math_abs =  df_train['ABSTRACT_clean'][df_train['Mathematics'] == 1]

computer_abs = df_train['ABSTRACT_clean'][df_train['Computer Science'] == 1]

stats_abs = df_train['ABSTRACT_clean'][df_train['Statistics'] == 1]




import collections
import re
import sys
import time

def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    return re.findall(r'\w+', string.lower())


def count_ngrams(lines, min_length=2, max_length=4):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams1 = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams1[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams1


def print_most_frequent(ngrams1, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams1):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams1[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')



ngrams_physics = count_ngrams(physics_abs)
ngrams_comp = count_ngrams(computer_abs)
ngrams_math = count_ngrams(math_abs)
ngrams_stats = count_ngrams(stats_abs)


print_most_frequent(ngrams_physics)

physics_bigrams = []
physics_trigrams = []
physics_fourgrams = []

comp_bigrams = []
comp_trigrams = []
comp_fourgrams = []

stats_bigrams = []
stats_trigrams = []
stats_fourgrams = []

maths_bigrams = []
maths_trigrams = []
maths_fourgrams = []


numbers = [2,3,4]
for num in numbers:
    for key in ngrams_physics[num].most_common(200):
        splitted = str(key).split("'") 
        if(num == 2):
            new_text = str(splitted[1]+' '+splitted[3])
            physics_bigrams.append(new_text)
        elif(num == 3):
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5])
            physics_trigrams.append(new_text)
        else:
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5])
            physics_fourgrams.append(new_text)
    for key in ngrams_comp[num].most_common(200):
        splitted = str(key).split("'") 
        if(num == 2):
            new_text = str(splitted[1]+' '+splitted[3])
            comp_bigrams.append(new_text)
        elif(num == 3):
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5])
            comp_trigrams.append(new_text)
        else:
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5]+ ' ' + splitted[7])
            comp_fourgrams.append(new_text)
    for key in ngrams_stats[num].most_common(200):
        splitted = str(key).split("'") 
        if(num == 2):
            new_text = str(splitted[1]+' '+splitted[3])
            stats_bigrams.append(new_text)
        elif(num == 3):
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5])
            stats_trigrams.append(new_text)
        else:
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5]+ ' ' + splitted[7])
            stats_fourgrams.append(new_text)
    for key in ngrams_math[num].most_common(200):
        splitted = str(key).split("'") 
        if(num == 2):
            new_text = str(splitted[1]+' '+splitted[3])
            maths_bigrams.append(new_text)
        elif(num == 3):
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5])
            maths_trigrams.append(new_text)
        else:
            new_text = str(splitted[1]+' '+splitted[3] + ' ' + splitted[5]+ ' ' + splitted[7])
            maths_fourgrams.append(new_text) 
  


maths_bigrams
def count_ngrams(line):
    bio_bigramscnt = 0 
    math_bigramscnt = 0
    stats_bigramscnt = 0
    physics_bigramscnt = 0
    comp_bigramscnt = 0
    for i in [x.lower().strip() for x in maths_bigrams]:
        if(i in line.lower()):
            math_bigramscnt+=1
    for i in [x.lower().strip()for x in stats_bigrams]:
        if(i in line.lower()):
            stats_bigramscnt+=1
    for i in [x.lower().strip() for x in physics_bigrams]:
        if(i in line.lower()):
            physics_bigramscnt+=1   
    for i in [x.lower().strip() for x in comp_bigrams]:
        if(i in line.lower()):
            comp_bigramscnt+=1         

    math_trigramscnt = 0
    stats_trigramscnt = 0
    physics_trigramscnt = 0
    comp_trigramscnt = 0


    for i in [x.lower().strip() for x in maths_trigrams]:
        if(i in line.lower()):
            math_trigramscnt+=1
    for i in [x.lower().strip()for x in stats_trigrams]:
        if(i in line.lower()):
            stats_trigramscnt+=1
    for i in [x.lower().strip() for x in physics_trigrams]:
        if(i in line.lower()):
            physics_trigramscnt+=1   
    for i in [x.lower().strip() for x in comp_trigrams]:
        if(i in line.lower()):
            comp_trigramscnt+=1    


    math_fourgramscnt = 0
    stats_fourgramscnt = 0
    physics_fourgramscnt = 0
    comp_fourgramscnt = 0


    for i in [x.lower().strip() for x in maths_fourgrams]:
        if(i in line.lower()):
            math_fourgramscnt+=1
    for i in [x.lower().strip()for x in stats_fourgrams]:
        if(i in line.lower()):
            stats_fourgramscnt+=1
    for i in [x.lower().strip() for x in physics_fourgrams]:
        if(i in line.lower()):
            physics_fourgramscnt+=1   
    for i in [x.lower().strip() for x in comp_fourgrams]:
        if(i in line.lower()):
            comp_fourgramscnt+=1    

    return [math_bigramscnt, stats_bigramscnt, physics_bigramscnt, comp_bigramscnt,
             math_trigramscnt, stats_trigramscnt, physics_trigramscnt, comp_trigramscnt,
             math_fourgramscnt, stats_fourgramscnt, physics_fourgramscnt, comp_fourgramscnt]
count_ngrams(df_train['ABSTRACT_clean'].loc[0][4])

df_train['math_bigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[0])
df_train['stats_bigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[1])
df_train['physics_bigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[2])
df_train['computer_bigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[3])


df_train['math_trigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[4])
df_train['stats_trigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[5])
df_train['physics_trigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[6])
df_train['computer_trigrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[7])

df_train['math_fourgrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[8])
df_train['stats_fourgrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[9])
df_train['physics_fourgrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[10])
df_train['computer_fourgrams'] = df_train['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[11])
    


#do this for test





df_test['math_bigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(64).apply(lambda x: count_ngrams(x)[0])
df_test['stats_bigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(64).apply(lambda x: count_ngrams(x)[1])
df_test['physics_bigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(64).apply(lambda x: count_ngrams(x)[2])
df_test['computer_bigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(64).apply(lambda x: count_ngrams(x)[3])

df_test['math_trigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[4])
df_test['stats_trigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[5])
df_test['physics_trigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[6])
df_test['computer_trigrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[7])

df_test['math_fourgrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[8])
df_test['stats_fourgrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[9])
df_test['physics_fourgrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[10])
df_test['computer_fourgrams'] = df_test['ABSTRACT_clean'].swifter.set_npartitions(32).apply(lambda x: count_ngrams(x)[11])

#do this for test
df_train.head()
df_train.to_csv("df_train.csv",index=False)
df_test.to_csv("df_test.csv",index=False)
df_train.columns
df_train['train_flag'] = 1
df_test['train_flag'] = 0
df_train_a = df_train[['id','ABSTRACT_clean','train_flag']]

df_test_a = df_test[['id','ABSTRACT_clean','train_flag']]


#df_data = pd.concat((df_train, df_test))


combined_df=df_train_a.append(df_test_a)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier

from sklearn import preprocessing
from scipy import sparse
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc


#Hashing vectorizer
## Word
hash_word = HashingVectorizer(analyzer='word', stop_words= 'english' , ngram_range= (1,3), 
                              token_pattern= r'w{1,}', strip_accents= 'unicode',
                             dtype= np.float32, tokenizer= nltk.tokenize.word_tokenize ,n_features=100000)
#Char
hash_char = HashingVectorizer(analyzer='char', stop_words= 'english' , ngram_range= (3,6),
                              strip_accents= 'unicode',dtype= np.float32 ,n_features=1000000)

hash_word.fit(combined_df['ABSTRACT_clean'])

hash_char.fit(combined_df['ABSTRACT_clean'])
FE = ['totalwords_text',
       'digit_cnts', 'punct_cnts', 'total_presence', 'math_bigrams',
       'stats_bigrams', 'physics_bigrams', 'computer_bigrams', 'math_trigrams',
       'stats_trigrams', 'physics_trigrams', 'computer_trigrams',
       'math_fourgrams', 'stats_fourgrams', 'physics_fourgrams',
       'computer_fourgrams']
from sklearn.model_selection import train_test_split 
trn, val = train_test_split(df_train, test_size=0.001, random_state=2)

# Word
tr_hash = hash_word.transform(df_train['ABSTRACT_clean'])
#val_hash = hash_word.transform(val['ABSTRACT_clean'])
ts_hash = hash_word.transform(df_test['ABSTRACT_clean'])

# char

tr_hash_char = hash_char.transform(df_train['ABSTRACT_clean'])
#val_hash_char = hash_char.transform(val['ABSTRACT_clean'])
ts_hash_char = hash_char.transform(df_test['ABSTRACT_clean'])

df_train_FE = preprocessing.scale(df_train[FE])
#df_val_FE = preprocessing.scale(val[FE])
df_test_FE = preprocessing.scale(df_test[FE])
# Sparse
#print(trn.shape, val.shape, df_test.shape)

trn2 = sparse.hstack([tr_hash, tr_hash_char,df_train_FE,df_train[TOPIC_COLS]]).tocsr() 
#val2 = sparse.hstack([val_hash, val_hash_char,df_val_FE,val[TOPIC_COLS]])
tst2 = sparse.hstack([ts_hash, ts_hash_char,df_test_FE, df_test[TOPIC_COLS]]).tocsr() 

print(trn2.shape, tst2.shape)


%%time

from sklearn.multiclass import OneVsRestClassifier
#clf = OneVsRestClassifier(LogisticRegression(C = 2.0, n_jobs=-1,class_weight = 'balanced'))
lsvc = LinearSVC(class_weight = 'balanced',random_state = 42,max_iter=300, C=2.0)
clf =  OneVsRestClassifier(lsvc)
clf.fit(trn2, trn[TARGET_COLS])


val_preds = clf.predict(val2)
f1_score(val[TARGET_COLS], val_preds, average='micro')
def get_best_thresholds(true, preds):
    thresholds = [i/100 for i in range(100)]
    best_thresholds = []
    for idx in range(25):
        f1_scores = [f1_score(true[:, idx], (preds[:, idx] >= thresh) * 1) for thresh in thresholds]
        best_thresh = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_thresh)
    return best_thresholds
#val_preds = clf.predict_proba(val2)
val_preds = clf.predict(val2)
best_thresholds = get_best_thresholds(val[TARGET_COLS].values, val_preds)

for i, thresh in enumerate(best_thresholds):
    val_preds[:, i] = (val_preds[:, i] >= thresh) * 1

f1_score(val[TARGET_COLS], val_preds, average='micro')
#preds_test = clf.predict_proba(tst2)
preds_test = clf.predict(tst2)
for i, thresh in enumerate(best_thresholds):
    preds_test[:, i] = (preds_test[:, i] >= thresh) * 1
ids = df_test[['id']]
#submission = pd.read_csv('../input/topic-modeling-for-research-articles-20/SampleSubmission.csv', index_col='id')
predictions = pd.DataFrame(preds_test,columns = TARGET_COLS)
predictions.head()
submission = ids.join(predictions)
submission.to_csv("seven submission added bigrams trigramsmore features.csv",index=False)