# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk # natural language processing
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from gensim.models import word2vec

lemmatizer = nltk.stem.WordNetLemmatizer()
stop = set(stopwords.words('english'))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def strip_punctuation(s):
    # input str, output str, strip out punctuations
    return ''.join(c for c in s if c not in punctuation)

def data_transformation(s):
    return strip_punctuation(str.lower(s))
def to_nltk_text(text):
    #input dataframe, output nltk text object. 
    token = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in token if t not in stop]
    return lemmas
def shared_differed(s1,s2):
    s1_text = to_nltk_text(s1)
    s2_text = to_nltk_text(s2)
    return list(set(s1_text).difference(s2_text))

def check_synonym(word, word2):
    """checks to see if word and word2 are synonyms"""
    l_syns = list()
    synsets = wn.synsets(word)
    for synset in synsets:
        if word2 in synset.lemma_names():
            return True
            break
    return False
def syn_dist(word_list1,word_list2):
    try:
        share = 1/float(len(word_list1)+len(word_list2))
    except :
        return 0         
    result = 1.0
    for word1 in word_list2:
        for word2 in word_list2:
            if check_synonym(word1,word2):
                result = result - share
    return result
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['source'] = 1.0
test['source'] = 0.0
alls = pd.concat([train,test],ignore_index=True)
alls['question1'] = alls['question1'].fillna("").apply(lambda x: data_transformation(x))
alls['question2'] = alls['question2'].fillna("").apply(lambda x: data_transformation(x))
mydoclist = alls['question1'].tolist() + alls['question2'].tolist()
mydoclist_lemma = [sent.split(" ") for sent in mydoclist]
mydoclist_lemma[0:5]
n = len(alls['question1'])
num_features = 50    # Word vector dimensionality                      
min_word_count = 5  # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 4          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
model = word2vec.Word2Vec(mydoclist_lemma, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)


model.init_sims(replace=True)

model_name = "quoraPair"
model.save(model_name)
mydoclist_lemma[0:100]
model = word2vec.Word2Vec.load("quoraPair") 

model.most_similar("geologist")
mydoclist[0:5]
model.wv.most_similar("invest")
#Compute Tf-Idf
count_vectorizer = CountVectorizer(min_df=1)
term_freq_matrix = count_vectorizer.fit_transform(mydoclist)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix)
tf_idf_matrix = tfidf.transform(term_freq_matrix)
tf_idf_matrix_q1  = tf_idf_matrix[0:n,]
tf_idf_matrix_q2  = tf_idf_matrix[n:2*n,]
dot_dist = tf_idf_matrix_q1.multiply(tf_idf_matrix_q2).sum(axis=1).tolist()
dot_dist = [dot for li in dot_dist for dot in li]
alls['dot_dist'] = dot_dist
alls['word_count_q1'] = alls['question1'].apply(len)
alls['word_count_q2'] = alls['question2'].apply(len)
#alls['differed_word_q1'] = alls.apply(lambda x: shared_differed(x['question1'],x['question2']),axis=1)
##alls['differed_word_q2'] = alls.apply(lambda x: shared_differed(x['question2'],x['question1']),axis=1)
#alls['syn_dist'] = alls.apply(lambda x: syn_dist(x['differed_word_q1'],x['differed_word_q2']),axis=1)
#alls['diff_ct'] = alls.apply(lambda x: len(x['differed_word_q1'])+len(x['differed_word_q2']),axis=1)
train = alls[alls['source'] == 1.0]
test = alls[alls['source'] == 0.0]
train.drop('source',axis=1,inplace=True)
test.drop('source',axis=1,inplace=True)
test.drop('is_duplicate',axis=1,inplace=True)
x = train[['dot_dist','word_count_q1','word_count_q2']]
x_test = test[['dot_dist','word_count_q1','word_count_q1']]
Y = train['is_duplicate']
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(x,Y)
#Training Score
Y_pred = clf.predict_proba(x)
from sklearn.metrics import log_loss
print(log_loss(Y,Y_pred))
#Prediction
Y_Test_Pred = clf.predict_proba(x_test)
test['is_duplicate'] =Y_Test_Pred[:,1]
submission = test[['test_id','is_duplicate']]
submission['test_id'] = submission['test_id'].apply(int)
submission.to_csv('submission.csv',index=False)