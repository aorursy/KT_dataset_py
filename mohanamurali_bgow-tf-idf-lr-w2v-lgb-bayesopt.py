import re

import nltk

import numpy as np

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import pandas as pd

stopWords = set(stopwords.words('english'))

import os

print(os.listdir("../input"))

from nltk.stem import SnowballStemmer

stem = SnowballStemmer('english')



from imblearn.metrics import sensitivity_score
# credit to https://www.kaggle.com/taindow/simple-cudnngru-python-keras 

specials = ["’", "‘", "´", "`"]

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

    '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

    '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

    '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

swear_words_re = ' 4r5e | 5h1t | 5hit | ass-fucker | assfucker | assfukka | asswhole | a_s_s | b!tch | b17ch | blow job | boiolas | bollok | boooobs | booooobs | booooooobs | bunny fucker | buttmuch | c0cksucker | carpet muncher | cl1t | cockface | cockmunch | cockmuncher | cocksuka | cocksukka | cokmuncher | coksucka | cunillingus | cuntlick | cuntlicker | cuntlicking | cyalis | cyberfuc | cyberfuck | cyberfucked | cyberfucker | cyberfuckers | cyberfucking | dirsa | dlck | dog-fucker | donkeyribber | ejaculatings | ejakulate | f u c k | f u c k e r | f4nny | faggitt | faggs | fannyflaps | fannyfucker | fanyy | fingerfucker | fingerfuckers | fingerfucks | fistfuck | fistfucked | fistfucker | fistfuckers | fistfucking | fistfuckings | fistfucks | fuckingshitmotherfucker | fuckwhit | fudge packer | fudgepacker | fukwhit | fukwit | fux0r | f_u_c_k | god-dam | kawk | knobead | knobed | knobend | knobjocky | knobjokey | kondum | kondums | kummer | kumming | kums | kunilingus | l3itch | m0f0 | m0fo | m45terbate | ma5terb8 | ma5terbate | master-bate | masterb8 | masterbat3 | masterbations | mof0 | mothafuck | mothafuckaz | mothafucked | mothafucking | mothafuckings | mothafucks | mother fucker | motherfucked | motherfuckings | motherfuckka | motherfucks | muthafecker | muthafuckker | n1gga | n1gger | nigg3r | nigg4h | nob jokey | nobjocky | nobjokey | penisfucker | phuked | phuking | phukked | phukking | phuks | phuq | pigfucker | pimpis | pissflaps | rimjaw | s hit | scroat | sh!t | shitdick | shitfull | shitings | shittings | s_h_i_t | t1tt1e5 | t1tties | teez | tittie5 | tittiefucker | tittywank | tw4t | twathead | twunter | v14gra | v1gra | w00se | whoar'
def clean_contractions(text):

    for s in specials:

        if s in text:

            text = text.replace(s, "'")

    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])

    return text



# tips to accelerate string processing

def clean_text_slow(x, maxlen=None):

    x = x.lower()

    for punct in puncts[:maxlen]:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_text_fast(x, maxlen=None):

    x = x.lower()

    for punct in puncts[:maxlen]:

        if punct in x:  # add this line

            x = x.replace(punct, f' {punct} ')

    return x

#######

def word_extraction(sentence):

    #sentence=clean_contractions(sentence)

    for s in specials:

        if s in sentence: # this line to first speed up

            sentence=sentence.replace(s,"''")

    for punct in puncts:

        if punct in sentence:

            sentence=sentence.replace(punct, f' {punct} ')

            

    sentence=' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in sentence.split(" ")])

    #sentence=re.sub(swear_words_re,' fuck',sentence) # comment because swear type may be linked to context

    words = re.sub("[^\w]"," ", sentence).split()

    cleaned_text=[stem.stem(w.lower()) for w in words]

    return str(cleaned_text)
dataset = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
toy=False

if toy==True:

    dataset=dataset.sample(n=10000)

    test=test.sample(n=10000)
dataset[dataset["target"]>0.5][['target','comment_text']].sample(n=10)
test.sample(n=5) 
dataset.target=dataset.target.apply(lambda x: 1 if x>0.5 else 0)
dataset.target.value_counts()  # classes are unbalanced

# 
%%time

dataset['cleaned_comment']=dataset.comment_text.apply(lambda x: word_extraction(x))
# try to parallelize preprocessing #no gain



import tqdm

from multiprocessing import Pool



def parallelize_apply(df,func,colname,num_process,newcolnames):

    # takes as input a df and a function for one of the columns in df

    pool =Pool(processes=num_process)

    arraydata = pool.map(func,df[colname].values)

    pool.close()

    newdf = pd.DataFrame(arraydata,columns = newcolnames)

    df = pd.concat([df,newdf],axis=1)

    return df



%%time

#parallelized_dataset = parallelize_apply(dataset,word_extraction,'comment_text',4,['cleaned_parallelized'])  # no gain 22 min
%%time

test['cleaned_comment']=test.comment_text.apply(lambda x: word_extraction(x))
# CountVectorizer can bien replaced by TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_features=4000)
# Corpus definition

corpus = pd.concat([dataset['cleaned_comment'], test['cleaned_comment']])

corpus = corpus.drop_duplicates()
%%time

#tfidf.fit(corpus)
%%time

#X=tfidf.transform(dataset['cleaned_comment'])
#y=dataset.target
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33

#                                    ,random_state=1)
#X_train.shape
#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
#clf.score(X_train,y_train)
#clf.score(X_test,y_test)
#predictions=clf.predict(X_test)

#predictions.shape
#X_test=tfidf.transform(test['cleaned_comment'])
#y_test= clf.predict(X_test)
#submission_df = pd.read_csv("../input/sample_submission.csv")

#submission_df['prediction'] = y_test
#submission_df.to_csv("submission.csv", index=False)
%%time

import gensim

from gensim.models.word2vec import Word2Vec

modelW2V = Word2Vec(sentences=corpus, size=100, window=5, min_count=5, workers=2,sg=0)

w2v = {w: vec for w, vec in zip(modelW2V.wv.index2word, modelW2V.wv.syn0)}
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict

class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec):

        self.word2vec = word2vec

        if len(word2vec)>0:

            self.dim=len(word2vec[next(iter(w2v))])

        else:

            self.dim=0

            

    def fit(self, X, y):

        return self 



    def transform(self, X):

        return np.array([

            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 

                    or [np.zeros(self.dim)], axis=0)

            for words in X

        ])



class TfidfEmbeddingVectorizer(object):

    def __init__(self, word2vec):

        self.word2vec = word2vec

        self.word2weight = None

        self.dim = len(next(iter(w2v)))



    def fit(self, X, y):

        tfidf = TfidfVectorizer(analyzer=lambda x: x)

        tfidf.fit(X)

        # if a word was never seen - it must be at least as infrequent

        # as any of the known words - so the default idf is the max of 

        # known idf's

        max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(

            lambda: max_idf,

            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])



        return self



    def transform(self, X):

        return np.array([

                np.mean([self.word2vec[w] * self.word2weight[w]

                         for w in words if w in self.word2vec] or

                        [np.zeros(self.dim)], axis=0)

                for words in X

            ])
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC

vectorizer=TfidfEmbeddingVectorizer(w2v)

vectorizer.fit(corpus,dataset.target)

X_train=vectorizer.transform(dataset.cleaned_comment)
X_test=vectorizer.transform(test.cleaned_comment)
# for making train - valid sets

from sklearn.model_selection import train_test_split

import lightgbm as lgb
def f(x):

    value=0

    if x>0.5:

        value=1

    return value

vf = np.vectorize(f)
def LGB_bayesian(

    bagging_freq,  # int

    bagging_fraction,

    feature_fraction,     

    learning_rate,

    min_data_in_leaf, #int

    min_sum_hessian_in_leaf,

    num_leaves): # int

    

    # LGB expects next three parameters need to be integer. So we make them integer

    bagging_freq = int(bagging_freq)

    min_data_in_leaf=int(min_data_in_leaf)

    num_leaves=int(num_leaves)

    assert type(bagging_freq) == int

    assert type(min_data_in_leaf) == int

    assert type(num_leaves)==int

    param = {

        'bagging_freq': bagging_freq,

        'bagging_fraction':bagging_fraction,

        'boost_from_average' :'false',

        'boost':'gbdt',

        'feature_fraction':feature_fraction,

        'learning_rate':learning_rate,

        'max_depth':-1,

        'min_data_in_leaf':min_data_in_leaf,

        'min_sum_hessian_in_leaf':min_sum_hessian_in_leaf,

        'num_leaves':num_leaves,

        'tree_learner':'serial',

        'objective': 'binary',

        'num_threads': 8, 

        "device" : "cpu"



    }    



    lgb_train = lgb.Dataset(X_train[bayesian_tr_index],

                           label=dataset.target.iloc[bayesian_tr_index].values

                           )

    lgb_valid = lgb.Dataset(X_train[bayesian_val_index], label=dataset.target.iloc[bayesian_val_index].values

                           )   



    num_round = 5000

    clf = lgb.train(param, lgb_train, 1000000, valid_sets = [lgb_train, lgb_valid], verbose_eval=100, early_stopping_rounds = 40)

    

    predictions = clf.predict(X_train[bayesian_val_index])# , num_iteration=clf.best_iteration)   

    

    predictions=  np.array([f(xi) for xi in predictions])

    #score = metrics.roc_auc_score(dataset.target.iloc[bayesian_val_index].values, predictions)

    score=sensitivity_score(dataset.target.iloc[bayesian_val_index].values, predictions, average='binary')

    return score
from sklearn import metrics

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits= 5, shuffle=True, random_state=13).split(X_train, dataset.target))[0]
Bounds_LGB = {

    'bagging_freq': (2,10),  'bagging_fraction': (0.1,0.9),

    'feature_fraction': (0.05,0.5), 'learning_rate':(0.05,0.1),

    'min_data_in_leaf': (2,100),     

    'min_sum_hessian_in_leaf': (2,30),'num_leaves': (2,100)

}
#from bayes_opt import BayesianOptimization

#LGB_BO = BayesianOptimization(LGB_bayesian, Bounds_LGB, random_state=13)



#init_points = 3  

#n_iter = 10
import warnings

#with warnings.catch_warnings():

#    warnings.filterwarnings('ignore')

#    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

#LGB_BO.max['params']
opt_param={'bagging_fraction': 0.72,

 'bagging_freq': 7,

        'boost_from_average' :'false',

        'boost':'gbdt',

                   'max_depth':-1,

 'feature_fraction': 0.375,

 'learning_rate': 0.05,

 'min_data_in_leaf': 31,

 'min_sum_hessian_in_leaf': 3.6,

 'num_leaves': 86,

'tree_learner':'serial',

        'objective': 'binary',

        'num_threads': 8, 

        "device" : "cpu"}
n_fold=5

kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=13)



cvscores = []

predictions=0

for trn_idx, val_idx in StratifiedKFold(n_splits= 5, shuffle=True, random_state=13).split(X_train, dataset.target):

    trn_data=lgb.Dataset(X_train[trn_idx], label=dataset.target.iloc[trn_idx].values)

    val_data=lgb.Dataset(X_train[val_idx], label=dataset.target.iloc[val_idx].values)



    clf = lgb.train(opt_param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=50, early_stopping_rounds = 30)

    

    y_test= clf.predict(X_train[val_idx])

    y_test=  np.array([f(xi) for xi in y_test])



    score=sensitivity_score(dataset.target.iloc[val_idx].values, y_test, average='binary')

    print('final score '+str(score))

    predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / n_fold

    

    

submission_df = pd.read_csv("../input/sample_submission.csv")

submission_df.columns

submission_df['predictions'] = predictions

submission_df.to_csv("submission.csv", index=False)