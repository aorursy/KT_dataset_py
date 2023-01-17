import numpy as np

import pandas as pd



from pylab import rcParams

import matplotlib.pyplot as plt

import seaborn as sns

import re



%matplotlib inline

rcParams['figure.figsize'] = 10, 8

sns.set_style('whitegrid')



import sklearn

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score



from sklearn import metrics 

from sklearn.metrics import log_loss, classification_report, make_scorer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC



import nltk

from nltk import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer



import string



from datetime import datetime



from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore", message="lbfgs failed to converge. Increase the number of iterations.")

warnings.filterwarnings("ignore", message="The max_iter was reached which means the coef_ did not converge")
def replacePattern(ps, pattern='url', replace='', regex=True):

    

    if type(ps) != pd.core.series.Series:

        raise ValueError('"ps" parameter must be a Pandas Series object type')

    

    if pattern == 'url':

        pattern = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    

    ps.replace(to_replace = pattern, value = replace, regex = regex, inplace=True)

    return ps
def removePunctuation(post):

    punc_numbers = string.punctuation + '0123456789'

    return ''.join([l for l in post if l not in punc_numbers])
# Basic preprocessing on the text body to eliminate delimiters and URLs

def preProcess(ps):

    ps = ps.str.lower()

    ps = replacePattern(ps, 'url', 'url-web') 

    ps = replacePattern(ps, '\.\.\.\|\|\|', ' ') # Longs posts end with "...|||"

    ps = replacePattern(ps, '\|\|\|', ' ') # Posts are separated with "|||"

    return ps
def getData():

    

#   Import

    mbti_train = pd.read_csv('../input/train.csv')

    mbti_test = pd.read_csv('../input/test.csv')

    mbti = pd.concat([mbti_train, mbti_test], sort=False)

    

#   Basic preprocessing

    mbti['posts'] = preProcess(mbti['posts'])

    

#   Create target categories

    mbti['mind_i'] = mbti_train['type'].apply(lambda x: 1 if x[0] == 'I' else 0)

    mbti['energy_s'] = mbti_train['type'].apply(lambda x: 1 if x[1] == 'S' else 0)

    mbti['nature_t'] = mbti_train['type'].apply(lambda x: 1 if x[2] == 'T' else 0)

    mbti['tactics_j'] = mbti_train['type'].apply(lambda x: 1 if x[3] == 'J' else 0)



    return (mbti, mbti_train, mbti_test)
# Returns the number of occurances of a string in a body of text

def getStringCount(ps, s):

    sumCount = 0

    if type(s) == list:

        for item in s:

            sumCount = sumCount + ps.apply(lambda x: len(re.findall(item, x))).sum()

    else:

        sumCount = sumCount + ps.apply(lambda x: len(re.findall(s, x))).sum()

    return sumCount



# Returns a dataframe of word occurances, normalised according to frequency

def getCommonWordCloud(mbti, th=100, ngrams=None):

    

#   Remove punctuation and get word vector

    mbti = mbti.copy()

    posts_no_punct = mbti['posts'].apply(removePunctuation)

    

    if ngrams != None:

        mbti_vect = CountVectorizer(lowercase=True, max_features=None, min_df=th, ngram_range=ngrams)

    else:

        mbti_vect = CountVectorizer(lowercase=True, max_features=None, min_df=th)

        

    X = mbti_vect.fit_transform(posts_no_punct)



#   Summarise vector for each personality type as new dataframe

    dfX = pd.DataFrame(X.A)

    dfX.columns = mbti_vect.get_feature_names()



    df = pd.DataFrame(columns=dfX.columns)



    df = df.append(dfX[list(mbti['mind_i'] == 1)].sum(), ignore_index=True)

    df = df.append(dfX[list(mbti['mind_i'] == 0)].sum(), ignore_index=True)



    df = df.append(dfX[list(mbti['energy_s'] == 1)].sum(), ignore_index=True)

    df = df.append(dfX[list(mbti['energy_s'] == 0)].sum(), ignore_index=True)



    df = df.append(dfX[list(mbti['nature_t'] == 1)].sum(), ignore_index=True)

    df = df.append(dfX[list(mbti['nature_t'] == 0)].sum(), ignore_index=True)



    df = df.append(dfX[list(mbti['tactics_j'] == 1)].sum(), ignore_index=True)

    df = df.append(dfX[list(mbti['tactics_j'] == 0)].sum(), ignore_index=True)



    df = df.append(dfX.sum(), ignore_index=True)



    df.index = ['i', 'e', 's', 'n', 't', 'f', 'j', 'p', 'total']



#   Adjust each row according to the word count for that personality type

#   And by total word frequency

    for k in ['i', 'e', 's', 'n', 't', 'f', 'j', 'p']:

        df.loc[k] = (df.loc[k] / max(1,df.loc[k].sum())) / df.loc['total']



#   Normalise rows

    df.iloc[0:8] = df.iloc[0:8]/df.iloc[0:8].mean().mean()



#   Calculate the standard deviation of each word

    df.loc['std_dev'] = df.drop('total').std(axis=0)

    

    return df.transpose()
mbti_base, mbti_train, mbti_test = getData()

mbti = mbti_base.copy()
df = getCommonWordCloud(mbti, th=100)
df.head()
df.shape
len(df[(df['i'] == 0) | (df['e'] == 0) | (df['s'] == 0) | (df['n'] == 0) | (df['t'] == 0) | (df['f'] == 0) | (df['j'] == 0) | (df['p'] == 0)])
sns.distplot(list(df['std_dev']))
np.array(df['std_dev']).mean()
df[df['total'] > 2000].shape
sns.distplot(list(df['total']))
sns.distplot(list(df[df['total'] > 2000]['std_dev']))
df[df['total'] > 2000]['std_dev'].mean()
sns.distplot(list(df[df['total'] < 2000]['std_dev']))
df[df['total'] < 2000]['std_dev'].mean()
df = df.sort_values(by='std_dev', axis=0, ascending=False)

df[['std_dev','total']][0:10]
df = df.sort_values(by='total', axis=0, ascending=False)

df[['std_dev','total']][0:10]
mbti_emoticons = mbti_base.copy()



# Here we define what emoticons we are searching for

emoticons_list = {

    'smile': [':‑\)',':\)',':-\]',':\]',':-3',':3',':->',':>','8-\)','8\)',':-\}',':\}',':o\)',':c\)',':\^\)','=\]','=\)'],

    'laugh': [':‑d',':d','8‑d','8d','x‑d','xd','x‑d','=d','=3','b\^d'],

    'unhappy': [':‑\(', ':\(', ':‑c', ':c', ':‑< ', ':<', ':‑\[', ':\[', ':-\|\|', ':\{', ':@'],

    'surprise': [':‑o', ':o', ':-0', '8‑0'],

    'wink': [';\)'],

}



emoticons_count = {}



for k,v in emoticons_list.items():

    emoticons_count[f'emoticon{k}'] = getStringCount(mbti_emoticons['posts'], emoticons_list[k])
# Graph the frequency of emoticons

yRange = np.arange(len(emoticons_count))

plt.barh(yRange, list(emoticons_count.values()))

plt.yticks(yRange, list(emoticons_count.keys()))

plt.show()
def replaceEmoji(ps):

    ps = ps.apply(lambda x:  re.sub(r":([A-Za-z]+):", r'emoji\1', x))

    return ps



def replaceEmoticons(ps, emoticons_list):

    for k,v in emoticons_list.items():

        for item in v:

            ps = replacePattern(ps, item, f'emoticon{k}')

    return ps
mbti_emoticons = mbti_base.copy()

corpus = ' '.join(list(mbti_emoticons['posts']))

emojiSet = set(re.findall(r":([A-Za-z]+):", corpus))
len(list(emojiSet))
list(emojiSet)[0:8]
mbti_emoticons['posts'] = replaceEmoji(mbti_emoticons['posts'])

mbti_emoticons['posts'] = replaceEmoticons(mbti_emoticons['posts'], emoticons_list)

mbti_emoticons['posts'] = mbti_emoticons['posts'].apply(removePunctuation)



df = getCommonWordCloud(mbti_emoticons)
df.loc[list(emoticons_count.keys())][['total', 'std_dev']]
matchingList = []

for i in emojiSet:

    if 'emoji'+i in list(df.index):

        matchingList.append('emoji' + i)



df.loc[matchingList].sort_values(['std_dev'], ascending=False)[['total', 'std_dev']][0:8]
def mbti_stemmer(words, stemmer):

    return [stemmer.stem(word) for word in words]



def mbti_lemma(words, lemmatizer):

    return [lemmatizer.lemmatize(word) for word in words]



def replaceWords(words, vocab_dict):

    return [vocab_dict.get(word, word) for word in words]



def reduceUncommonWords(mbti, th=100, uncommon_th=1000):

    mbti_vect = CountVectorizer(lowercase=True, max_features=None, min_df=th)

    X_count = mbti_vect.fit_transform(mbti['posts'])



    X = X_count.A

    X = X.sum(axis=0)



    df_vocab = pd.DataFrame(mbti_vect.vocabulary_, index=['index']).transpose()

    df_vocab = df_vocab.sort_values(['index'], ascending=True)



    df_vocab['count'] = X

    df_vocab = df_vocab.reset_index()

    df_vocab.rename(columns={'level_0': 'word'}, inplace=True)

    df_vocab.drop(['index'], axis=1, inplace=True)



    stemmer = SnowballStemmer('english')

    lemmatizer = WordNetLemmatizer()



    # Get low count words

    s_vocab_low = df_vocab[df_vocab['count'] < uncommon_th]['word'].apply(str)

    df_vocab_low = pd.DataFrame(s_vocab_low)



    # Only keep english words

    s_isEnglish = df_vocab_low['word'].apply(lambda x: len(wordnet.synsets(x)) > 0)

    df_vocab_low = df_vocab_low[s_isEnglish]



    # Get lemmas & stems

    df_vocab_low['stems'] = mbti_stemmer(mbti_lemma(df_vocab_low['word'], lemmatizer), stemmer)



    df_vocab_conver = df_vocab_low[df_vocab_low['word'] != df_vocab_low['stems']][['word', 'stems']]

    df_vocab_conver_dict = dict(zip(df_vocab_conver['word'], df_vocab_conver['stems']))



    tokeniser = TreebankWordTokenizer()

    

    mbti['tokens'] = mbti['posts'].apply(tokeniser.tokenize)

    mbti['posts'] = mbti['tokens'].apply(lambda x: ' '.join(x))

    mbti['tokens_replaced'] = mbti['tokens'].apply(lambda x: replaceWords(x, df_vocab_conver_dict))

    mbti['post_replaced'] = mbti['tokens_replaced'].apply(lambda x: ' '.join(x))

    mbti['posts'] = mbti['post_replaced']

    

    mbti.drop(['tokens', 'tokens_replaced', 'post_replaced'], axis=1, inplace=True)

    return mbti
mbti = mbti_base.copy()

mbti['posts'] = mbti['posts'].apply(removePunctuation)

mbti_reduced = reduceUncommonWords(mbti, th=10, uncommon_th=100)



df = getCommonWordCloud(mbti_reduced)



df['std_dev'].mean()
mbti_devisive = mbti_base.copy()

mbti_devisive['posts'] = mbti_devisive['posts'].apply(removePunctuation)

df = getCommonWordCloud(mbti_devisive, th=5, ngrams=None)
df.shape
df['std_dev'].mean()
df = df[(df['i'] != 0) & (df['e'] != 0) & (df['s'] != 0) & (df['n'] != 0) & (df['t'] != 0) & (df['f'] != 0) & (df['j'] != 0) & (df['p'] != 0)]
df['std_dev'].mean()
devisive_vocab = list(df[df['std_dev'] > 0.15].index)
# Runs for a long while

mbti_combined = mbti_base.copy()

mbti_combined['posts'] = replaceEmoji(mbti_emoticons['posts'])

mbti_combined['posts'] = replaceEmoticons(mbti_combined['posts'], emoticons_list)

mbti_combined['posts'] = mbti_combined['posts'].apply(removePunctuation)

df = getCommonWordCloud(mbti_combined, th=100, ngrams=(1,2))
df = df[(df['i'] != 0) & (df['e'] != 0) & (df['s'] != 0) & (df['n'] != 0) & (df['t'] != 0) & (df['f'] != 0) & (df['j'] != 0) & (df['p'] != 0)]
df = df.sort_values(by='std_dev', axis=0, ascending=False)

df[['std_dev','total']][0:10]
df['std_dev'].mean()
df[df['std_dev'] > 0.1]['std_dev'].mean()
combined_vocab = list(df[df['std_dev'] > 0.1].index)



vect = CountVectorizer(lowercase=True, max_features=None, min_df=100, ngram_range=(1,2))

ngramX = vect.fit_transform(mbti_combined['posts'])



dfX = pd.DataFrame(ngramX.A)



dfX.columns = vect.get_feature_names()



dfX_reduced = dfX[combined_vocab]
dfX_reduced.shape
# Returns CountVectorized X_train and X_test

def getX(mbti, splitAt, min_df=100, vocab=None, ngram_range=None):

    if vocab != None:

        mbti_vect = CountVectorizer(lowercase=True, vocabulary=vocab)

    elif ngram_range != None:

        mbti_vect = CountVectorizer(lowercase=True, max_features=None, min_df=min_df, ngram_range=ngram_range)

    else:

        mbti_vect = CountVectorizer(lowercase=True, max_features=None, min_df=min_df)

        

    X = mbti_vect.fit_transform(mbti['posts'])

    X_train = X[:splitAt]

    X_test = X[splitAt:]

    return (X_train, X_test)



def logLossScorer(y_true, y_pred):

    return log_loss(y_true, y_pred)



# Returns mean cross-validated score

def getCrossValScore(X, y, scorer, custom_regressor=None):

    logreg = LogisticRegression(C=np.inf, solver='lbfgs')

    

    if custom_regressor != None:

        return cross_val_score(custom_regressor, X, y, cv=2, scoring=scorer).mean()

    return cross_val_score(logreg, X, y, cv=2, scoring=scorer).mean()



# Makes a custom logLoss scorer and return the cross-validated score

def getScore(X, targets, custom_regressor=None):

    scores = []

    scorer = make_scorer(logLossScorer)

    for target in targets:

        scores.append(getCrossValScore(X, target, scorer, custom_regressor))

    return round(np.array(scores).mean(), 12)



# Creates a regressor then fits and predicts

def predictColumn(X, y, X_test, logreg=None):

    if logreg == None:

        logreg = LogisticRegression(C=np.inf, solver='lbfgs')

    logreg.fit(X, y)

    return logreg.predict_proba(X_test)



# Return the target values as boolean arrays

def getTargets(train):

    y_ie = train['type'].apply(lambda x: 1 if x[0] == 'I' else 0)

    y_sn = train['type'].apply(lambda x: 1 if x[1] == 'S' else 0)

    y_tf = train['type'].apply(lambda x: 1 if x[2] == 'T' else 0)

    y_jp = train['type'].apply(lambda x: 1 if x[3] == 'J' else 0)

    return (y_ie, y_sn, y_tf, y_jp)



# A GridSearchCV method for finding the best paramaters for a regressor

def tune_model(X, y): 

    parameters = {'C':(0.0001, 0.001, 0.01, 1, 10, 100),

                 'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}

    clf = GridSearchCV(LogisticRegression(), parameters, cv=3, iid=True)

    clf.fit(X,y)

    return clf
# Get universal targets

y_ie, y_sn, y_tf, y_jp = getTargets(mbti_train)
mbti_basic = mbti_base.copy()

mbti_basic['posts'] = mbti_basic['posts'].apply(removePunctuation)



X, X_test = getX(mbti_basic, len(mbti_train), 100)

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
X, X_test = getX(mbti_basic, len(mbti_train), 100)

logreg = LogisticRegression(C=np.inf, solver='lbfgs', class_weight='balanced')

score = getScore(X, [y_ie, y_sn, y_tf, y_jp], logreg)

print(f'Score {score}')
X, X_test = getX(mbti_basic, len(mbti_train), 5)

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
X, X_test = getX(mbti_basic, len(mbti_train), 100, ngram_range=(1,2))

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
X, X_test = getX(mbti_basic, len(mbti_train), 5, ngram_range=(1,2))

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
mbti_punct = mbti_base.copy()

X, X_test = getX(mbti_punct, len(mbti_train), 100)

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
(X, X_test) = getX(mbti_reduced, len(mbti_train), 100)

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
(X, X_test) = getX(mbti_emoticons, len(mbti_train), 100)

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
X, X_test = getX(mbti_devisive, len(mbti_train), vocab=devisive_vocab)

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
X = dfX_reduced[:len(mbti_train)]

score = getScore(X, [y_ie, y_sn, y_tf, y_jp])

print(f'Score {score}')
X, X_test = getX(mbti_basic, len(mbti_train), 100)

y = mbti_train['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



logreg = LogisticRegression(C=np.inf, solver='lbfgs', multi_class='auto')

logreg.fit(X_train, y_train)

y_pred = logreg.predict_proba(X_test)

df_y_pred = pd.DataFrame(y_pred)



pred_labels = df_y_pred.apply(lambda x: logreg.classes_[x.idxmax()], axis=1)



pred_labels = pred_labels.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)

(pred_labels == y_test).mean()*100
# use the best scoring preprocessed text body

X_test = dfX_reduced[len(mbti_train):]



# Set our targets

y_ie = mbti_train['type'].apply(lambda x: 0 if x[0] == 'I' else 1)

y_sn = mbti_train['type'].apply(lambda x: 0 if x[1] == 'S' else 1)

y_tf = mbti_train['type'].apply(lambda x: 0 if x[2] == 'F' else 1)

y_jp = mbti_train['type'].apply(lambda x: 0 if x[3] == 'P' else 1)



# Train test split

X_train_ie, X_test_ie, y_train_ie, y_test_ie = train_test_split(X, y_ie, test_size=0.2, random_state=42)

X_train_sn, X_test_sn, y_train_sn, y_test_sn = train_test_split(X, y_sn, test_size=0.2, random_state=42)

X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X, y_tf, test_size=0.2, random_state=42)

X_train_jp, X_test_jp, y_train_jp, y_test_jp = train_test_split(X, y_jp, test_size=0.2, random_state=42)
# Create the regressors with custom parameters

logreg_ie = LogisticRegression(C=0.01, solver='saga')

logreg_sn = LogisticRegression(C=1, solver='saga')

logreg_tf = LogisticRegression(C=0.001, solver='liblinear')

logreg_jp = LogisticRegression(C=0.001, solver='lbfgs')
# Predict the respective target values

predict_ie = pd.Series(predictColumn(X_train_ie, y_train_ie, X_test_ie, logreg_ie)[:,1])

predict_sn = pd.Series(predictColumn(X_train_sn, y_train_sn, X_test_sn, logreg_sn)[:,1])

predict_tf = pd.Series(predictColumn(X_train_tf, y_train_tf, X_test_tf, logreg_tf)[:,1])

predict_jp = pd.Series(predictColumn(X_train_jp, y_train_jp, X_test_jp, logreg_jp)[:,1])
## Scoring

predict_ie_r = predict_ie.apply(lambda x: 1 if x > 0.4 else 0)

score_ie = logLossScorer(y_test_ie, predict_ie_r)



predict_sn_r = predict_sn.apply(lambda x: 1 if x > 0.6 else 0)

score_sn = logLossScorer(y_test_sn, predict_sn_r)



predict_tf_r = predict_tf.apply(lambda x: 1 if x > 0.5 else 0)

score_tf = logLossScorer(y_test_tf, predict_tf_r)



predict_jp_r = predict_jp.apply(lambda x: 1 if x > 0.45 else 0)

score_jp = logLossScorer(y_test_jp, predict_jp_r)



np.array([score_ie, score_sn, score_tf, score_jp]).mean()
X = dfX_reduced[:len(mbti_train)]

X_test = dfX_reduced[len(mbti_train):]



# IE and SN targets are flipped for the submission

y_ei = mbti_train['type'].apply(lambda x: 0 if x[0] == 'I' else 1)

y_ns = mbti_train['type'].apply(lambda x: 0 if x[1] == 'S' else 1)

y_tf = mbti_train['type'].apply(lambda x: 0 if x[2] == 'F' else 1)

y_jp = mbti_train['type'].apply(lambda x: 0 if x[3] == 'P' else 1)



predict_ie = predictColumn(X, y_ei, X_test, logreg_ie)

predict_sn = predictColumn(X, y_ns, X_test, logreg_sn)

predict_tf = predictColumn(X, y_tf, X_test, logreg_tf)

predict_jp = predictColumn(X, y_jp, X_test, logreg_jp)



df_pred = pd.DataFrame(data={'id':mbti_test['id'], 'mind': predict_ie[:,1], 'energy': predict_sn[:,1], 'nature': predict_tf[:,1], 'tactics': predict_jp[:,1]})



df_pred['mind'] = df_pred['mind'].apply(lambda x: 1 if x > 0.4 else 0)

df_pred['energy'] = df_pred['energy'].apply(lambda x: 1 if x > 0.6 else 0)

df_pred['nature'] = df_pred['nature'].apply(lambda x: 1 if x > 0.5 else 0)

df_pred['tactics'] = df_pred['tactics'].apply(lambda x: 1 if x > 0.55 else 0)



df_pred.to_csv('./submission.csv', index=False)