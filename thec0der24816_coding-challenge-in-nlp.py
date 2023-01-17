### Jupyter setup to expand cell display to 100% width on your screen (optional). Change this cell to code to change the HTML view
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn import neighbors

import statsmodels.api as sm
import statsmodels.formula.api as smf
%matplotlib inline
plt.style.use('seaborn-white')
import codecs
infile = codecs.open('../input/globalenglish_essay_scoring.csv', 'r', encoding='latin1')
outfile = codecs.open('Gessay.csv', 'w', encoding='utf-8')
for line in infile:
     outfile.write(line)
infile.close()
outfile.close()

Ges=pd.read_csv('Gessay.csv')
#Ges= open('globalenglish_essay_scoring.csv', mode='r+')
Ges.info()
Ges.head()
Ges.describe()
Ges.hist(bins=80, figsize=(20,15))
plt.show()
corr_matrix = Ges.corr()
print(corr_matrix)
s = sns.PairGrid(Ges)
s.map(plt.scatter)
num = len(Ges.essay_id) #Total number of elements
print('Length of column is: ',num)
print(Ges.essay_id)
num = len(Ges.essay_set) #Total number of elements
print('Length of column is: ',num)
print(Ges.essay_set)
num = len(Ges.essay) #Total number of elements
print('Length of column is: ',num)
print(Ges.essay)
num = len(Ges.rater1_domain1) #Total number of elements
print('Length of column is: ',num)
print(Ges.rater1_domain1)
num = len(Ges.rater2_domain1) #Total number of elements
print('Length of column is: ',num)
print(Ges.rater2_domain1)
esi=Ges["essay_set"].value_counts() # This tells us how many elements in each set
print(esi)
plt.figure(figsize=(8,4))
sns.barplot(esi.index, esi.values, alpha=0.8)
plt.ylabel('Essay Set', fontsize=12)
plt.xlabel('Set name', fontsize=12)
plt.show()

esp=Ges["essay"].value_counts() # Essays
#print(esp)
Ges["rater2_domain1"].value_counts()
Ges["rater1_domain1"].value_counts()
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

tqdm.pandas()
color = sns.color_palette()

nltk.download('stopwords')

%matplotlib inline

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
fill_na = ['essay_set', 'essay']
Ges.fillna({value:'NaN' for value in fill_na}, inplace=True)

X_s= Ges[["essay_set" , "essay"]]
y_s = Ges["domain1_score"]


ts=0.8 # Chnging test size split

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=ts, random_state=42)
print (X_train.shape)
print (X_test.shape)
X_train.head()
alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Clean text
punctuation = ['.', '..', '...', ',', ':', ';', '-', '*', '"', '!', '?']
def clean_text(x):
    x.lower()
    for p in punctuation:
        x.replace(p, '')
    return x


X_train['text_cleaned'] = X_train['essay'].apply(lambda x: clean_text(x))
X_test['text_cleaned'] = X_test['essay'].apply(lambda x: clean_text(x))


# Count Vectorizer
cvect = CountVectorizer(ngram_range=(1, 3), stop_words='english')
cvect.fit(pd.concat((X_train['text_cleaned'], X_test['text_cleaned']), axis=0))
cvect_train = cvect.transform(X_train['text_cleaned'])
cvect_test = cvect.transform(X_test['text_cleaned'])

# TFIDF
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tfidf.fit(pd.concat((X_train['text_cleaned'], X_test['text_cleaned']), axis=0))
tfidf_train = tfidf.transform(X_train['text_cleaned'])
tfidf_test = tfidf.transform(X_test['text_cleaned'])

def extract_features(df):
    df['len'] = df['essay'].apply(lambda x: len(x))
    df['n_words'] = df['essay'].apply(lambda x: len(x.split(' ')))
    df['n_.'] = df['essay'].str.count('\.')
    df['n_...'] = df['essay'].str.count('\...')
    df['n_,'] = df['essay'].str.count('\,')
    df['n_:'] = df['essay'].str.count('\:')
    df['n_;'] = df['essay'].str.count('\;')
    df['n_-'] = df['essay'].str.count('\-')
    df['n_?'] = df['essay'].str.count('\?')
    df['n_!'] = df['essay'].str.count('\!')
    df['n_\''] = df['essay'].str.count('\'')
    df['n_"'] = df['essay'].str.count('\"')

    # First words in a sentence
    df['n_The '] = df['essay'].str.count('The ')
    df['n_I '] = df['essay'].str.count('I ')
    df['n_It '] = df['essay'].str.count('It ')
    df['n_He '] = df['essay'].str.count('He ')
    df['n_Me '] = df['essay'].str.count('Me ')
    df['n_She '] = df['essay'].str.count('She ')
    df['n_We '] = df['essay'].str.count('We ')
    df['n_They '] = df['essay'].str.count('They ')
    df['n_You '] = df['essay'].str.count('You ')

    # Find numbers of different combinations
    for c in tqdm(alphabet.upper()):
        df['n_' + c] = df['essay'].str.count(c)
        df['n_' + c + '.'] = df['essay'].str.count(c + '\.')
        df['n_' + c + ','] = df['essay'].str.count(c + '\,')

        for c2 in alphabet:
            df['n_' + c + c2] = df['essay'].str.count(c + c2)
            df['n_' + c + c2 + '.'] = df['essay'].str.count(c + c2 + '\.')
            df['n_' + c + c2 + ','] = df['essay'].str.count(c + c2 + '\,')

    for c in tqdm(alphabet):
        df['n_' + c + '.'] = df['essay'].str.count(c + '\.')
        df['n_' + c + ','] = df['essay'].str.count(c + '\,')
        df['n_' + c + '?'] = df['essay'].str.count(c + '\?')
        df['n_' + c + ';'] = df['essay'].str.count(c + '\;')
        df['n_' + c + ':'] = df['essay'].str.count(c + '\:')

        for c2 in alphabet:
            df['n_' + c + c2 + '.'] = df['essay'].str.count(c + c2 + '\.')
            df['n_' + c + c2 + ','] = df['essay'].str.count(c + c2 + '\,')
            df['n_' + c + c2 + '?'] = df['essay'].str.count(c + c2 + '\?')
            df['n_' + c + c2 + ';'] = df['essay'].str.count(c + c2 + '\;')
            df['n_' + c + c2 + ':'] = df['essay'].str.count(c + c2 + '\:')
            df['n_' + c + ', ' + c2] = df['essay'].str.count(c + '\, ' + c2)

    # And now starting processing of cleaned text
    for c in tqdm(alphabet):
        df['n_' + c] = df['text_cleaned'].str.count(c)
        df['n_' + c + ' '] = df['text_cleaned'].str.count(c + ' ')
        df['n_' + ' ' + c] = df['text_cleaned'].str.count(' ' + c)

        for c2 in alphabet:
            df['n_' + c + c2] = df['text_cleaned'].str.count(c + c2)
            df['n_' + c + c2 + ' '] = df['text_cleaned'].str.count(c + c2 + ' ')
            df['n_' + ' ' + c + c2] = df['text_cleaned'].str.count(' ' + c + c2)
            df['n_' + c + ' ' + c2] = df['text_cleaned'].str.count(c + ' ' + c2)

            for c3 in alphabet:
                df['n_' + c + c2 + c3] = df['text_cleaned'].str.count(c + c2 + c3)
                # df['n_' + c + ' ' + c2 + c3] = df['text_cleaned'].str.count(c + ' ' + c2 + c3)
                # df['n_' + c + c2 + ' ' + c3] = df['text_cleaned'].str.count(c + c2 + ' ' + c3)

    df['n_the'] = df['text_cleaned'].str.count('the ')
    df['n_ a '] = df['text_cleaned'].str.count(' a ')
    df['n_appear'] = df['text_cleaned'].str.count('appear')
    df['n_little'] = df['text_cleaned'].str.count('little')
    df['n_was '] = df['text_cleaned'].str.count('was ')
    df['n_one '] = df['text_cleaned'].str.count('one ')
    df['n_two '] = df['text_cleaned'].str.count('two ')
    df['n_three '] = df['text_cleaned'].str.count('three ')
    df['n_ten '] = df['text_cleaned'].str.count('ten ')
    df['n_is '] = df['text_cleaned'].str.count('is ')
    df['n_are '] = df['text_cleaned'].str.count('are ')
    df['n_ed'] = df['text_cleaned'].str.count('ed ')
    df['n_however'] = df['text_cleaned'].str.count('however')
    df['n_ to '] = df['text_cleaned'].str.count(' to ')
    df['n_into'] = df['text_cleaned'].str.count('into')
    df['n_about '] = df['text_cleaned'].str.count('about ')
    df['n_th'] = df['text_cleaned'].str.count('th')
    df['n_er'] = df['text_cleaned'].str.count('er')
    df['n_ex'] = df['text_cleaned'].str.count('ex')
    df['n_an '] = df['text_cleaned'].str.count('an ')
    df['n_ground'] = df['text_cleaned'].str.count('ground')
    df['n_any'] = df['text_cleaned'].str.count('any')
    df['n_silence'] = df['text_cleaned'].str.count('silence')
    df['n_wall'] = df['text_cleaned'].str.count('wall')

    df.drop(['essay', 'text_cleaned'], axis=1, inplace=True)



print('Processing train...')
extract_features(X_train)
print('Processing test...')
extract_features(X_test)

print('train.shape = ' + str(X_train.shape) + ', test.shape = ' + str(X_test.shape))
# Drop non-relevant columns
print('Searching for columns with non-changing values...')
counts = X_train.sum(axis=0)
cols_to_drop = counts[counts == 0].index.values
X_train.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)
print('Dropped ' + str(len(cols_to_drop)) + ' columns.')
print('train.shape = ' + str(X_train.shape) + ', test.shape = ' + str(X_test.shape))

print('Searching for columns with low STD...')
counts = X_train.std(axis=0)
cols_to_drop = counts[counts < 0.01].index.values
X_train.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)
print('Dropped ' + str(len(cols_to_drop)) + ' columns.')
print('train.shape = ' + str(X_train.shape) + ', test.shape = ' + str(X_test.shape))

import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

tqdm.pandas()





# K-fold 

# New Split train dataset on train and CV
X = X_s
y = y_train
X_test = X_test

p_valid = []
p_test = []


kf = KFold(n_splits=5, shuffle=False, random_state=0)
# for train_index in X:
for train_index in range(5):
    X_train = X_train
    y_train = y

    # LightGBM
    d_train = lgb.Dataset(X_train, label=y_train)
    #d_valid = lgb.Dataset(X_valid, label=y_valid)

    params = {
        'max_depth': 27, 
        'learning_rate': 0.1,
        'verbose': 0, 
        'early_stopping_round': 50,
        'metric': 'multi_logloss',
        'objective': 'multiclass',
        'num_classes': 10,
        'nthread': 1
    }
    n_estimators = 5000
    model = lgb.train(params, d_train, n_estimators)

    p_valid.append(model.predict(X_train, num_iteration=model.best_iteration))
    acc = accuracy_score(y_train, np.argmax(p_valid[-1], axis=1))
    logloss = log_loss(y_train, p_valid[-1])
    print('LGB:\tAccuracy = ' + str(round(acc, 6)) + ',\tLogLoss = ' + str(round(logloss, 6)))
    p_test.append(model.predict(X_test, num_iteration=model.best_iteration))

    # MultinomialNB Count Vectorizer
    X_train = cvect_train[train_index]
    y_train = y[train_index]
    print('X_train.shape = ' + str(X_train.shape) + ', y_train.shape = ' + str(y_train.shape))
    #print('X_valid.shape = ' + str(X_valid.shape) + ', y_valid.shape = ' + str(y_valid.shape))

    model = MultinomialNB()
    model.fit(X_train, y_train)
    p_valid.append(model.predict_proba(X_train))
    acc = accuracy_score(y_train, np.argmax(p_valid[-1], axis=1))
    logloss = log_loss(y_train, p_valid[-1])
    print('MNBc:\tAccuracy = ' + str(round(acc, 6)) + ',\tLogLoss = ' + str(round(logloss, 6)))
    p_test.append(model.predict_proba(cvect_test))
    # break

# Ensemble
print('Ensemble contains ' + str(len(p_valid)) + ' models.')
p_test_ens = np.mean(p_test, axis=0)

kf = KFold(n_splits=10
           , shuffle=False, random_state=0)
# niter =0
for train_index, valid_index in kf.split(X):

            # MultinomialNB Count Vectorizer
            X_train = cvect_train
            y_train = y_train
            print('X_train.shape = ' + str(X_train.shape) + ', y_train.shape = ' + str(y_train.shape))
                #print('X_valid.shape = ' + str(X_valid.shape) + ', y_valid.shape = ' + str(y_valid.shape))

            model = MultinomialNB()
            model.fit(X_train, y_train)
            p_valid.append(model.predict_proba(X_train))
            acc = accuracy_score(y_train, np.argmax(p_valid[-1], axis=1))
            logloss = log_loss(y_train, p_valid[-1])
            print('MNBc:\tAccuracy = ' + str(round(acc, 6)) + ',\tLogLoss = ' + str(round(logloss, 6)))
            

import itertools
from sklearn.metrics import confusion_matrix



conf_mat = confusion_matrix(y_train, np.argmax(model.predict_proba(X_train),axis=1))

print(conf_mat)





