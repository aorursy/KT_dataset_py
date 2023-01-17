!pip install pyspellchecker
import pandas as pd

import spellchecker as spc

import matplotlib.pyplot as plt

import sklearn

from sklearn.linear_model import LogisticRegression,RidgeClassifier

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import scipy

from sklearn.decomposition import NMF

from tqdm.notebook import tqdm



np.random.seed(31415)



spell = spc.SpellChecker()
def add_location_to_text(row):

    """add the location to the text of the tweet"""

    if row['location'] is not np.nan:

        return row['text']+" "+row['location']

    return row['text']



def add_keywords_to_text(row):

    """add the keyword to the text of the tweet"""

    if row['keyword'] is not np.nan:

        return row['text']+" "+row['keyword']

    return row['text']





def add_sp1(text):

    """ number of mispelled words"""

    def prepocess_text_for_spell(words):

        words = list(filter(lambda x: len(x)>0, words))

        return list(filter(lambda x:

                not x.startswith("#") and

                x[0] != x[0].capitalize(), words))

    words = prepocess_text_for_spell(text.split(" "))

    return len(spell.unknown(words))



def add_wc(text):

    """word count"""

    return len(text.split(' '))



def number_hash(text):

    """ number of hastag """

    words = list(filter(lambda x: len(x)>0, text.split(' ')))

    return len(list(filter(lambda x: x.startswith('#'), words)))



def number_of_chars(text):

    """ number of chars in the tweet """

    return len(text)



def has_keyword(keyword):

    """ boolean to check if the tweet had or not a keyword"""

    return int(not keyword is np.nan)



def has_location(location):

    """ boolean to check if the tweet had or not a location"""

    return int(not location is np.nan)



def apply_all_feature(df):

    """ which extra feature to apply to the dataset"""

    df.loc[:,'sp1'] = df['text'].apply(add_sp1).values

    df.loc[:,'wc'] = df['text'].apply(add_wc).values

    df.loc[:,'hst'] = df['text'].apply(number_hash).values

    df.loc[:,'ch'] = df['text'].apply(number_of_chars).values

    df.loc[:,'loc'] = df['location'].apply(has_location).values

    df.loc[:,'text'] = df.apply(add_location_to_text,axis=1)

    df.loc[:,'text'] = df.apply(add_keywords_to_text,axis=1)

    df.loc[:,'hl'] = df.apply(has_location,axis=1)

    df.loc[:,'hk'] = df.apply(has_keyword,axis=1)



def build_features(df_train,df_test,use_extra=True,use_nmf=False):

    """ build the final features to run classical regression algos """

    Y_train = df_train['target'].values

    n_train = len(Y_train)

    n_test = len(df_test)

    

    # append the test and train data to vectorize the entire df

    df_train = df_train.drop('target',axis=1)

    

    # vectorize text

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english',strip_accents='ascii',min_df=0.,ngram_range=(1,2))

    df = df_train.append(df_test, ignore_index = True, sort=False)  

    assert len(df)==n_train+n_test



    df.loc[:,'text'] = df['text'].apply(lambda x: x.lower())

    X = vectorizer.fit_transform(df['text'])

    if use_extra:

        # use or not the extra hand coded features

        apply_all_feature(df)

        X_custom = sklearn.preprocessing.scale(df[['sp1','wc','hst','ch','loc']].values,with_mean=False)

        X = scipy.sparse.hstack((X,X_custom)).tocsr()



    if use_nmf:

        # to factorize the feature space (and the log feature space) into a linear approx

        

        model = NMF(n_components=15, init='random', random_state=0)

        X = scipy.sparse.hstack((X,model.fit_transform(X),model.fit_transform(X.log1p()))).tocsr()



    X_train = X[:n_train]

    X_test = X[n_train:,:]

    

    if 'target' in df_test.columns:

        Y_test = df_test['target']

        return X_train,X_test,Y_train,Y_test

    else:

        return X_train,X_test,Y_train,None
from sklearn.linear_model import RidgeClassifier, Lasso

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC,SVC



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



df_train,df_test = sklearn.model_selection.train_test_split(train,test_size=0.33)

X_train,X_test,Y_train,Y_test = build_features(df_train,df_test,use_extra=True,use_nmf=True)
rreg = RidgeClassifier(alpha=.8,solver='sag')

lass = Lasso(alpha=.01)

rf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=100)

svm = LinearSVC(penalty='l2',max_iter=10000)

svm = SVC(kernel='linear',max_iter=10000)
rreg.fit(X_train, Y_train)

Y_pred = rreg.predict(X_test)

print(sklearn.metrics.classification_report(Y_test,Y_pred))
lass.fit(X_train, Y_train)

Y_pred = list(map(lambda x: x>=.5, lass.predict(X_test)))

print(sklearn.metrics.classification_report(Y_test,Y_pred))
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

print(sklearn.metrics.classification_report(Y_test,Y_pred))
svm.fit(X_train, Y_train)

Y_pred = svm.predict(X_test)

print(sklearn.metrics.classification_report(Y_test,Y_pred))
x_train,x_test,y_train,y_test = build_features(train,

                                               test,

                                               use_extra = True,

                                               use_nmf = True)



kf = KFold(n_splits=5)
F = []

A = np.round(np.linspace(0,1,11),2)

for a in tqdm(A):

    rreg = RidgeClassifier(alpha=a,solver='sag')

    f1 = []

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):

        kf_x_train, kf_y_train = x_train[train_index], y_train[train_index]

        kf_x_val, kf_y_val = x_train[test_index], y_train[test_index]



        rreg.fit(kf_x_train, kf_y_train)

        y_pred = rreg.predict(kf_x_val)



        res = sklearn.metrics.classification_report(kf_y_val,y_pred, output_dict=True)

        f1.append(res['1']['f1-score'])

    F.append(f1)
boxplot = pd.DataFrame(np.array(F).T,columns=A).boxplot(column=list(A))

plt.xlabel('regularizer')

plt.ylabel('regularizer')
X_train,X_test,Y_train,Y_test = build_features(train,test,use_extra=True,use_nmf=True)
rreg = RidgeClassifier(alpha=.3,solver='sag')

rreg.fit(X_train, Y_train)



sub = rreg.predict(X_test)



# save submission file

pd.DataFrame(np.array([test['id'].values,sub]).T,columns=['id','target']).to_csv('submission.csv',sep=',',

                                                                                             header=True,

                                                                                             index=False)