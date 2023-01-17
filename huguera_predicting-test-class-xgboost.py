import pandas as pd

import numpy as np

seed = 42

np.random.seed(seed)



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.datasets import dump_svmlight_file,load_svmlight_file

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
def instantiate_xgb(rounds=100, lr = 0.1, depth=3, sub=1, col=1, seed=seed) :

    clf = XGBClassifier( learning_rate=           lr,

                         n_estimators=           rounds,

                         max_depth=              depth,

                         subsample=              sub,

                         colsample_bytree=       col,

                         objective=              'multi:softmax',

                         seed=                   seed)

                     

    return clf



def get_count_vectorizer(key, train, test, components=25, iters=25):

    print ('Count Vectorizer: {}'.format(key))

    count = CountVectorizer(analyzer=u'char', ngram_range=(1, 3)).fit(train[key].apply(str))

    X_train = count.transform(train[key].apply(str))

    X_test = count.transform(test[key].apply(str))

    

    print ('TruncatedSVD: {}'.format(key))

    svd = TruncatedSVD(n_components=components, n_iter=iters, random_state=seed

                                     ).fit(X_train)

    

    X_train = svd.transform(X_train)

    X_test = svd.transform(X_test)

    

    print ('Shapes: {}\t{}'.format(X_train.shape, X_test.shape))

    return X_train, X_test



def get_tfidf(key, train, test, components=25, iters=25):

    print ('TFIDF: {}'.format(key))

    tfidf = TfidfVectorizer(

        min_df=5, strip_accents='unicode', lowercase =True,

        analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 

        smooth_idf=True, sublinear_tf=True, stop_words = 'english'

        ).fit(train[key].apply(str))



    X_train = tfidf.transform(train[key].apply(str))

    X_test = tfidf.transform(test[key].apply(str))

    

    print ('TruncatedSVD: {}'.format(key))

    svd = TruncatedSVD(n_components=25, n_iter=25, random_state=seed).fit(X_train)

    X_train = svd.transform(X_train)

    X_test = svd.transform(X_test)

    

    print ('Shapes: {}\t{}'.format(X_train.shape, X_test.shape))

    return X_train, X_test
x_train = pd.read_csv('../input/en_train.csv', nrows=100000)

x_test = pd.read_csv('../input/en_test.csv')
y_train = pd.factorize(x_train['class'])

x_train = x_train.drop(['class'], axis=1)

key = 'before'
train_cnt, test_cnt = get_count_vectorizer(key, x_train, x_test)
train_tfidf, test_tfidf = get_tfidf(key, x_train, x_test)
x_train[key+'_len'] = x_train[key].map(lambda x: len(str(x)))

x_test[key+'_len'] = x_test[key].map(lambda x: len(str(x)))



x_train['sentence_length'] = x_train.groupby(['sentence_id'])['sentence_id'].transform(np.size)

x_test['sentence_length'] = x_test.groupby(['sentence_id'])['sentence_id'].transform(np.size)
x_train = np.concatenate((x_train.drop(['before', 'after','sentence_id'],axis=1), 

                          train_cnt,

                          train_tfidf), axis=1)

x_test = np.concatenate((x_test.drop(['before', 'sentence_id'], axis=1),

                         test_cnt,

                         test_tfidf), axis=1)



print ('Shapes: {}\t{}'.format(x_train.shape, x_test.shape))

del train_cnt, train_tfidf, test_cnt, test_tfidf
X_train, X_valid, Y_train, Y_valid = train_test_split(

    x_train, y_train[0], test_size=0.3, random_state=seed)



clf = instantiate_xgb(rounds=100, seed=seed)



clf.fit(X_train, Y_train, 

        eval_set=[(X_valid, Y_valid)],

        eval_metric='merror',

        early_stopping_rounds=30,

        verbose=10)
preds = clf.predict(x_test)

y = pd.Series(preds).apply(lambda x: y_train[1][x])

y.to_csv('./test_class_preds.csv', index=False)
df = pd.DataFrame()

df['valid_class'] = pd.Series(Y_valid).apply(lambda x: y_train[1][x])

df['valid_preds'] = pd.Series(clf.predict(X_valid)).apply(lambda x: y_train[1][x])



print(df.head(30))