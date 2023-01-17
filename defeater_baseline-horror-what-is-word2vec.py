import numpy as np

import pandas as pd 

from subprocess import check_output

from gensim.models import Word2Vec



from nltk.tokenize import RegexpTokenizer

from nltk import WordNetLemmatizer

from nltk.corpus import stopwords



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier



from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, train_test_split



from sklearn.metrics import f1_score, accuracy_score



import xgboost as xgb



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer



alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')

lemmatizer = WordNetLemmatizer()

stop = stopwords.words('english')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test_id = test['id'].values



author_mapping = {'EAP':0, 'HPL':1, 'MWS':2}

y_train = train['author'].map(author_mapping).values
# data = [[lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop] for sent in train.text.values]
vectorizers = [ # ('3-gram TF-IDF Vectorizer on words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),

                # ('3-gram Count Vectorizer on words', CountVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),

                # ('3-gram Hashing Vectorizer on words', HashingVectorizer(ngram_range=(1, 5), analyzer='word', binary=False)),

                ('TF-IDF + SVD', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),

                                 ('svd', TruncatedSVD(n_components=150)),

                                ])),

                ('TF-IDF + SVD + Normalizer', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),

                                 ('svd', TruncatedSVD(n_components=150)),

                                 ('norm', Normalizer()),

                                ]))

              ]
estimators = [

              (KNeighborsClassifier(n_neighbors=3), 'K-Nearest Neighbors', 'yellow'),

              (SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False), 'Support Vector Machine', 'red'),

              (LogisticRegression(tol=1e-8, penalty='l2', C=0.1), 'Logistic Regression', 'green'),

              (MultinomialNB(), 'Naive Bayes', 'magenta'),

              (RandomForestClassifier(n_estimators=10, criterion='gini'), 'Random Forest', 'gray'),

              (None, 'XGBoost', 'pink')

]
params = {}

params['objective'] = 'multi:softprob'

params['eta'] = 0.1

params['max_depth'] = 3

params['silent'] = 1

params['num_class'] = 3

params['eval_metric'] = 'mlogloss'

params['min_child_weight'] = 1

params['subsample'] = 0.8

params['colsample_bytree'] = 0.3

params['seed'] = 0
def vectorize():

    

    test_size = 0.3



    train_split, test_split = train_test_split(train, test_size=test_size)



    y_train_split = train_split['author'].map(author_mapping).values

    y_test_split = test_split['author'].map(author_mapping).values

    

    for vectorizer in vectorizers:

        print(vectorizer[0] + '\n')

        X = vectorizer[1].fit_transform(train.text.values)

        X_train, X_test = train_test_split(X, test_size=test_size)

        for estimator in estimators:

            if estimator[1] == 'XGBoost': 

                xgtrain = xgb.DMatrix(X_train, y_train_split)

                xgtest = xgb.DMatrix(X_test)

                model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)

                predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit).argmax(axis=1)

            else:

                estimator[0].fit(X_train, y_train_split)

                predictions = estimator[0].predict(X_test)

            print(accuracy_score(predictions, y_test_split), estimator[1])
train['num_words'] = train.text.apply(lambda x: len(str(x).split()))

test['num_words'] = test.text.apply(lambda x: len(str(x).split()))



train['num_unique_words'] = train.text.apply(lambda x: len(set(str(x).split())))

test['num_unique_words'] = test.text.apply(lambda x: len(set(str(x).split())))



train['num_chars'] = train.text.apply(lambda x: len(str(x)))

test['num_chars'] = test.text.apply(lambda x: len(str(x)))



train['num_stopwords'] = train.text.apply(lambda x: len([w for w in str(x).lower().split() if w in stop]))

test['num_stopwords'] = test.text.apply(lambda x: len([w for w in str(x).lower().split() if w in stop]))



train['mean_word_len'] = train.apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test['mean_word_len'] = test.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in train.text.values]

test_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in test.text.values]
vectorizer = CountVectorizer(ngram_range=(1,7), analyzer='char')



full = vectorizer.fit_transform(train_text + test_text)

X_train = vectorizer.transform(train_text)

X_test = vectorizer.transform(test_text)



pred_full_test = 0

pred_train = np.zeros([train.shape[0], 3])



for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):

    dev_X, val_X = X_train[dev_index], X_train[val_index]

    dev_y, val_y = y_train[dev_index], y_train[val_index]

    model = MultinomialNB()

    model.fit(dev_X, dev_y)

    pred_full_test = pred_full_test + model.predict_proba(X_test)

    pred_train[val_index,:] = model.predict_proba(val_X)



pred_full_test = pred_full_test / 5.



train['CH_EAP'] = pred_train[:,0]

train['CH_HPL'] = pred_train[:,1]

train['CH_MWS'] = pred_train[:,2]

test['CH_EAP'] = pred_full_test[:,0]

test['CH_HPL'] = pred_full_test[:,1]

test['CH_MWS'] = pred_full_test[:,2]
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))

full = vectorizer.fit_transform(train_text + test_text)

X_train = vectorizer.transform(train_text)

X_test = vectorizer.transform(test_text)



pred_full_test = 0

pred_train = np.zeros([train.shape[0], 3])



for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):

    dev_X, val_X = X_train[dev_index], X_train[val_index]

    dev_y, val_y = y_train[dev_index], y_train[val_index]

    model = MultinomialNB()

    model.fit(dev_X, dev_y)

    pred_full_test = pred_full_test + model.predict_proba(X_test)

    pred_train[val_index,:] = model.predict_proba(val_X)



pred_full_test = pred_full_test / 5.



train['C_EAP'] = pred_train[:,0]

train['C_HPL'] = pred_train[:,1]

train['C_MWS'] = pred_train[:,2]

test['C_EAP'] = pred_full_test[:,0]

test['C_HPL'] = pred_full_test[:,1]

test['C_MWS'] = pred_full_test[:,2]
vectorizer = TfidfVectorizer(ngram_range=(1,5), analyzer='char')

full = vectorizer.fit_transform(train_text + test_text)

X_train = vectorizer.transform(train_text)

X_test = vectorizer.transform(test_text)



pred_full_test = 0

pred_train = np.zeros([train.shape[0], 3])



for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):

    dev_X, val_X = X_train[dev_index], X_train[val_index]

    dev_y, val_y = y_train[dev_index], y_train[val_index]

    model = MultinomialNB()

    model.fit(dev_X, dev_y)

    pred_full_test = pred_full_test + model.predict_proba(X_test)

    pred_train[val_index,:] = model.predict_proba(val_X)



pred_full_test = pred_full_test / 5.



train['T_EAP'] = pred_train[:,0]

train['T_HPL'] = pred_train[:,1]

train['T_MWS'] = pred_train[:,2]

test['T_EAP'] = pred_full_test[:,0]

test['T_HPL'] = pred_full_test[:,1]

test['T_MWS'] = pred_full_test[:,2]
svd = TruncatedSVD(n_components=20, algorithm='arpack')

svd.fit(full)

train_svd = pd.DataFrame(svd.transform(X_train))

test_svd = pd.DataFrame(svd.transform(X_test))

    

train_svd.columns = ['SVD_' + str(i) for i in range(20)]

test_svd.columns = ['SVD_' + str(i) for i in range(20)]

train = pd.concat([train, train_svd], axis=1)

test = pd.concat([test, test_svd], axis=1)
train = train.drop(['id', 'text', 'author'], axis=1)

test = test.drop(['id', 'text'], axis=1)
# train = pd.read_csv('../input/train.csv')

# test = pd.read_csv('../input/test.csv')

# test_id = test['id'].values



# author_mapping = {'EAP':0, 'HPL':1, 'MWS':2}

# y_train = train['author'].map(author_mapping).values
NUM_FEATURES = 100



model = Word2Vec(train_text + test_text, min_count=2, size=NUM_FEATURES, window=4, sg=1, alpha=1e-4, workers=4)
len(model.wv.vocab)
model.most_similar('raven')
def get_feature_vec(tokens, num_features, model):

    featureVec = np.zeros(shape=(1, num_features), dtype='float32')

    missed = 0

    for word in tokens:

        try:

            featureVec = np.add(featureVec, model[word])

        except KeyError:

            missed += 1

            pass

    if len(tokens) - missed == 0:

        return np.zeros(shape=(num_features), dtype='float32')

    return np.divide(featureVec, len(tokens) - missed).squeeze()
train_vectors = []

for i in train_text:

    train_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))
test_vectors = []

for i in test_text:

    test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))
full_vectors = []

for i in train_text + test_text:

    full_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))
svd = TruncatedSVD(n_components=30, algorithm='arpack')



svd.fit(full_vectors)

train_svd = pd.DataFrame(svd.transform(np.array(train_vectors)))

test_svd = pd.DataFrame(svd.transform(np.array(test_vectors)))

    

train_svd.columns = ['W2V_' + str(i) for i in range(30)]

test_svd.columns = ['W2V_' + str(i) for i in range(30)]



train = pd.concat([train, train_svd], axis=1)

test = pd.concat([test, test_svd], axis=1)
pred_full_test = 0

pred_train = np.zeros([train.shape[0], 3])

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train):

    dev_X, val_X = train.loc[dev_index], train.loc[val_index]

    dev_y, val_y = y_train[dev_index], y_train[val_index]

    xgtrain = xgb.DMatrix(dev_X, dev_y)

    xgtest = xgb.DMatrix(test)

    model = xgb.train(params=list(params.items()), dtrain=xgtrain, num_boost_round=1000)

    predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

    pred_full_test = pred_full_test + predictions

pred_full_test = pred_full_test / 5.
# xgtrain = xgb.DMatrix(train_vectors, y_train)

# xgtest = xgb.DMatrix(test_vectors)

# model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)

# probs = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
author = pd.DataFrame(pred_full_test)



final = pd.DataFrame()

final['id'] = test_id

final['EAP'] = author[0]

final['HPL'] = author[1]

final['MWS'] = author[2]



final.to_csv('submission.csv', sep=',',index=False)