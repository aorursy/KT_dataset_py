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



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submissions = pd.read_csv('../input/sample_submission.csv')



train.comment_text.fillna('', inplace=True)

test.comment_text.fillna('', inplace=True)
train.head()
plt.figure(figsize=(12,6))

sns.barplot(train.toxic.value_counts().index, train.toxic.value_counts().values, alpha=0.8)

plt.ylabel('Amount of class instances', fontsize=16)

plt.xlabel('Class', fontsize=16)

plt.show();
plt.figure(figsize=(12,6))

sns.barplot(train.obscene.value_counts().index, train.obscene.value_counts().values, alpha=0.8)

plt.ylabel('Amount of class instances', fontsize=16)

plt.xlabel('Class', fontsize=16)

plt.show();
vectorizers = [ ('3-gram TF-IDF Vectorizer on words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),

               ('3-gram Count Vectorizer on words', CountVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),

               ('3-gram Hashing Vectorizer on words', HashingVectorizer(ngram_range=(1, 5), analyzer='word', binary=False)),

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



    train_split, test_split = train_test_split(train[:10], test_size=test_size)



    for column in range(2, len(train.axes[1])):

        for vectorizer in vectorizers:

            print(vectorizer[0] + '\n')

            X = vectorizer[1].fit_transform(train.comment_text.values)

            X_train, X_test = train_test_split(X, test_size=test_size)

            for estimator in estimators:

                if estimator[1] == 'XGBoost': 

                    xgtrain = xgb.DMatrix(X_train, train_split.iloc[:,column].values)

                    xgtest = xgb.DMatrix(X_test)

                    model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)

                    predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit).argmax(axis=1)

                else:

                    estimator[0].fit(X_train, train_split.iloc[:,column].values)

                    predictions = estimator[0].predict(X_test)

                print(accuracy_score(predictions, test_split.iloc[:,column].values), estimator[1])
train_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in train.comment_text.values]

test_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in test.comment_text.values]
vectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer='word')



full = vectorizer.fit_transform(train_text + test_text)

X_train = vectorizer.transform(train_text)

X_test = vectorizer.transform(test_text)



# NUM_FEATURES = 100



# model = Word2Vec(train_text + test_text, min_count=2, size=NUM_FEATURES, window=4, sg=1, alpha=1e-4, workers=4)



# def get_feature_vec(tokens, num_features, model):

#     featureVec = np.zeros(shape=(1, num_features), dtype='float32')

#     missed = 0

#     for word in tokens:

#         try:

#             featureVec = np.add(featureVec, model[word])

#         except KeyError:

#             missed += 1

#             pass

#     if len(tokens) - missed == 0:

#         return np.zeros(shape=(num_features), dtype='float32')

#     return np.divide(featureVec, len(tokens) - missed).squeeze()



# train_vectors = []

# for i in train_text:

#     train_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))

    

# test_vectors = []

# for i in test_text:

#     test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))
X_train = train_vectors

X_test = test_vectors
start = 2

predictions = np.zeros((len(test), len(train.axes[1]) - start))



for column in range(start, len(train.axes[1])):

    y_train = train.iloc[:,column].values

    # estimator = LogisticRegression(C = 0.01)

    # estimator.fit(X_train, y_train)

    # predictions[:,column - start] = estimator.predict_proba(X_test)[:,1]

    xgtrain = xgb.DMatrix(X_train, y_train)

    xgtest = xgb.DMatrix(X_test)

    model = xgb.train(params=list(params.items()), dtrain=xgtrain, num_boost_round=500)

    predictions[:,column - start]  = model.predict(xgtest, ntree_limit=model.best_ntree_limit)[:,1]
result = pd.concat([pd.DataFrame({'id': submissions['id']}), pd.DataFrame(predictions, columns = train.columns.values[2:])], axis=1)

result.to_csv('submission.csv', index=False)