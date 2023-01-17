# basic text pre-processing

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer, LancasterStemmer

from nltk import pos_tag



# core modules

from pandas import DataFrame, concat, options

import numpy as np

import re

from collections import defaultdict, Counter



# tools and models for machine learning

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import Normalizer

from gensim.models import Word2Vec



# visualization

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import matplotlib.pyplot as plt

from matplotlib import use

import seaborn as sns

from wordcloud import WordCloud
wordnet_lemmatizer = WordNetLemmatizer()

stemmer =  LancasterStemmer()

stop = stopwords.words('english')

py.init_notebook_mode(connected=True)

options.mode.chained_assignment = None
import warnings

warnings.filterwarnings('ignore')
data = DataFrame.from_csv('../input/Combined_News_DJIA.csv').reset_index().fillna(' ')
train = data[data['Date'] < '2015-01-01']

test = data[data['Date'] > '2014-12-31']

y_train = train.Label.values

y_test = test.Label.values

col_number = 26
print('The size of the training set is {} news, the size of the test set is {} news'.format(len(train), len(test)))
train.head()
plt.figure(figsize=(12,6))

sns.barplot(data.Label.value_counts().index, data.Label.value_counts().values, alpha=0.8)

plt.ylabel('Amount of objects', fontsize=16)

plt.xlabel('Class', fontsize=16)

plt.show();
lmb_f = [lambda x: re.sub("""^b("|')""",'', str(x)),  

         lambda x: str(x).lower(),

         lambda x: str(x).replace("'",''),

         lambda x: word_tokenize(str(x)),

         lambda x: [wordnet_lemmatizer.lemmatize(str(i)) for i in x],

         lambda x: [stemmer.stem(str(i)) for i in x],

         lambda x: ' '.join(x)

        ]
def parse_trainset(data, preproc='lem'):

    if preproc == 'lem':

        lambdas = lmb_f[0:5] + lmb_f[6:]

    elif preproc == 'stem':

        lambdas = lmb_f[0:4] + lmb_f[5:]

    elif preproc == 'lem+stem':

        lambdas = lmb_f

    elif preproc == '_':

        lambdas = lmb_f[0:5]

    li = []

    for col in range(1, col_number):

        s = data.loc[:,'Top' + str(col)]

        for a in lambdas:

            s = s.apply(a)

        li.append(s)

    return li
train_lem = concat([train.drop(train.columns[2:], axis=1), DataFrame(parse_trainset(train)).transpose()], axis=1)

test_lem = concat([test.drop(test.columns[2:], axis=1), DataFrame(parse_trainset(test)).transpose()], axis=1)
train_stem = concat([train.drop(train.columns[2:], axis=1), DataFrame(parse_trainset(train, 'stem')).transpose()], axis=1)

test_stem = concat([test.drop(test.columns[2:], axis=1), DataFrame(parse_trainset(test, 'stem')).transpose()], axis=1)
train_lem_stem = concat([train.drop(train.columns[2:], axis=1), DataFrame(parse_trainset(train, 'lem+stem')).transpose()], axis=1)

test_lem_stem = concat([test.drop(test.columns[2:], axis=1), DataFrame(parse_trainset(test, 'lem+stem')).transpose()], axis=1)
train_ = concat([train.drop(train.columns[2:], axis=1), DataFrame(parse_trainset(train, '_')).transpose()], axis=1)

test_ = concat([test.drop(test.columns[2:], axis=1), DataFrame(parse_trainset(test, '_')).transpose()], axis=1)
train_stem.head()
mean_list = []

for number in range(train_.shape[0]):

    s = np.mean([len(f) for f in list(train_.loc[number][2:].values)])

    mean_list.append((s, y_train[number]))



for number2 in range(number+1, number+1+test_.shape[0]):

    s = np.mean([len(f) for f in list(test_.loc[number2][2:].values)])

    mean_list.append((s, y_test[number2-number-1]))

mean_list = sorted(mean_list)
plt.figure(figsize=(20,6))

plt.plot(range(len(mean_list))[:20], list(y_train)[:20])

plt.legend()

plt.xticks(range(20))

plt.xlabel('Average text length', size=16)

plt.show();
lists = ['obama', 'us', 'u.s.', 'u s', 'u.s', 'united', 'state', 'america']



all_sum = []

for number in range(train_.shape[0]):

    s = [i for f in list(train_.loc[number][2:].values) for i in f]

    summ = 0

    for i in lists:

        summ+=Counter(s)[i]

    all_sum.append((summ, y_train[number]))



for number2 in range(number+1, number+1+test_.shape[0]):

    s = [i for f in list(test_.loc[number2][2:].values) for i in f]

    summ = 0

    for i in lists:

        summ+=Counter(s)[i]

    all_sum.append((summ, y_test[number2-number-1]))
a = sorted([(i[0], dict(Counter(all_sum))[i]) for i in dict(Counter(all_sum)) if i[1]==0])

a1= [i[1] for i in a]

b = sorted([(i[0], dict(Counter(all_sum))[i]) for i in dict(Counter(all_sum)) if i[1]==1])

b1= [i[1] for i in b]

red = '#B2182B'

blue = '#2166AC'

width = 0.35



plt.figure(figsize=(12,6))

plt.bar(np.arange(12), a1, width, color=red, label='0 label')

plt.bar(np.arange(12), b1, width, bottom=a1, color=blue, label='1 label')

plt.legend()

plt.xticks(range(12))

plt.xlabel('Key word amount', size=16)

plt.show()
news_topics_russia = ['russian', 'russia', 'putin']

news_topics_isis = ['isil', 'isis', 'levant', 'daesh']
news_russia = []

news_isis = []



for number in range(train_.shape[0]):

    news_russia += [i for f in list(train_.loc[number][2:].values) for i in f if i.lower() in news_topics_russia]

    news_isis += [i for f in list(train_.loc[number][2:].values) for i in f if i.lower() in news_topics_isis]

for number2 in range(number+1, number+1+test_.shape[0]):

    news_russia += [i for f in list(test_.loc[number2][2:].values) for i in f if i.lower() in news_topics_russia]

    news_isis += [i for f in list(train_.loc[number][2:].values) for i in f if i.lower() in news_topics_isis]



news_counter_russia = Counter(news_russia)

news_counter_isis = Counter(news_isis)
print('Russia is mentioned {} times, and ISIS is mentioned {} times'.format(sum(news_counter_russia.values()), sum(news_counter_isis.values())))



russia_bar = go.Bar(

    x=list(news_counter_russia.keys()),

    y=list(news_counter_russia.values()),

    name='Russia',

    marker=dict(

        color='red'

    )

)

isis_bar = go.Bar(

    x=list(news_counter_isis.keys()),

    y=list(news_counter_isis.values()),

    name='ISIS',

    marker=dict(

        color='black'

    )

)



layout = go.Layout(

    title='Amount of mentioning Russia and ISIS in the news',

    xaxis=dict(

        tickfont=dict(

            size=14,

            color='black'

        )

    ),

    yaxis=dict(

        title='Amount of news containing this word',

        titlefont=dict(

            size=16,

            color='black'

        ),

        tickfont=dict(

            size=14,

            color='black'

        )

    ),

    legend=dict(

        x=0,

        y=1.0,

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)



fig = go.Figure(data=[russia_bar, isis_bar], layout=layout)

py.iplot(fig, filename='style-bar')
crisis_marker = 'crisis'
list_of_crisis = defaultdict(lambda: 0.0)



for ind in range(train_.shape[0]):

    list_of_articles = list(train_.loc[ind])[2:]

    for i in list_of_articles:

        if crisis_marker in i:

            try:

                word = pos_tag([i[i.index(crisis_marker)-1]])[0]

                if word[1]=='JJ':

                    list_of_crisis[word[0]] += 1.0

            except Exception as e:

                pass



for ind2 in range(ind+1, test_.shape[0]+1):

    list_of_articles = list(test_.loc[ind2])[2:]

    for i in list_of_articles:

        if crisis_marker in i:

            try:

                word = pos_tag([i[i.index(crisis_marker)-1]])[0]

                if word[1]=='JJ':

                    list_of_crisis[word[0]] += 1.0

            except Exception as e:

                pass
wordcloud = WordCloud(background_color='white')

wordcloud.generate_from_frequencies(dict(list_of_crisis))

plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
def make_raw_text_col(df):

    df['text'] = df['Top1'].str[1:] + ' '

    for i in df.loc[:,'Top2':'Top25']:

        df['text'] += df[i].str[1:] + ' '

    df['text'] = df['text'].str.lower().str.replace('[^a-zA-Z ]', '')

    return df
def make_raw_text_col_lem(df):

    df['text'] = df['Top1']

    for i in df.loc[:,'Top2':'Top25']:

        df['text'] += df[i]

    df['text'] = df['text'].str.lower().str.replace('[^a-zA-Z ]', '')

    return df
train = make_raw_text_col(train)

test = make_raw_text_col(test)
train_lem = make_raw_text_col_lem(train_lem)

test_lem = make_raw_text_col_lem(test_lem)



train_stem = make_raw_text_col_lem(train_stem)

test_stem = make_raw_text_col_lem(test_stem)



train_lem_stem = make_raw_text_col_lem(train_lem_stem)

test_lem_stem = make_raw_text_col_lem(test_lem_stem)
model = Word2Vec([word_tokenize(text) for text in np.hstack((train.text.values, test.text.values))], min_count=4, size=300, window=4, sg=1, alpha=1e-4)
topn=10



labels = []

tokens = []



for word in model.most_similar(crisis_marker, topn=topn):

    tokens.append(model[word[0]])

    labels.append(word[0])



    

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

new_values = tsne_model.fit_transform(tokens)



x = []

y = []

for value in new_values:

    x.append(value[0])

    y.append(value[1])



plt.figure(figsize=(5, 5))

sns.set_style('whitegrid')

plt.grid(False)

for i in range(len(x)):

    plt.scatter(x[i],y[i])

    plt.annotate(labels[i],

                fontsize=15,

                color='black',

                xy=(x[i], y[i]),

                xytext=(2, 2),

                textcoords='offset points',

                ha='right',

                va='bottom')
lambda_func_features = [

    (lambda x: len(str(x).split()), 'NumWords'),

    (lambda x: len(set(str(x).split())), 'NumUniqueWords'),

    (lambda x: len(str(x)), 'NumChars'),

    (lambda x: len([w for w in str(x).lower().split() if w in stop]), 'NumStopWords'),

    (lambda x: np.mean([len(w) for w in str(x).split()]), 'MeanWordLen'),

]
def generate_features(train, test, train_la, test_la, lambda_func, func_name):

    train_features = DataFrame([train.loc[:,'Top' + str(col)].apply(lambda_func) for col in range(1, col_number)]).transpose()

    test_features = DataFrame([test.loc[:,'Top' + str(col)].apply(lambda_func) for col in range(1, col_number)]).transpose()

    train_features.columns = [func_name + str(i) for i in range(1, col_number)]

    test_features.columns = [func_name + str(i) for i in range(1, col_number)]

    return concat([train_la, train_features], axis=1), concat([test_la, test_features], axis=1)
def create_linguistic_features(train_data, test_data):

    train_la = DataFrame()

    test_la = DataFrame()

    for lambda_func, func_name in lambda_func_features:

        train_la, test_la = generate_features(train_data, test_data, train_la, test_la, lambda_func, func_name)

    test_la = test_la.reset_index(drop=True)

    return train_la, test_la
train_la, test_la = create_linguistic_features(train, test)
train_la.head()
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
boost_rounds = 20

kfolds = 5

pred_full_test = 0



def make_xgboost_predictions(train_df, test_df):

    pred_full_test = 0

    pred_train = np.zeros([train_df.shape[0], len(set(y_train))])

    

    for dev_index, val_index in KFold(n_splits=kfolds, shuffle=True, random_state=42).split(train_df):

        dev_X, val_X = train_df.loc[dev_index], train_df.loc[val_index]

        dev_y, val_y = y_train[dev_index], y_train[val_index]

        xgtrain = xgb.DMatrix(dev_X, dev_y)

        xgtest = xgb.DMatrix(test_df)

        model = xgb.train(params=list(params.items()), dtrain=xgtrain, num_boost_round=boost_rounds)

        predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

        pred_full_test = pred_full_test + predictions

    return pred_full_test / kfolds
f1_score(make_xgboost_predictions(train_la, test_la).argmax(axis=1), y_test) 
vectorizers = [

    (CountVectorizer(stop_words='english', ngram_range=(1,5), analyzer='char'), 'CountVectorizerChar'),

    (TfidfVectorizer(stop_words='english', ngram_range=(1,5), analyzer='char'), 'TfIdfVectorizerChar'),

    (CountVectorizer(stop_words='english', ngram_range=(1,2), analyzer='word'), 'CountVectorizerWord'),

    (TfidfVectorizer(stop_words='english', ngram_range=(1,2), analyzer='word'), 'TfIdfVectorizerWord')

]
models = [#(MultinomialNB(), {'alpha':[0, 0.1, 0.5, 0.8, 1]}, "Naive Bayes"),

          (xgb.XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, 

                        colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),

                        {'max_depth':range(3,10,9), 'min_child_weight':range(1,6,5)}, "XGB")

         ]

def do_grid_search(alg, array_of_vectors, array_of_tags, parameters):

    clf = model_selection.GridSearchCV(alg, parameters, error_score=0.0)

    clf.fit(array_of_vectors, array_of_tags)

    print(clf.best_estimator_)

    return clf.best_estimator_
def create_proba_features(train_data, test_data):



    kfolds = 5

    train_vecs = DataFrame()

    test_vecs = DataFrame()



    for vec, vec_name in vectorizers:

        vectorizer = vec

        full = vectorizer.fit_transform(np.hstack((train_data.text.values, test_data.text.values)))



        X_train_raw = vectorizer.transform(train_data.text.values)

        X_test_raw = vectorizer.transform(test_data.text.values)



        normalized = Normalizer()

        normalized.fit_transform(full)

        X_train = normalized.transform(X_train_raw)

        X_test = normalized.transform(X_test_raw)



        pred_full_test = 0

        pred_train = np.zeros([train_data.shape[0], len(train_data.Label.unique())])



        for dev_index, val_index in KFold(n_splits=kfolds, shuffle=True, random_state=42).split(train_data):

            dev_X, val_X = X_train[dev_index], X_train[val_index]

            dev_y, val_y = y_train[dev_index], y_train[val_index]

            model = MultinomialNB()

            model.fit(dev_X, dev_y)

            pred_full_test = pred_full_test + model.predict_proba(X_test)

            pred_train[val_index,:] = model.predict_proba(val_X)



        pred_full_test = pred_full_test / kfolds



        train_vecs[vec_name + 'Zero'] = pred_train[:,0]

        train_vecs[vec_name + 'One'] = pred_train[:,1]

        test_vecs[vec_name + 'Zero'] = pred_full_test[:,0]

        test_vecs[vec_name + 'One'] = pred_full_test[:,1]

        

        return train_vecs, test_vecs
train_vecs, test_vecs = create_proba_features(train, test)
f1_score(make_xgboost_predictions(train_vecs, test_vecs).argmax(axis=1), y_test) 
def create_svd_features(train_data, test_data):

    svd_components = 20



    vectorizer = TfidfVectorizer(ngram_range=(1,2), analyzer='word')

    full = vectorizer.fit_transform(np.hstack((train_data.text.values, test_data.text.values)))

    X_train = vectorizer.transform(train_data.text.values)

    X_test = vectorizer.transform(test_data.text.values)



    svd = TruncatedSVD(n_components=svd_components, algorithm='arpack')

    svd.fit(full)

    train_svd = DataFrame(svd.transform(X_train))

    test_svd = DataFrame(svd.transform(X_test))



    train_svd.columns = ['SVD' + str(i) for i in range(svd_components)]

    test_svd.columns = ['SVD' + str(i) for i in range(svd_components)]

    

    return train_svd, test_svd
train_svd, test_svd = create_svd_features(train, test)
f1_score(make_xgboost_predictions(train_svd, test_svd).argmax(axis=1), y_test) 
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
def create_embedding_features(train_data, test_data):

    num_features = 100



    model = Word2Vec([word_tokenize(text) for text in np.hstack((train_data.text.values, test_data.text.values))], min_count=4, size=num_features, window=4, sg=0, alpha=1e-4)



    train_embedding_vectors = []

    for i in train_data.text.values:

        train_embedding_vectors.append(get_feature_vec(word_tokenize(i), num_features, model))



    test_embedding_vectors = []

    for i in test_data.text.values:

        test_embedding_vectors.append(get_feature_vec(word_tokenize(i), num_features, model))



    train_w2v = DataFrame(train_embedding_vectors)

    test_w2v = DataFrame(test_embedding_vectors)

    

    train_w2v.columns = ['W2V' + str(i) for i in range(num_features)]

    test_w2v.columns = ['W2V' + str(i) for i in range(num_features)]

    

    return train_w2v, test_w2v
train_w2v, test_w2v = create_embedding_features(train, test)
f1_score(make_xgboost_predictions(train_w2v, test_w2v).argmax(axis=1), y_test) 
X_train = concat([train_la, train_svd, train_vecs, train_w2v], axis=1)

X_test = concat([test_la, test_svd, test_vecs, test_w2v], axis=1)

print('Feature matrix consists of {} features'.format(len(X_train.columns.values)))
print(f1_score(make_xgboost_predictions(X_train, X_test).argmax(axis=1), y_test))
topf=200

features = X_train.columns.values

model_xgb = xgb.XGBClassifier()

model_xgb.fit(X_train, y_train)

x, y = (list(x) for x in zip(*sorted(zip(model_xgb.feature_importances_, features), reverse = False)[:topf]))

trace2 = go.Bar(

    x=x ,

    y=y,

    marker=dict(

        color=x,

        colorscale = 'Viridis',

        reversescale = True

    ),

    name='Feature importance for XGBoost',

    orientation='h',

)



layout = dict(

    title='Barplot of TOP-{} Features importances for XGBoost'.format(topf),

    width = 1000, height = 1000,

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')
preprocessed_data = [

                    (train, test, 'raw'),

                    (train_lem, test_lem, 'lem'),

                    (train_stem, test_stem, 'stem'),

                    (train_lem_stem, test_lem_stem, 'lem+stem')

]
for train, test, preprocess in preprocessed_data:

    train_la, test_la = create_linguistic_features(train, test)

    train_proba, test_proba = create_proba_features(train, test)

    train_svd, test_svd = create_svd_features(train, test)

    train_w2v, test_w2v = create_embedding_features(train, test)



    X_train = concat([train_la, train_svd, train_proba, train_w2v], axis=1)

    X_test = concat([test_la, test_svd, test_proba, test_w2v], axis=1)

    

    print('F1 with preprocessing by {} is {:0.4f}'.format(preprocess, f1_score(make_xgboost_predictions(X_train, X_test).argmax(axis=1), y_test)))

    print('Accuracy with preprocessing by {} is {:0.4f}\n'.format(preprocess, accuracy_score(make_xgboost_predictions(X_train, X_test).argmax(axis=1), y_test)))