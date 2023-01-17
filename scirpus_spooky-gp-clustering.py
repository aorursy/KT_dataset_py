import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import nltk

from nltk.corpus import stopwords

import string

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn import ensemble, metrics, model_selection, naive_bayes

color = sns.color_palette()



%matplotlib inline



eng_stopwords = set(stopwords.words("english"))

pd.options.mode.chained_assignment = None
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.1

    param['max_depth'] = 3

    param['silent'] = 1

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = child

    param['subsample'] = 0.8

    param['colsample_bytree'] = colsample

    param['seed'] = seed_val

    num_rounds = 2000



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)

    if test_X2 is not None:

        xgtest2 = xgb.DMatrix(test_X2)

        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)

    return pred_test_y, pred_test_y2, model



def runMNB(train_X, train_y, test_X, test_y, test_X2):

    model = naive_bayes.MultinomialNB()

    model.fit(train_X, train_y)

    pred_test_y = model.predict_proba(test_X)

    pred_test_y2 = model.predict_proba(test_X2)

    return pred_test_y, pred_test_y2, model
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

## Number of words in the text ##

train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))

test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))

test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))

test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test_df["num_stopwords"] = test_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text ##

train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}

train_y = train_df['author'].map(author_mapping_dict)

train_id = train_df['id'].values

test_id = test_df['id'].values



### recompute the trauncated variables again ###

train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))

test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



cols_to_drop = ['id', 'text']

train_X = train_df.drop(cols_to_drop+['author'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)



tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())



n_comp = 20

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    

train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)

test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd



tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))

tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))

pred_full_test = pred_full_test / 5.



# add the predictions as new features #

train_df["nb_cvec_eap"] = pred_train[:,0]

train_df["nb_cvec_hpl"] = pred_train[:,1]

train_df["nb_cvec_mws"] = pred_train[:,2]

test_df["nb_cvec_eap"] = pred_full_test[:,0]

test_df["nb_cvec_hpl"] = pred_full_test[:,1]

test_df["nb_cvec_mws"] = pred_full_test[:,2]



### Fit transform the tfidf vectorizer ###

tfidf_vec = CountVectorizer(ngram_range=(1,7), analyzer='char')

tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))

pred_full_test = pred_full_test / 5.



# add the predictions as new features #

train_df["nb_cvec_char_eap"] = pred_train[:,0]

train_df["nb_cvec_char_hpl"] = pred_train[:,1]

train_df["nb_cvec_char_mws"] = pred_train[:,2]

test_df["nb_cvec_char_eap"] = pred_full_test[:,0]

test_df["nb_cvec_char_hpl"] = pred_full_test[:,1]

test_df["nb_cvec_char_mws"] = pred_full_test[:,2]



### Fit transform the tfidf vectorizer ###

tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')

tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))

pred_full_test = pred_full_test / 5.



# add the predictions as new features #

train_df["nb_tfidf_char_eap"] = pred_train[:,0]

train_df["nb_tfidf_char_hpl"] = pred_train[:,1]

train_df["nb_tfidf_char_mws"] = pred_train[:,2]

test_df["nb_tfidf_char_eap"] = pred_full_test[:,0]

test_df["nb_tfidf_char_hpl"] = pred_full_test[:,1]

test_df["nb_tfidf_char_mws"] = pred_full_test[:,2]
cols_to_drop = ['id', 'text']

train_X = train_df.drop(cols_to_drop+['author'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)
ss = StandardScaler()

ss.fit(pd.concat([train_X,test_X]))

features = train_X.columns

train_X[features] = ss.transform(train_X[features])

test_X[features] = ss.transform(test_X[features])
def GPClusterX(data):

    v = pd.DataFrame()

    v["0"] = np.tanh((((data["nb_cvec_eap"] - ((data["svd_word_0"] + data["svd_word_12"]) + data["nb_cvec_char_hpl"])) * 2.0) * 2.0))

    v["1"] = np.tanh(((data["nb_tfidf_char_eap"] + ((((data["svd_word_1"] * 2.0) * 2.0) - data["nb_cvec_char_hpl"]) * 2.0)) * 2.0))

    v["2"] = np.tanh((((data["svd_word_1"] - (data["svd_word_0"] - (data["svd_word_17"] - data["nb_cvec_char_hpl"]))) * 2.0) * 2.0))

    v["3"] = np.tanh(((data["nb_tfidf_char_eap"] - ((data["svd_word_0"] + (data["nb_cvec_char_hpl"] + data["svd_word_12"])) * 2.0)) * 2.0))

    v["4"] = np.tanh((((((data["nb_cvec_eap"] - data["svd_word_0"]) * 2.0) - data["nb_cvec_char_hpl"]) * 2.0) * 2.0))

    v["5"] = np.tanh(((((data["nb_cvec_char_eap"] - data["num_words"]) - (data["svd_word_12"] + data["nb_cvec_char_hpl"])) * 2.0) * 2.0))

    v["6"] = np.tanh((((((data["svd_word_1"] * 2.0) - data["nb_cvec_char_hpl"]) * 2.0) + data["svd_word_13"]) * 2.0))

    v["7"] = np.tanh((((data["nb_cvec_char_eap"] + ((data["svd_word_17"] - data["nb_cvec_char_hpl"]) - data["num_stopwords"])) * 2.0) * 2.0))

    v["8"] = np.tanh(((((data["nb_cvec_eap"] - data["svd_word_0"]) - data["svd_word_0"]) - data["nb_cvec_char_hpl"]) * 2.0))

    v["9"] = np.tanh((((data["nb_tfidf_char_eap"] - (data["svd_word_12"] + data["num_unique_words"])) - data["nb_cvec_hpl"]) * 2.0))

    v["10"] = np.tanh((((((data["nb_cvec_eap"] - data["num_words"]) - data["svd_word_0"]) * 2.0) - data["nb_tfidf_char_hpl"]) * 2.0))

    v["11"] = np.tanh((((((-(data["nb_tfidf_char_hpl"])) - data["svd_word_0"]) - data["num_words"]) * 2.0) - data["num_words_title"]))

    v["12"] = np.tanh(((((data["nb_cvec_eap"] - data["nb_cvec_char_hpl"]) - data["num_punctuations"]) - data["svd_word_12"]) * 2.0))

    v["13"] = np.tanh(((((((data["svd_word_1"] * 2.0) * 2.0) - data["num_chars"]) + data["nb_tfidf_char_eap"]) * 2.0) * 2.0))

    v["14"] = np.tanh((((data["svd_word_17"] - data["nb_cvec_char_hpl"]) + data["nb_cvec_char_eap"]) * 2.0))

    v["15"] = np.tanh((((-(((data["svd_word_0"] + data["svd_word_12"]) + data["svd_word_8"]))) - data["nb_cvec_char_hpl"]) * 2.0))

    v["16"] = np.tanh((((((data["nb_tfidf_char_eap"] - data["num_unique_words"]) - data["svd_word_12"]) - data["svd_word_2"]) * 2.0) * 2.0))

    v["17"] = np.tanh((((data["svd_word_17"] - data["svd_word_12"]) - (data["nb_tfidf_char_hpl"] + data["svd_word_0"])) - data["num_unique_words"]))

    v["18"] = np.tanh(((((data["svd_word_10"] + data["nb_cvec_char_eap"]) - data["svd_word_8"]) - data["num_punctuations"]) - data["svd_word_2"]))

    v["19"] = np.tanh((data["svd_word_1"] - (data["svd_word_0"] + (data["num_words_upper"] + (data["svd_word_12"] + data["nb_tfidf_char_hpl"])))))

    v["20"] = np.tanh(((((data["nb_cvec_char_eap"] - data["svd_word_15"]) - data["svd_word_8"]) - data["svd_word_18"]) * 2.0))

    v["21"] = np.tanh(((((((data["svd_word_1"] * 2.0) * 2.0) - data["num_words"]) - data["num_punctuations"]) * 2.0) * 2.0))

    v["22"] = np.tanh((data["nb_cvec_char_eap"] + (data["nb_cvec_eap"] * ((data["num_unique_words"] * -3.0) - data["nb_cvec_char_eap"]))))

    v["23"] = np.tanh(((data["svd_word_17"] - (data["svd_word_0"] + data["nb_tfidf_char_hpl"])) - (data["svd_word_3"] + data["svd_word_8"])))

    v["24"] = np.tanh((((-((data["svd_word_18"] + (data["nb_cvec_char_eap"] * data["num_unique_words"])))) * 2.0) - data["svd_word_8"]))

    v["25"] = np.tanh(((((-1.0 - (data["num_words_title"] + data["num_words"])) - data["svd_word_0"]) * 2.0) * 2.0))

    v["26"] = np.tanh((((data["svd_word_17"] - ((data["svd_word_3"] + data["svd_word_12"]) + data["num_words_upper"])) * 2.0) * 2.0))

    v["27"] = np.tanh((-((((data["svd_word_7"] * 2.0) * data["svd_word_7"]) + (data["svd_word_4"] + data["svd_word_8"])))))

    v["28"] = np.tanh((((data["svd_word_0"] * data["num_unique_words"]) - (data["nb_cvec_mws"] * data["nb_tfidf_char_mws"])) * 2.0))

    v["29"] = np.tanh(((0.78320163488388062) - (data["svd_word_5"] + (data["svd_word_8"] + (data["svd_word_7"] * data["svd_word_7"])))))

    v["30"] = np.tanh((-1.0 + (data["svd_word_0"] * (((data["nb_cvec_char_eap"] * 2.0) * 2.0) * data["num_unique_words"]))))

    v["31"] = np.tanh(((data["nb_cvec_char_eap"] * ((data["svd_word_17"] - data["num_words_upper"]) - data["num_stopwords"])) - data["svd_word_8"]))

    v["32"] = np.tanh((((data["num_stopwords"] * data["svd_word_0"]) - (data["svd_word_7"] * data["svd_word_7"])) - data["svd_word_5"]))

    v["33"] = np.tanh((((data["svd_word_11"] * (data["svd_word_6"] - data["svd_word_11"])) - data["svd_word_9"]) - data["svd_word_6"]))

    v["34"] = np.tanh(((data["svd_word_8"] * (data["nb_tfidf_char_eap"] - data["svd_word_8"])) + (data["num_words_upper"] * data["num_unique_words"])))

    v["35"] = np.tanh(((((data["svd_word_6"] * data["svd_word_19"]) + data["svd_word_7"])/2.0) - (data["svd_word_6"] * data["svd_word_6"])))

    v["36"] = np.tanh(((((data["svd_word_1"] * data["nb_cvec_char_eap"]) * 2.0) - ((data["svd_word_6"] + data["svd_word_9"])/2.0)) * 2.0))

    v["37"] = np.tanh((((data["svd_word_12"] * (data["svd_word_13"] - data["svd_word_12"])) - data["nb_cvec_char_mws"]) - data["svd_word_12"]))

    v["38"] = np.tanh(((data["nb_tfidf_char_eap"] * data["nb_cvec_hpl"]) - (data["svd_word_9"] + (data["svd_word_8"] - data["nb_cvec_hpl"]))))

    v["39"] = np.tanh((data["svd_word_14"] - (data["svd_word_4"] + ((data["svd_word_11"] * data["svd_word_11"]) + data["svd_word_6"]))))

    v["40"] = np.tanh(((data["svd_word_12"] * data["svd_word_4"]) - (data["svd_word_9"] + (data["num_words_upper"] * data["nb_cvec_char_eap"]))))

    v["41"] = np.tanh(((data["svd_word_1"] * data["svd_word_8"]) - ((data["svd_word_2"] + (data["svd_word_6"] + data["svd_word_19"]))/2.0)))

    v["42"] = np.tanh((data["svd_word_7"] - ((data["svd_word_6"] + (data["svd_word_7"] * 2.0)) * (data["svd_word_7"] * 2.0))))

    v["43"] = np.tanh(((((data["num_chars"] + data["svd_word_11"])/2.0) * (data["num_chars"] - data["svd_word_11"])) - data["svd_word_6"]))

    v["44"] = np.tanh((((data["svd_word_11"] - (data["num_unique_words"] + data["num_stopwords"])) - data["svd_word_9"]) - data["nb_cvec_eap"]))

    v["45"] = np.tanh(((data["nb_cvec_char_hpl"] * (data["svd_word_11"] - (data["svd_word_13"] - data["nb_cvec_mws"]))) - data["svd_word_9"]))

    v["46"] = np.tanh((((data["num_unique_words"] * data["nb_cvec_char_eap"]) * (data["num_unique_words"] - data["mean_word_len"])) - data["nb_cvec_eap"]))

    v["47"] = np.tanh(((((data["num_words"] - (data["svd_word_0"] * data["mean_word_len"])) + data["svd_word_11"])/2.0) - data["svd_word_9"]))

    v["48"] = np.tanh((((data["svd_word_0"] - (data["svd_word_0"] * data["nb_cvec_char_hpl"])) - data["svd_word_17"]) * data["num_words"]))

    v["49"] = np.tanh(((data["svd_word_18"] * data["svd_word_11"]) - (data["svd_word_15"] + (data["nb_cvec_char_mws"] * data["nb_cvec_mws"]))))



    return v.sum(axis=1)





def GPClusterY(data):

    v = pd.DataFrame()

    v["0"] = np.tanh((((data["svd_word_12"] + (data["nb_cvec_char_mws"] * 2.0)) * 2.0) + (data["svd_word_7"] + data["svd_word_11"])))

    v["1"] = np.tanh((((data["svd_word_8"] + ((data["nb_cvec_char_mws"] + data["svd_word_2"]) + data["nb_cvec_mws"])) * 2.0) * 2.0))

    v["2"] = np.tanh((((data["nb_cvec_char_mws"] - data["nb_cvec_char_hpl"]) + (data["svd_word_8"] + data["nb_cvec_mws"])) * 2.0))

    v["3"] = np.tanh((((data["svd_word_2"] + ((data["nb_cvec_char_mws"] + data["nb_cvec_mws"]) - data["nb_cvec_char_hpl"])) * 2.0) * 2.0))

    v["4"] = np.tanh(((((data["svd_word_2"] - (data["nb_cvec_char_hpl"] + data["nb_cvec_eap"])) * 2.0) + data["svd_word_12"]) * 2.0))

    v["5"] = np.tanh(((((data["svd_word_3"] + (data["svd_word_8"] + data["nb_cvec_char_mws"])) + data["nb_cvec_char_mws"]) * 2.0) * 2.0))

    v["6"] = np.tanh(((((data["svd_word_2"] + (data["nb_cvec_mws"] - data["nb_cvec_char_hpl"])) * 2.0) + data["svd_word_12"]) * 2.0))

    v["7"] = np.tanh(((((data["num_words_upper"] + data["nb_cvec_char_mws"]) + (data["svd_word_12"] - data["nb_cvec_char_hpl"])) * 2.0) * 2.0))

    v["8"] = np.tanh((((data["num_words_upper"] + ((-(data["nb_cvec_eap"])) - (data["nb_cvec_char_hpl"] * 2.0))) * 2.0) * 2.0))

    v["9"] = np.tanh((((data["nb_cvec_char_mws"] + (data["svd_word_7"] + (data["svd_word_12"] + data["svd_word_2"]))) * 2.0) * 2.0))

    v["10"] = np.tanh(((((data["num_words_upper"] - (data["nb_cvec_eap"] + (data["nb_cvec_char_hpl"] * 2.0))) * 2.0) * 2.0) * 2.0))

    v["11"] = np.tanh((((data["nb_cvec_char_mws"] + (data["svd_word_7"] + data["svd_word_8"])) * 2.0) * 2.0))

    v["12"] = np.tanh((((data["nb_tfidf_char_mws"] - data["nb_cvec_char_hpl"]) * 2.0) - (data["mean_word_len"] + data["nb_cvec_char_hpl"])))

    v["13"] = np.tanh(((((((data["nb_tfidf_char_mws"] - data["nb_cvec_char_hpl"]) - data["svd_word_4"]) * 2.0) * 2.0) * 2.0) * 2.0))

    v["14"] = np.tanh(((((data["svd_word_12"] - (data["svd_word_4"] + data["nb_cvec_char_hpl"])) * 2.0) - data["nb_cvec_eap"]) * 2.0))

    v["15"] = np.tanh(((1.0 + ((data["svd_word_0"] + (data["nb_cvec_char_mws"] + data["svd_word_12"])) * 2.0)) * 2.0))

    v["16"] = np.tanh(((data["svd_word_2"] * (data["nb_cvec_char_hpl"] * (9.78357696533203125))) - ((data["nb_cvec_char_hpl"] * 2.0) * 2.0)))

    v["17"] = np.tanh((((((data["svd_word_3"] * 2.0) - data["nb_cvec_hpl"]) - data["nb_cvec_eap"]) * 2.0) - data["nb_tfidf_char_hpl"]))

    v["18"] = np.tanh((((data["nb_cvec_char_mws"] + ((data["svd_word_8"] + data["svd_word_15"]) + data["svd_word_7"])) * 2.0) * 2.0))

    v["19"] = np.tanh(((((data["svd_word_4"] * (data["nb_cvec_char_hpl"] * 2.0)) - data["nb_cvec_char_hpl"]) * 2.0) * 2.0))

    v["20"] = np.tanh((((data["svd_word_0"] - ((data["svd_word_0"] * 2.0) * data["nb_cvec_char_mws"])) - data["svd_word_17"]) * 2.0))

    v["21"] = np.tanh(((((data["nb_cvec_char_hpl"] * 2.0) * (data["svd_word_0"] * 2.0)) + 1.0) * 2.0))

    v["22"] = np.tanh((((data["nb_tfidf_char_mws"] + (data["svd_word_15"] - data["nb_cvec_char_hpl"])) * 2.0) + data["svd_word_18"]))

    v["23"] = np.tanh((((((data["nb_cvec_char_mws"] + data["svd_word_6"]) - data["svd_word_17"]) + data["svd_word_11"]) * 2.0) * 2.0))

    v["24"] = np.tanh((((data["nb_cvec_char_hpl"] * 2.0) * ((data["nb_cvec_char_hpl"] * data["svd_word_0"]) * 2.0)) - -3.0))

    v["25"] = np.tanh(((((data["num_chars"] * data["nb_cvec_char_hpl"]) - (data["svd_word_0"] * data["nb_cvec_char_mws"])) * 2.0) * 2.0))

    v["26"] = np.tanh(((data["svd_word_14"] + data["num_words_upper"]) + (data["nb_tfidf_char_eap"] * (data["svd_word_17"] - data["nb_cvec_char_mws"]))))

    v["27"] = np.tanh(((data["svd_word_0"] * data["nb_cvec_char_hpl"]) + (data["nb_tfidf_char_mws"] + (data["svd_word_12"] * data["nb_cvec_char_hpl"]))))

    v["28"] = np.tanh(((((data["svd_word_12"] - data["svd_word_7"]) + data["svd_word_4"]) + data["num_unique_words"]) * data["nb_cvec_char_hpl"]))

    v["29"] = np.tanh((data["num_words"] + (data["nb_tfidf_char_mws"] * (data["svd_word_7"] - (data["svd_word_0"] * 2.0)))))

    v["30"] = np.tanh((((data["nb_tfidf_char_mws"] * data["nb_tfidf_char_mws"]) + (data["num_stopwords"] + data["nb_cvec_char_eap"])) + data["svd_word_6"]))

    v["31"] = np.tanh(((data["nb_tfidf_char_mws"] + data["svd_word_18"]) + ((data["svd_word_8"] + data["svd_word_0"]) - data["svd_word_17"])))

    v["32"] = np.tanh((((data["nb_cvec_hpl"] * 2.0) * 2.0) * ((data["svd_word_5"] + data["num_chars"]) - data["svd_word_7"])))

    v["33"] = np.tanh((((data["mean_word_len"] + data["svd_word_15"]) + (data["nb_cvec_char_hpl"] + data["nb_tfidf_char_mws"])) * data["nb_cvec_char_mws"]))

    v["34"] = np.tanh((((((data["num_unique_words"] * data["nb_cvec_char_mws"]) * data["num_words_upper"]) * 2.0) * 2.0) + data["nb_cvec_hpl"]))

    v["35"] = np.tanh(((data["nb_cvec_char_hpl"] * (-((data["svd_word_7"] - (data["num_punctuations"] - data["svd_word_16"]))))) * 2.0))

    v["36"] = np.tanh(((data["nb_cvec_char_eap"] * data["svd_word_17"]) + ((data["svd_word_8"] - data["svd_word_4"]) - data["nb_cvec_char_hpl"])))

    v["37"] = np.tanh(((np.tanh(data["nb_cvec_hpl"]) - (data["nb_cvec_char_mws"] * (data["svd_word_0"] + data["num_unique_words"]))) * 2.0))

    v["38"] = np.tanh((((-(data["svd_word_16"])) * data["nb_cvec_eap"]) + ((-(data["mean_word_len"])) * data["nb_cvec_eap"])))

    v["39"] = np.tanh((((data["nb_cvec_char_hpl"] * data["svd_word_5"]) + ((data["nb_cvec_char_hpl"] * data["svd_word_8"]) - data["nb_cvec_char_hpl"]))/2.0))

    v["40"] = np.tanh(((((data["num_words"] * data["svd_word_0"]) * data["nb_cvec_mws"]) + ((data["svd_word_7"] + data["nb_cvec_hpl"])/2.0))/2.0))

    v["41"] = np.tanh((((((data["svd_word_17"] / 2.0) * data["nb_cvec_char_eap"]) + (data["svd_word_1"] * data["nb_cvec_char_eap"]))/2.0) * 2.0))

    v["42"] = np.tanh((((data["mean_word_len"] * data["num_words_upper"]) - data["nb_cvec_mws"]) - (data["svd_word_18"] * data["nb_cvec_char_mws"])))

    v["43"] = np.tanh((data["nb_cvec_char_hpl"] * ((data["svd_word_5"] + (data["svd_word_8"] - data["nb_cvec_char_hpl"])) - data["svd_word_7"])))

    v["44"] = np.tanh((((data["svd_word_16"] + np.tanh((data["nb_tfidf_char_mws"] / 2.0)))/2.0) * data["nb_tfidf_char_mws"]))

    v["45"] = np.tanh((data["nb_tfidf_char_mws"] * (data["nb_cvec_hpl"] * data["svd_word_0"])))

    v["46"] = np.tanh((data["mean_word_len"] * (data["nb_cvec_char_eap"] * data["svd_word_0"])))

    v["47"] = np.tanh((((data["mean_word_len"] * data["svd_word_0"]) * (-(data["nb_tfidf_char_mws"]))) - data["nb_tfidf_char_mws"]))

    v["48"] = np.tanh(((data["svd_word_14"] + (data["nb_cvec_char_hpl"] - data["num_words_upper"])) * (data["svd_word_12"] + data["svd_word_12"])))

    v["49"] = np.tanh(((data["nb_cvec_char_mws"] * data["num_unique_words"]) * (((data["nb_cvec_mws"] * data["num_words"]) + data["num_words"])/2.0)))

    return v.sum(axis=1)
colors = ['red', 'green','blue']

plt.figure(figsize=(15,15))

plt.scatter(GPClusterX(train_X),GPClusterY(train_X),s=10, color=[colors[o]for o in train_y])