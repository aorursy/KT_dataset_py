import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import nltk, os, re, string

from nltk.corpus import stopwords

import string

print(os.listdir('../input/'))
train = pd.read_excel('../input/machinehacknewscategory/Data_Train.xlsx')

train.head()
sns.countplot(train.SECTION)
## A. TOTAL NUMBER OF WORDS USED 

train['nb_words'] = train.STORY.apply(lambda x: len(x.split()))



## B. TOTAL NUMBER OF UNIQUE WORDS USED

train['nb_unique_words'] = train.STORY.apply(lambda x: len(set(x.split())))



## C. TOTAL NUMBER OF CHARACTERS USED

train['nb_char'] = train.STORY.apply(lambda x: len(x))
## D. TOTAL NUMBER OF PUNCTUATION USED

def punct(text):

    return(len([w for w in text.split() if w in list(string.punctuation)]))

train['nb_punct'] = train.STORY.apply(lambda x: punct(x))



## E. TOTAL NUMBER OF STOPWORDS USED

stopword = stopwords.words('english')

def stop(text):

    return(len([w for w in text.split() if w in stopword]))

train['nb_stopwords'] = train.STORY.apply(lambda x: stop(x))



## F. TOTAL NUMBER OF TITLE WORDS USED

def title(text):

    return(len([w for w in text.split() if w.istitle()]))

train['nb_title_case'] = train.STORY.apply(lambda x: title(x))



## G. AVERAGE LENGTH OF WORDS

def length(text):

    return(np.mean([len(w) for w in text.split()]))

train['avg_len_word'] = train.STORY.apply(lambda x: length(x))
## H. NUMBER OF MOST FREQUENT TERMS

token = nltk.word_tokenize(''.join(train.STORY))

frequent = nltk.FreqDist(token)

frequent.most_common(15)
## REMOVING PUNCTUATION AND STOPWORDS FROM MOST FREQUENT WORDS

for sym in string.punctuation:

    del frequent[sym]

for word in stopword:

    del frequent[word]

frequent.most_common(15)
%%time

## I. NUMBER OF WORDS CONTAIN OUT OF MOST COMMON 100 WORDS 

freq_words = list(dict(frequent.most_common()[:100]).keys())

def freq(text):

    return(len([w for w in text.split() if w in freq_words]))

train['nb_freq_words'] = train.STORY.apply(lambda x: freq(x))
%%time

## J. AVERAGE OF FREQ TERMS WITH TOTAL WORDS USED

freq_words = list(dict(frequent.most_common()[:100]).keys())

def freq(text):

    return(len([w for w in text.split() if w in freq_words])/len(text.split()))

train['avg_freq_word']= train.STORY.apply(lambda x: freq(x))
train_label = train.SECTION

train_backup = train

train = train.drop(columns=['SECTION','STORY'])

train.head(1)
from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, log_loss

import xgboost as xgb

import lightgbm as lgb
## Helper function to train and plot LGB and XGB models

param_xgb = {}

param_xgb['objective'] = 'multi:softprob'

param_xgb['num_class'] = 4

param_xgb['learning_rate'] = 0.1

param_xgb['seed'] = 666

param_xgb['eval_metric'] = 'mlogloss'



param_lgb = {}

param_lgb['objective'] = 'multiclass'

param_lgb['num_classes'] = 4

param_lgb['learning_rate'] = 0.1

param_lgb['seed'] = 666

param_lgb['metric'] = 'multi_logloss'



def lgb_xgb_helper(train, train_label ,name):

    cv = []

    pred_based_on_cv = pd.DataFrame(data = np.zeros(shape = (train.shape[0], 4)))

    kfold = KFold(n_splits=5, shuffle= True, random_state=2019)

    for t_index, v_index in kfold.split(train_label.ravel()):

        xtrain, ytrain = train.loc[t_index,:], train_label[t_index]

        xtest, ytest = train.loc[v_index,:], train_label[v_index]

        if (name == 'xgb'):

            trainset = xgb.DMatrix(xtrain, label=ytrain)

            testset = xgb.DMatrix(xtest, label=ytest)

            model = xgb.train(list(param_xgb.items()), trainset, evals=[(trainset,'train'), (testset,'test')], 

                             num_boost_round = 5000, early_stopping_rounds = 200, verbose_eval= 200)

            pred_based_on_cv.loc[v_index,:] = model.predict(testset, ntree_limit = model.best_ntree_limit)

        else :

            trainset = lgb.Dataset(xtrain, label=ytrain)

            testset = lgb.Dataset(xtest, label=ytest, reference=trainset)

            model = lgb.train(param_lgb, trainset, valid_sets= testset ,

                             num_boost_round= 5000, early_stopping_rounds = 200,  verbose_eval= 200)

            pred_based_on_cv.loc[v_index,:] = model.predict(xtest, best_iteration = model.best_iteration)

    cv.append(log_loss(ytest, pred_based_on_cv.loc[v_index,:]))

    return(np.mean(cv), pred_based_on_cv, model)



def lgb_xgb_plotting(cv, pred, label, model, name=None):

    fig, ax = plt.subplots(1,2,figsize=(18,5))

    print("CV score : %s" %cv)

    sns.heatmap(confusion_matrix(label, np.argmax(pred.values, axis=1)), annot=True, ax= ax[0])

    ax[0].set_title("Accuracy : %s" % accuracy_score(np.argmax(pred.values, axis=1), train_label))

    name.plot_importance(model, ax= ax[1])

    plt.title("Feature Importance")

    return(accuracy_score(np.argmax(pred.values, axis=1), train_label), cv)
# Let's check data distribution

plt.subplots(3,3, figsize = (18,10))

i = 1

for col in train.columns :

    plt.subplot(3,3,i)

    sns.distplot(train[col])

    i = i+1
std_scaler = StandardScaler()

train = pd.DataFrame(std_scaler.fit_transform(train), columns = train.columns)
# TRAIN LGB MODEL ON META FEATURES

cv, pred, model = lgb_xgb_helper(train, train_label, 'lgb')
# PLOTTING LGB MODEL CONFUSION MATRIX AND FEATURE IMPORTANCE

meta_acc_lgb, meta_cv_lgb = lgb_xgb_plotting(cv , pred, train_label, model, lgb)
# TRAIN XGB MODEL ON META FEATURES

cv, pred, model = lgb_xgb_helper(train, train_label, 'xgb')
# PLOTTING XGB MODEL CONFUSION MATRIX AND FEATURE IMPORTANCE

meta_acc_xgb, meta_cv_xgb = lgb_xgb_plotting(cv, pred, train_label, model, xgb)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.naive_bayes import MultinomialNB
def helper(train, train_label ,model):

    cv = []

    pred_based_on_cv = pd.DataFrame(data = np.zeros(shape = (train.shape[0], 4)))

    kfold = KFold(n_splits=5, shuffle= True, random_state= 2019)

    for t_index, v_index in kfold.split(train_label.ravel()):

        xtrain, ytrain = train[t_index,:], train_label[t_index]

        xtest, ytest = train[v_index,:], train_label[v_index]

        

        model.fit(xtrain, ytrain)

        pred_based_on_cv.loc[v_index,:] = model.predict_proba(xtest)

        cv.append(log_loss(ytest, pred_based_on_cv.loc[v_index,:]))

    return(np.mean(cv), pred_based_on_cv)



def plotting_helper(cv, pred, label, name=None):

    print("CV score : %s" %cv)

    plt.figure(figsize = (9,5))

    sns.heatmap(confusion_matrix(label, np.argmax(pred.values, axis=1)), annot=True)

    plt.title("Accuracy : %s" % accuracy_score(np.argmax(pred.values, axis=1), train_label))

    return(accuracy_score(np.argmax(pred.values, axis=1), train_label), cv)
# COUNT VECTORIZATION USING WORD AS LOWEST LEVEL

count_vec = CountVectorizer(ngram_range=(1,3), stop_words='english')

count_vec.fit(train_backup['STORY'].values.tolist())

train_count_vec = count_vec.transform(train_backup['STORY'].values.tolist())

train_count_vec.shape
cv, pred = helper(train_count_vec, train_label, MultinomialNB())

count_acc_mnb, count_cv_mnb = plotting_helper(cv, pred, train_label)
# REDUCING DIMENSION OF SPARSE MATRIX TO 20 COMPONENTS

svd = TruncatedSVD(n_components=20)

svd.fit_transform(train_count_vec)

train_count_vec_svd = svd.transform(train_count_vec)

train_count_vec_svd.shape
# # TRAIN XGB MODEL ON TEXT BASED FEATURES

cv, pred, model = lgb_xgb_helper(pd.DataFrame(train_count_vec_svd), train_label, 'xgb')
# PLOTTING XGB MODEL CONFUSION MATRIX AND FEATURE IMPORTANCE

count_acc_xgb, count_cv_xgb = lgb_xgb_plotting(cv, pred, train_label, model, xgb)
%%time

# # TRAIN LGB MODEL ON TEXT BASED FEATURES

cv, pred, model = lgb_xgb_helper(pd.DataFrame(train_count_vec_svd), train_label, 'lgb')
# PLOTTING LGB MODEL CONFUSION MATRIX AND FEATURE IMPORTANCE

count_acc_lgb, count_cv_lgb = lgb_xgb_plotting(cv, pred, train_label, model, lgb)
count_vec_char = CountVectorizer(ngram_range = (1,5), analyzer='char', stop_words='english')

count_vec_char.fit(train_backup.STORY.values.tolist())

train_count_vec_char = count_vec_char.transform(train_backup.STORY.values.tolist())
cv, pred = helper(train_count_vec_char, train_label, MultinomialNB())

count_acc_mnb_char, count_cv_mnb_char = plotting_helper(cv, pred, train_label)
## COMPARISON OF MODELS TILL NOW

performance_accuracy = pd.DataFrame({'Model': ['Meta_LGB', 'Meta_XGB', 'Count_LGB', 'Count_XGB', 'Count_MNB', 'Count_MNB_Char'], 

              'Accuracy': [meta_acc_lgb, meta_acc_xgb, count_acc_lgb, count_acc_xgb, count_acc_mnb, count_acc_mnb_char]})



performance_logloss = pd.DataFrame({'Model': ['Meta_LGB', 'Meta_XGB', 'Count_LGB', 'Count_XGB', 'Count_MNB','Count_MNB_Char'], 

              'logloss': [meta_cv_lgb, meta_cv_xgb, count_cv_lgb, count_cv_xgb, count_cv_mnb, count_cv_mnb_char]})

fig, ax = plt.subplots(1,2,figsize=(18,6))

sns.barplot(x = 'Model', y = 'Accuracy', data = performance_accuracy , ax= ax[0])

sns.barplot(x = 'Model', y = 'logloss', data = performance_logloss , ax= ax[1])
tfidf_vec = TfidfVectorizer(ngram_range = (1,3), stop_words= 'english')

tfidf_vec.fit(train_backup.STORY.values.tolist())

train_tfidf_vec = tfidf_vec.transform(train_backup['STORY'].values.tolist())
cv, pred = helper(train_tfidf_vec, train_label, MultinomialNB())

tfidf_acc_mnb, tfidf_cv_mnb = plotting_helper(cv, pred, train_label)
train_tfidf_vec_svd = svd.fit_transform(train_tfidf_vec)

# TRAIN XGB MODEL ON TEXT BASED FEATURES

cv, pred, model = lgb_xgb_helper(pd.DataFrame(train_tfidf_vec_svd), train_label, 'xgb')
# PLOTTING XGB MODEL CONFUSION MATRIX AND FEATURE IMPORTANCE

tfidf_acc_xgb, tfidf_cv_xgb = lgb_xgb_plotting(cv, pred, train_label, model, xgb)
%%time

# # TRAIN LGB MODEL ON TEXT BASED FEATURES

cv, pred, model = lgb_xgb_helper(pd.DataFrame(train_tfidf_vec_svd), train_label, 'lgb')
# PLOTTING LGB MODEL CONFUSION MATRIX AND FEATURE IMPORTANCE

tfidf_acc_lgb, tfidf_cv_lgb = lgb_xgb_plotting(cv, pred, train_label, model, lgb)
tfidf_vec_char = TfidfVectorizer(ngram_range = (1,5), stop_words= 'english', analyzer='char')

tfidf_vec_char.fit(train_backup.STORY.values.tolist())

train_tfidf_vec_char = tfidf_vec_char.transform(train_backup['STORY'].values.tolist())
cv, pred = helper(train_tfidf_vec_char, train_label, MultinomialNB())

tfidf_acc_mnb_char, tfidf_cv_mnb_char = plotting_helper(cv, pred, train_label)
performance_accuracy = pd.concat([performance_accuracy, pd.DataFrame({'Model': ['Tfidf_MNB', 'Tfidf_XGB', 'Tfidf_LGB','Tfidf_MNB_Char'], 

                                          'Accuracy': [tfidf_acc_mnb, tfidf_acc_xgb, tfidf_acc_lgb,tfidf_acc_mnb_char]})], axis=0)



performance_logloss = pd.concat([performance_logloss, pd.DataFrame({'Model': ['Tfidf_MNB', 'Tfidf_XGB', 'Tfidf_LGB','Tfidf_MNB_Char'], 

                                          'logloss': [tfidf_cv_mnb, tfidf_cv_xgb, tfidf_cv_lgb, tfidf_acc_mnb_char]})], axis=0)
fig,ax= plt.subplots(2,1, figsize = (16,8))

sns.barplot('Model', 'Accuracy', data= performance_accuracy.sort_values(by = 'Accuracy', ascending=False), ax= ax[1], palette= 'Set3')

sns.barplot('Model', 'logloss', data= performance_logloss.sort_values(by = 'logloss'), ax= ax[0], palette= 'Set2')

plt.xticks(rotation= 90)
train = pd.concat([pd.DataFrame(train_backup['STORY']), pd.DataFrame(train_backup['SECTION'])], axis = 1)

train.head()
%%time

# STEPS TAKEN FROM SRK NOTEBOOK

# LOWER CASE ALL CHARACTERS 

train.STORY = train.STORY.apply(lambda x: x.lower())



## LEMMATIZATION

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()



def lem(text):

    pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}

    return(' '.join([lemmatizer.lemmatize(w,pos_dict.get(t, wn.NOUN)) for w,t in nltk.pos_tag(text.split())]))



train.STORY = train.STORY.apply(lambda x: lem(x))



# REMOVING PUNCTUATION

def cleaner(text):

    return(text.translate(str.maketrans('','', string.punctuation)))

train.STORY = train.STORY.apply(lambda x: cleaner(x))



# REMOVING STOPWORDS

st_words = stopwords.words()

def stopword(text):

    return(' '.join([w for w in text.split() if w not in st_words ]))

train.STORY = train.STORY.apply(lambda x: stopword(x))
train.head()
from keras.layers import Input, LSTM, Bidirectional, SpatialDropout1D, Dropout, Flatten, Dense, Embedding, BatchNormalization

from keras.models import Model

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
train_label = to_categorical(train.SECTION, 4)

train_label.shape
max_words = 40000

max_len = 256



tok = Tokenizer(lower=True, char_level=False)

tok.fit_on_texts(train.STORY)

sequence = tok.texts_to_sequences(train.STORY)

sequence = pad_sequences(padding='post', sequences=sequence, maxlen= max_len)

sequence.shape
def modeling():

    inp = Input(shape=(max_len,))

    x = Embedding(max_words, 300 ,input_length = max_len)(inp)

    x = Bidirectional(LSTM(256, return_sequences=True))(x)

    x = BatchNormalization()(x)

    x = SpatialDropout1D(0.5)(x)

    x = Flatten()(x)

    x = Dense(128, activation= 'relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(4, activation='softmax')(x)

    model = Model(inputs = inp, outputs = x)

    return(model)
model = modeling()

model.compile(optimizer = 'adam', metrics = ['accuracy'], loss= 'categorical_crossentropy')

model.summary()
lr = ReduceLROnPlateau(monitor='valid_set', factor=0.002, min_lr=0.00001)

model.fit(sequence, train_label, validation_split=0.30, callbacks=[lr], batch_size=64, epochs=10)
performance_WO = pd.DataFrame({'epoch': model.history.epoch, 'loss': model.history.history['loss'], 

              'val_loss': model.history.history['val_loss'], 'acc': model.history.history['accuracy'],

             'val_acc': model.history.history['val_accuracy']})



fig, ax = plt.subplots(2,1, figsize=(12,6), sharex = True)  

sns.lineplot(x= 'epoch', y= 'loss', data = performance_WO, ax= ax[0])

sns.lineplot(x= 'epoch', y= 'val_loss', data = performance_WO, ax = ax[0])

ax[0].legend(['TRAINING LOSS', 'VALIDATION LOSS'])

ax[0].set_ylabel('Loss')



  

sns.lineplot(x= 'epoch', y= 'acc', data = performance_WO, ax= ax[1])

sns.lineplot(x= 'epoch', y= 'val_acc', data = performance_WO, ax = ax[1])

ax[1].legend(['TRAINING ACCURACY', 'VALIDATION ACCURACY'])

ax[1].set_ylabel('Accuracy')