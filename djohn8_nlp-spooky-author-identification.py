import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!unzip '/kaggle/input/spooky-author-identification/train.zip'

!unzip '/kaggle/input/spooky-author-identification/test.zip'
# check if the files have been unzipped

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd



import nltk

from nltk.corpus import stopwords



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn import ensemble, model_selection, metrics, naive_bayes



import time

import gc
# read train and test

train_df = pd.read_csv('/kaggle/working/train.csv')

test_df = pd.read_csv('/kaggle/working/test.csv')

print('train dataset shape', train_df.shape)

print('test dataset shape', test_df.shape)
train_df.head()
plt.figure(figsize=(10,6))

sns.countplot(train_df['author'])

plt.xlabel('Authors', fontsize=12)

plt.ylabel('Number of text occurences', fontsize=12)

plt.show()
# Examine some of the text from each author

grouped_df = train_df.groupby('author')['text']

for author, group in grouped_df:

    print('Author name :',author,'\n', 'Text\n',group.tolist()[:5])

    print('\n')
import string  # for use in string.punctuation here

eng_stopwords = set(stopwords.words('english'))



# number of words in text

train_df['num_words'] = train_df['text'].apply(lambda x: len(str(x).split()))

test_df['num_words'] = test_df['text'].apply(lambda x: len(str(x).split()))



# unique words count

train_df['num_uniq_words'] = train_df['text'].apply(lambda x: len(set(str(x).split())))

test_df['num_uniq_words'] = test_df['text'].apply(lambda x: len(set(str(x).split())))



# punctuation count

train_df['num_punctuations'] = train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

test_df['num_punctuations'] = test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))



# character count

train_df['num_chars'] = train_df['text'].apply(lambda x: len(str(x)))

test_df['num_chars'] = test_df['text'].apply(lambda x: len(str(x)))



# stop words count

train_df['num_stopwords'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test_df['num_stopwords'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



# title case words count

train_df['num_words_title'] = train_df['text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test_df['num_words_title'] = test_df['text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



# upper case words

train_df['num_words_upper'] = train_df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test_df['num_words_upper'] = test_df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



# average length of words

train_df['num_words_upper'] = train_df['text'].apply(lambda x: np.mean(len([w for w in str(x).split()])))

test_df['num_words_upper'] = test_df['text'].apply(lambda x: np.mean(len([w for w in str(x).split()])))
# after adding new features

train_df.head(10)
# truncating for better visuals

train_df.loc[train_df['num_words'] > 80, 'num_words']=80

plt.figure(figsize=(12,5))

sns.boxplot(x = 'author', y = 'num_words', data=train_df)

plt.title('Number of words in text for each author', fontsize=12)

plt.show()
# using violin plot

plt.figure(figsize=(12,5))

sns.violinplot(x = 'author', y = 'num_words', data=train_df)

plt.title('Number of words in text for each author', fontsize=12)

plt.show()
# using violin plot

plt.figure(figsize=(12,5))

sns.violinplot(x = 'author', y = 'num_uniq_words', data=train_df)

plt.title('Number of unqiue words in text for each author', fontsize=12)

plt.ylim(0,100)

plt.show()
# using violin plot

train_df.loc[train_df['num_punctuations'] > 10, 'num_punctuations'] = 10

plt.figure(figsize=(12,6))

sns.violinplot(x = 'author', y = 'num_punctuations', data=train_df)

plt.title('Number of punctuations in text for each author', fontsize=12)

#plt.ylim(0,10)

plt.show()
# encode label for author

author_mapping_dict ={'EAP': 0, 'HPL': 1, 'MWS': 2}

train_y = train_df['author'].map(author_mapping_dict)



# store the ids

train_id = train_df['id'].values

print('train_id shape', train_id.shape)

test_id = test_df['id'].values

print('test_id shape', test_id.shape)



cols_to_drop = ['id', 'text']

train_X = train_df.drop(cols_to_drop + ['author'], axis=1)

test_X = test_df.drop(cols_to_drop , axis=1)
import xgboost as xgb



def runXGB(train_X, train_y, val_X, val_y = None, test_X2=None, seed_val=0, child=1, colsample=0.3):

    

    # parameters

    params = {}

    params['max_depth'] = 3

    params['objective'] = 'multi:softprob'

    params['eval_metric'] = 'mlogloss'

    params['num_class'] = 3

    params['eta'] = 0.1

    params['silent'] = 1   # verbosity

    params['min_child_weight'] = child

    params['colsample_bytree'] = colsample

    params['subsample'] = 0.8

    params['seed'] = seed_val

    

    num_rounds = 2000

    

    # list of paramaters values to be passed to train XGB

    params_list = list(params.items())

    

    # create the dense matrix for train

    xgtrain = xgb.DMatrix(train_X, label= train_y)

    

    if not val_y is None:

        xgval = xgb.DMatrix(val_X, label = val_y)

        watchlist = [(xgtrain, 'train'), (xgval, 'val')]

        

        # stop training the model if no improvement for 50 iterations

        model = xgb.train(params_list, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)

    

    else:

        xgval = xgb.DMatrix(val_X)

        model.train(params_list, xgtrain, num_rounds)

        

    pred_val_y = model.predict(xgval, ntree_limit = model.best_ntree_limit)

        

        

    # On the actual test set

    if test_X2 is not None:

        xgtest2 = xgb.DMatrix(test_X2)

        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)

        

    

    return (pred_val_y, pred_test_y2, model)    
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2020)

cv_scores =[]



for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    # test_X is the actual test set

    pred_val_y, pred_test_y, model =  runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)

    

    # cv scores

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    

    break # running only 1 iteration for now



print('cv scores: for logloss on simple XGBoost model', cv_scores)    
fig, ax = plt.subplots(figsize=(12,10))

xgb.plot_importance(model, max_num_features=50, height=0.7, ax=ax)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



# instantiate

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

# fit_transform on the combined train and test,as there can be pattern in test set not seen in train

full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
train_tfidf
full_tfidf
# since there are 3 classes, we will use a Multinomial Naive Bayes(NB) model



def runMNB(train_X, train_y, val_X, val_y, test_X2):

    model = naive_bayes.MultinomialNB()

    model.fit(train_X, train_y)

    # use model.predict_proba to detect the probabilites for each of the classes

    # if using model.predict, the model just return a prediction

    pred_proba_val_y = model.predict_proba(val_X)

    pred_proba_test_y = model.predict_proba(test_X2)

    return pred_proba_val_y, pred_proba_test_y, model
%%time

cv_scores = []

predict_val = np.zeros([train_df.shape[0], 3]) # store prediction on validation set to use in confusion matrix



kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2020)



# create splits on train_X and use the tf-idf values from train_tfidf and test_tfidf



for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_proba_val_y, pred_proba_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    

    

    cv_scores.append(metrics.log_loss(val_y, pred_proba_val_y))

    predict_val[val_index, :] = pred_proba_val_y

    

print('cv scores for mlogloss using NB on tf-idf at word level: ', cv_scores)

print('Mean cv scores for mlogloss using NB on tf-idf at word level: ', np.mean(cv_scores))
from mlxtend.evaluate import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



cm = confusion_matrix(val_y, np.argmax(pred_proba_val_y, axis=1), binary=False)

fig, ax = plot_confusion_matrix(cm)

plt.show()
from sklearn.decomposition import TruncatedSVD



n_comp= 20



svd_obj = TruncatedSVD(n_components=n_comp, algorithm ='arpack')

svd_obj.fit(full_tfidf)



train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))



# renaming columns

train_svd.columns = ['svd_tfidf_word_'+ str(i) for i in range(n_comp)]

test_svd.columns = ['svd_tfidf_word_'+ str(i) for i in range(n_comp)]



# concat the information original train_df

train_df = pd.concat([train_df, train_svd], axis=1)

test_df = pd.concat([test_df, test_svd], axis=1)



del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

gc.collect()
# glimse after the merge with SVD features

display(train_df.sample(2))

display(test_df.sample(2))
%%time

count_vec = CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1,3))

count_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())



train_count_vec = count_vec.transform(train_df['text'].values.tolist())

test_count_vec = count_vec.transform(test_df['text'].values.tolist())
train_count_vec.shape
test_count_vec.shape
cv_scores =[]

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])



kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2020)



for dev_index, val_index in kf.split(train_df):

    dev_X, val_X = train_count_vec[dev_index], train_count_vec[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_proba_val_y, pred_proba_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_count_vec)

    # append the scores in test from each fold

    pred_full_test += pred_proba_test_y

    # store the data corresponding to the validation indexes

    pred_train[val_index,:] = pred_proba_val_y    

    

    cv_scores.append(metrics.log_loss(val_y, pred_proba_val_y))

    

print('Mean logloss score: using NB on Count Vectorize at word level ', np.mean(cv_scores))
# columns reprsent the probability of belonging to the 3 classes

pred_train.shape
# check if the number of rows is the same

assert pred_train.shape[0] == train_df.shape[0]



train_df['nb_cvec_eap'] = pred_train[:, 0]

train_df['nb_cvec_hpl'] = pred_train[:, 1]

train_df['nb_cvec_mws'] = pred_train[:, 2]



assert pred_full_test.shape[0] == test_df.shape[0]



test_df['nb_cvec_eap'] = pred_full_test[:, 0]

test_df['nb_cvec_hpl'] = pred_full_test[:, 1]

test_df['nb_cvec_mws'] = pred_full_test[:, 2]
# using mlxtend

cm = confusion_matrix(val_y, np.argmax(pred_proba_val_y, axis=1), binary=False)

#plt.figure(figsize=(10,10))

plot_confusion_matrix(cm)

plt.title('Confusion matrix for NB on CountVectorizer')

plt.show()
%%time



count_vec_char = CountVectorizer(ngram_range=(1,7), analyzer='char')

count_vec_char.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())



train_cvec_char = count_vec_char.transform(train_df['text'])

test_cvec_char = count_vec_char.transform(test_df['text'])



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])



kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2020)



for dev_index, val_index in kf.split(train_df):

    dev_X, val_X = train_cvec_char[dev_index], train_cvec_char[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_proba_val_y, pred_proba_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_cvec_char)

    # append the scores in test from each fold

    pred_full_test += pred_proba_test_y

    # store the data corresponding to the validation indexes

    pred_train[val_index,:] = pred_proba_val_y    

    

    cv_scores.append(metrics.log_loss(val_y, pred_proba_val_y))



print('Mean logloss score using Naive Bayes on character Count vectorizer', np.mean(cv_scores))
# check if the number of rows is the same

assert pred_train.shape[0] == train_df.shape[0]



train_df['nb_cvec_char_eap'] = pred_train[:, 0]

train_df['nb_cvec_char_hpl'] = pred_train[:, 1]

train_df['nb_cvec_char_mws'] = pred_train[:, 2]



assert pred_full_test.shape[0] == test_df.shape[0]



test_df['nb_cvec_char_eap'] = pred_full_test[:, 0]

test_df['nb_cvec_char_hpl'] = pred_full_test[:, 1]

test_df['nb_cvec_char_mws'] = pred_full_test[:, 2]
%%time

# tf-idf tranformation at character level



tfidf_char_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')

# use the fit_transform method if need to apply transformation on the data at a later stage. Since we are using SVD at a later

# stage it is better to apply the fit_transform method

full_tfidf = tfidf_char_vec.fit_transform(train_df['text'].values.tolist() +  test_df['text'].values.tolist())



train_tfidf_char_vec = tfidf_char_vec.transform(train_df['text'].values.tolist())

test_tfidf_char_vec = tfidf_char_vec.transform(test_df['text'].values.tolist())



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])



# use naive bayes

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2020)



for dev_index, val_index in kf.split(train_df):

    dev_X, val_X = train_tfidf_char_vec[dev_index], train_tfidf_char_vec[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_proba_val_y, pred_proba_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf_char_vec)

    # append the scores in test from each fold

    pred_full_test += pred_proba_test_y

    # store the data corresponding to the validation indexes

    pred_train[val_index,:] = pred_proba_val_y    

    

    cv_scores.append(metrics.log_loss(val_y, pred_proba_val_y))





print('Mean logloss score: using NB on tf-idf tranformation at character level', np.mean(cv_scores))
# check if the number of rows is the same

assert pred_train.shape[0] == train_df.shape[0]



train_df['nb_tfidf_char_eap'] = pred_train[:, 0]

train_df['nb_tfidf_char_hpl'] = pred_train[:, 1]

train_df['nb_tfidf_char_mws'] = pred_train[:, 2]



assert pred_full_test.shape[0] == test_df.shape[0]



print('test_df shape', test_df.shape)



test_df['nb_tfidf_char_eap'] = pred_full_test[:, 0]

test_df['nb_tfidf_char_hpl'] = pred_full_test[:, 1]

test_df['nb_tfidf_char_mws'] = pred_full_test[:, 2]
%%time

# extract 20 features from the high dimensional tf-idf vector space

n_comp = 20

svd_obj = TruncatedSVD(n_components = n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)



train_svd_char = svd_obj.transform(train_tfidf_char_vec)

test_svd_char  = svd_obj.transform(test_tfidf_char_vec)



# create pandas dataframe, so that we can add them as features and renaming the columns for more clarity

train_svd_char = pd.DataFrame(train_svd_char, columns = ['svd_tfidf_char_'+ str(i) for i in range(n_comp)])

test_svd_char = pd.DataFrame(test_svd_char, columns = ['svd_tfidf_char_'+ str(i) for i in range(n_comp)])



# concat them with the original train and test to add these new features there

train_df = pd.concat([train_df, train_svd_char], axis=1)

test_df  = pd.concat([test_df, test_svd_char], axis=1)



del svd_obj, train_svd_char, test_svd_char, full_tfidf, train_tfidf_char_vec, test_tfidf_char_vec

gc.collect()
train_df.columns
cols_to_drop = ['id', 'text']

train_X = train_df.drop(cols_to_drop + ['author'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)



cv_scores = []

pred_full_test = 0

pred_val = np.zeros([train_df.shape[0], 3])



# run K-Fold cross validation

kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 2020)



for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

        

    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7)

    

    pred_full_test = pred_full_test + pred_test_y

    pred_val[val_index, :] = pred_val_y

    

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    #break # remove to run across all 5 folds



# create a dataframe for the predictions

out_df = pd.DataFrame(pred_full_test, columns = ['EAP', 'HPL', 'MWS'])

# insert the id column at column 0

out_df.insert(0, 'id', test_id)

out_df.to_csv('submission.csv', index=False)



print('cv scores for log_loss using tf-idf at char level using xgboost is:', cv_scores)
fig, ax = plt.subplots(figsize=(14,14))

xgb.plot_importance(model, max_num_features=50, height=0.6, ax=ax)

plt.show()
from mlxtend.evaluate import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



cm = confusion_matrix(val_y, np.argmax(pred_val_y, axis=1))

plot_confusion_matrix(cm)

plt.title('Confusion matrix authors: EAP: 0, HPL: 1, MWS: 2')

plt.show()