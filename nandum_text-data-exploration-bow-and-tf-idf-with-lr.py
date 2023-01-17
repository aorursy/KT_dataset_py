# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.tsv' , delimiter= '\t')
test = pd.read_csv('../input/test.tsv', delimiter= '\t')
train.head()
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, vstack

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,10))
sentiment_counts = train.Sentiment.value_counts()
sns.barplot(sentiment_counts.index , sentiment_counts.values)
plt.show()
train_y = train['Sentiment'].values
tfv = TfidfVectorizer(ngram_range=(1,3), use_idf= True , analyzer= 'word')
tfv.fit(train['Phrase'].values.tolist() + test['Phrase'].values.tolist())
train_tfidf = tfv.transform(train['Phrase'].values)
test_tfidf = tfv.transform(test['Phrase'].values)
cv_object = CountVectorizer(ngram_range=(1,3), analyzer= 'word')
cv_object.fit(train['Phrase'].values.tolist() + test['Phrase'].values.tolist())
train_cv = cv_object.transform(train['Phrase'].values)
test_cv = cv_object.transform(test['Phrase'].values)
tf_char = TfidfVectorizer(ngram_range=(1,6), analyzer= 'char' , max_features=20000)
tf_char.fit(train['Phrase'].values.tolist() + test['Phrase'].values.tolist())
train_tfidf_char = tf_char.transform(train['Phrase'].values)
test_tfidf_char = tf_char.transform(test['Phrase'].values)
bow_char = CountVectorizer(ngram_range=(1,6), analyzer= 'char' , max_features=20000)
bow_char.fit(train['Phrase'].values.tolist() + test['Phrase'].values.tolist())
train_cv_char = bow_char.transform(train['Phrase'].values)
test_cv_char = bow_char.transform(test['Phrase'].values)
def SVD(train , test, keyword, n_components = 25):
    svd_obj = TruncatedSVD(n_components=n_components, algorithm='arpack')
    svd_obj.fit(vstack([train, test]))
    columns = ['svd_'+ keyword + '_' + str(i) for i in range(n_components)]
    train_df = pd.DataFrame(data = svd_obj.transform(train) , columns = columns)
    test_df = pd.DataFrame(data = svd_obj.transform(test) , columns = columns)
    return train_df, test_df
print(train_tfidf.shape)
print(test_tfidf.shape)
final_train , final_test = SVD(train_tfidf, test_tfidf , 'word_tfidf')
tf_char_train,  tf_char_test = SVD(train_tfidf_char, test_tfidf_char, 'char_tfidf')
final_train = pd.concat([final_train, tf_char_train], axis=1)
final_test = pd.concat([final_test, tf_char_test], axis= 1)
del tf_char_train, tf_char_test, train_tfidf, test_tfidf
def model_multinomial(train_X, train_y, test_X, test_y, test_X2):
    model = MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model
def calculate_cv_score(model, train_x , train_y, test_x, num_splits = 3, loss = log_loss, is_dataframe = False):
    ''' model needs to return validation prediction , test prediction and model itself after fitting'''
    cv_scores = []
    pred_train = np.zeros([train_x.shape[0] , 5])
    pred_test_final = 0
    kfold = KFold(n_splits= num_splits, random_state= 2018 , shuffle= True)
    for dev_index , val_index in kfold.split(train_x):
        if is_dataframe:
            dev_X, val_X = train_x.loc[dev_index], train_x.loc[val_index]
        else:            
            dev_X, val_X = train_x[dev_index], train_x[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val , pred_test , fit_model = model(dev_X, dev_y, val_X , val_y , test_x)
        pred_test_final = pred_test_final + pred_test
        loss_score = loss(val_y , pred_val)
        pred_train[val_index,:] = pred_val
        cv_scores.append(loss_score)
    avg_cv_score = np.mean(cv_scores)
    pred_test_final = pred_test_final / num_splits
    print(pred_train.shape)
    return avg_cv_score, pred_test_final, pred_train 
        
        
cvscore_bow_words, bow_predictions_words, pred_train = calculate_cv_score(model_multinomial, train_cv, train_y, test_cv)
cvscore_bow_char, bow_predictions_char, pred_train_char = calculate_cv_score(model_multinomial, train_cv_char, train_y, test_cv_char)
final_train["mnb_bow_word_0"] = pred_train[:,0]
final_train["mnb_bow_word_1"] = pred_train[:,1]
final_train["mnb_bow_word_2"] = pred_train[:,2]
final_train["mnb_bow_word_3"] = pred_train[:,3]
final_train["mnb_bow_word_4"] = pred_train[:,4]
final_test["mnb_bow_word_0"] = bow_predictions_words[:,0]
final_test["mnb_bow_word_1"] = bow_predictions_words[:,1]
final_test["mnb_bow_word_2"] = bow_predictions_words[:,2]
final_test["mnb_bow_word_3"] = bow_predictions_words[:,3]
final_test["mnb_bow_word_4"] = bow_predictions_words[:,4]
final_train["mnb_bow_char_0"] = pred_train_char[:,0]
final_train["mnb_bow_char_1"] = pred_train_char[:,1]
final_train["mnb_bow_char_2"] = pred_train_char[:,2]
final_train["mnb_bow_char_3"] = pred_train_char[:,3]
final_train["mnb_bow_char_4"] = pred_train_char[:,4]

final_test["mnb_bow_char_0"] = bow_predictions_char[:,0]
final_test["mnb_bow_char_1"] = bow_predictions_char[:,1]
final_test["mnb_bow_char_2"] = bow_predictions_char[:,2]
final_test["mnb_bow_char_3"] = bow_predictions_char[:,3]
final_test["mnb_bow_char_4"] = bow_predictions_char[:,4]
print(final_test.shape)
print(final_train.shape)
print(train_y.shape)
def run_lr(train_X, train_y, val_X, val_y, test_X):
    lr = LogisticRegression(C=0.01)
    lr.fit(train_X, train_y)
    pred_val = lr.predict_proba(val_X)
    pred_test = lr.predict_proba(test_X)
    return pred_val, pred_test, lr
final_cv, final_pred, _ = calculate_cv_score(run_lr, final_train, train_y, final_test, is_dataframe=True)
sub1 = np.argmax(final_pred, axis=1)
np.bincount(sub1)
test['Sentiment'] = sub1
test[['PhraseId' , 'Sentiment']].to_csv('sub_1.csv' , index=False)

