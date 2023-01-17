# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


import os

import gc

import time

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import networkx as nx



from sklearn import model_selection

from sklearn import linear_model



from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import roc_auc_score



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_test = pd.read_csv('../input/test.csv')

df_test.head()
# крутая штука)) самые популярные слова в тестовой треин



train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)



from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
# Количество различных штук в тексте

qmarks = np.mean(train_qs.apply(lambda x: '?' in x))

math = np.mean(train_qs.apply(lambda x: '[math]' in x))

fullstop = np.mean(train_qs.apply(lambda x: '.' in x))

capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))

capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))

numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))



print('Вопросов с вопросительным знаком: {:.2f}%'.format(qmarks * 100))

print('Вопросов про математику: {:.2f}%'.format(math * 100))

print('Вопросов с точками: {:.2f}%'.format(fullstop * 100))

print('Вопросов в которых первая буква заглавная: {:.2f}%'.format(capital_first * 100))

print('Вопросов с заглавными буквами: {:.2f}%'.format(capitals * 100))

print('Вопросов с цифирками: {:.2f}%'.format(numbers * 100))
# смотрим количество общих слов в вопросах и как от этого зависит их похожесть



from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

         # вопросы которые сгенерировал компутер

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R



plt.figure(figsize=(15, 5))

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)
df_train = df_train.dropna(how="any").reset_index(drop=True)



featureExtractionStartTime = time.time()

maxNumFeatures = 300



# bag of letter sequences (chars)

BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures, 

                                      analyzer='char', ngram_range=(1,2), 

                                      binary=True, lowercase=True)

# bag of words

#BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=10, max_features=maxNumFeatures, 

#                                      analyzer='word', ngram_range=(1,6), stop_words='english', 

#                                      binary=True, lowercase=True)



BagOfWordsExtractor.fit(pd.concat((df_train.ix[:,'question1'],df_train.ix[:,'question2'])).unique())



trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(df_train.ix[:,'question1'])

trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(df_train.ix[:,'question2'])

lables = np.array(df_train.ix[:,'is_duplicate'])



featureExtractionDurationInMinutes = (time.time()-featureExtractionStartTime)/60.0

print("feature extraction took %.2f minutes" % (featureExtractionDurationInMinutes))
# кросс валидация

crossValidationStartTime = time.time()



numCVSplits = 8

numSplitsToBreakAfter = 2



X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)

#X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int) + \

#      trainQuestion1_BOW_rep.multiply(trainQuestion2_BOW_rep)

y = lables



logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag')



logRegAccuracy = []

logRegLogLoss = []

logRegAUC = []



print('---------------------------------------------')

stratifiedCV = model_selection.StratifiedKFold(n_splits=numCVSplits, random_state=2)

for k, (trainInds, validInds) in enumerate(stratifiedCV.split(X, y)):

    foldTrainingStartTime = time.time()



    X_train_cv = X[trainInds,:]

    X_valid_cv = X[validInds,:]



    y_train_cv = y[trainInds]

    y_valid_cv = y[validInds]



    logisticRegressor.fit(X_train_cv, y_train_cv)



    y_train_hat =  logisticRegressor.predict_proba(X_train_cv)[:,1]

    y_valid_hat =  logisticRegressor.predict_proba(X_valid_cv)[:,1]



    logRegAccuracy.append(accuracy_score(y_valid_cv, y_valid_hat > 0.5))

    logRegLogLoss.append(log_loss(y_valid_cv, y_valid_hat))

    logRegAUC.append(roc_auc_score(y_valid_cv, y_valid_hat))

    

    foldTrainingDurationInMinutes = (time.time()-foldTrainingStartTime)/60.0

    print('fold %d took %.2f minutes: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (k+1,

             foldTrainingDurationInMinutes, logRegAccuracy[-1],logRegLogLoss[-1],logRegAUC[-1]))



    if (k+1) >= numSplitsToBreakAfter:

        break





crossValidationDurationInMinutes = (time.time()-crossValidationStartTime)/60.0



print('---------------------------------------------')

print('cross validation took %.2f minutes' % (crossValidationDurationInMinutes))

print('mean CV: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (np.array(logRegAccuracy).mean(),

                                                                 np.array(logRegLogLoss).mean(),

                                                                 np.array(logRegAUC).mean()))

print('---------------------------------------------')
# тренеруемся на полном наборе данных



trainingStartTime = time.time()



logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag', 

                                                    class_weight={1: 0.46, 0: 1.32})



# Стоит заметить: class_weight принимает значения {0.46, 1.32}.

# Оказывается, что распределение меток на трейне и тесте разное, то есть распределение меток на тесте просто скошено.

    



logisticRegressor.fit(X, y)



trainingDurationInMinutes = (time.time()-trainingStartTime)/60.0

print('full training took %.2f minutes' % (trainingDurationInMinutes))
# работает с полной тестовой информацией



testPredictionStartTime = time.time()





df_test.ix[df_test['question1'].isnull(),['question1','question2']] = 'random empty question'

df_test.ix[df_test['question2'].isnull(),['question1','question2']] = 'random empty question'



testQuestion1_BOW_rep = BagOfWordsExtractor.transform(df_test.ix[:,'question1'])

testQuestion2_BOW_rep = BagOfWordsExtractor.transform(df_test.ix[:,'question2'])



X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)



#  для избежания ошибки, связанной с памятью

seperators= [750000,1500000]

testPredictions1 = logisticRegressor.predict_proba(X_test[:seperators[0],:])[:,1]

testPredictions2 = logisticRegressor.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]

testPredictions3 = logisticRegressor.predict_proba(X_test[seperators[1]:,:])[:,1]

testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))





testPredictionDurationInMinutes = (time.time()-testPredictionStartTime)/60.0

print('predicting on test took %.2f minutes' % (testPredictionDurationInMinutes))
# создаем submission



submissionName = 'GalitskiyIgor[Technosphere]'



submission = pd.DataFrame()

submission['test_id'] = df_test['test_id']

submission['is_duplicate'] = testPredictions

submission.to_csv(submissionName + '.csv', index=False)