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
df_train = pd.read_csv('../input/train.csv')
df_train = df_train.loc[:10000,:]
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test = df_test.loc[:10000,:]

df_train.columns
label_name = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
train_label = df_train[label_name]
df_all = pd.concat([df_train[['id','comment_text']],df_test],axis=0)
import re
from nltk.corpus import stopwords
import string
eng_words = stopwords.words('english')
def word_tokenize(x):
    regex = re.compile('['+re.escape(string.punctuation)+'0-9\\n\\t]')
    text = regex.sub(' ',x)
    words = [word for word in text.split(' ') if len(word)>=1]
    words = [word.lower() for word in words if word not in eng_words]
    return words
    
test_text = 'hello\t \nfuc 998adf df'
print(word_tokenize(test_text))
from sklearn.feature_extraction.text import TfidfVectorizer
comment_texts = df_all['comment_text'].apply(word_tokenize)
comment_texts = [" ".join(text) for text in comment_texts]

df_all['comment_processed'] = comment_texts
#tfidf
tfidf = TfidfVectorizer()
tfidf_vector = tfidf.fit_transform(df_all['comment_processed'])
#do svd to tfidf_vector
from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components= 100)
pca.fit(tfidf_vector)
pca_transformed = pca.transform(tfidf_vector)
# 暂时只用这一个feature
training = pca_transformed[:df_train.shape[0],:]
testing = pca_transformed[df_train.shape[0]:,:]
# 对八个task，每个task都单独学习一个learner，用什么模型呢?
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import time
def modelfit(alg, dtrain, ytrain , useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    print('in modelfitting...')
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=ytrain)
        # metrics 需要改
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round= 500, nfold=cv_folds,
                          metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print('num_rounds %f'%cvresult.shape[0])
    alg.fit(dtrain, ytrain,eval_metric='logloss')
    dtrain_predictions = alg.predict(dtrain)
    print ("\n logloss score on the train data:")
    print ("logloss : %.4g" % metrics.log_loss(ytrain, dtrain_predictions))



def cv_score(model,train_X,train_y):
    # 5-fold crossvalidation error
    kf = KFold(n_splits = 5)
    logloss = []
    params = model.get_xgb_params()
    print('final model parameter :')
    print(params)
    for train_ind,test_ind in kf.split(train_X):
        train_valid_x,train_valid_y = train_X[train_ind],train_y[train_ind]
        test_valid_x,test_valid_y = train_X[test_ind],train_y[test_ind]
        dtrain = xgb.DMatrix(train_valid_x,label = train_valid_y)
        dtest = xgb.DMatrix(test_valid_x)
        pred_model = xgb.train(params,dtrain,num_boost_round=int(params['n_estimators']))

        pred_test = pred_model.predict(dtest)
        logloss.append(metrics.log_loss(test_valid_y,pred_test))
    print('final logloss on cv:')
    print(np.mean(logloss))
def cross_validation(dtrain,ytrain):
    #每次调整完一个参数，重新确定新的num_rounds
    #dtrain's type is array 
    xgb_model = XGBClassifier(
                learning_rate= 0.5,
                max_depth = 20,
                n_estimators = 100,
                min_child_weight = 1,
                gamma = 0,
                objective='binary:logistic',
                nthread=4,
                )
    modelfit(xgb_model,dtrain,ytrain)
    print('tunning learning rate...')
    params = {'learning_rate':[0.01,0.015,0.025,0.05,0.1]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = params, scoring = 'neg_log_loss',n_jobs = 1,iid=False,cv=3)
    gsearch.fit(dtrain,ytrain)
    xgb_model.set_params(learning_rate = gsearch.best_params_['learning_rate'])
    print(gsearch.best_params_)
    
    '''print('tunning max_depth...')
    params = { 'max_depth':[3,5,7,9]}
    print(xgb_model.get_params()['n_estimators'])
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = params, scoring='neg_log_loss',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(max_depth = gsearch.best_params_['max_depth'])
    print(gsearch.best_params_)
    #choose best num_round
    modelfit(xgb_model,dtrain,ytrain)
    print(xgb_model.get_params()['n_estimators'])
    
    
    print('tunning min_child_weight...')
    param_child_weight = {'min_child_weight':[1,3,5,7]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_child_weight, scoring='neg_log_loss',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(min_child_weight = gsearch.best_params_['min_child_weight'])
    print(xgb_model.get_params())
    modelfit(xgb_model,dtrain.values,ytrain)
    print(xgb_model.get_params()['n_estimators'])

    print('tunning gamma...')
    param_gamma = {'gamma':[0.05,0.1,0.3,0.5,0.7,0.9,1]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_gamma, scoring='neg_log_loss',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(gamma = gsearch.best_params_['gamma'])
    print(xgb_model.get_params())
    modelfit(xgb_model,dtrain.values,ytrain)
    print(xgb_model.get_params()['n_estimators'])
    
    #print('tunning colsample_bylevel')
    #param_colsample_bylevel = {'colsample_bylevel':[0.6,0.8,1]}
    #gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_colsample_bylevel, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    #gsearch.fit(dtrain.values,ytrain)
    #xgb_model.set_params(colsample_bylevel = gsearch.best_params_['colsample_bylevel'])
    #tunning colsample_bytree
    
    #print(xgb_model.get_params())
    #modelfit(xgb_model,dtrain.values,ytrain)
    #print('num_rounds after tunning colsample_bylevel:%f'%xgb_model.get_params()['n_estimators'])

    print('tunning colsample_bytree...')
    param_colsample_bytree = {'colsample_bytree':[0.6,0.7,0.8,1]}
    gsearch = GridSearchCV(estimator = xgb_model,param_grid = param_colsample_bytree, scoring='neg_log_loss',n_jobs=4,iid=False, cv=5)
    gsearch.fit(dtrain.values,ytrain)
    xgb_model.set_params(colsample_bytree = gsearch.best_params_['colsample_bytree'])
    print(xgb_model.get_params())
    modelfit(xgb_model,dtrain.values,ytrain)
    print('num_rounds after tunning colsample_bytree:%f'%xgb_model.get_params()['n_estimators'])'''
    # save and return model
    cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
    #pickle.dump(xgb_model,open('../models/autogridsearch_xgb_'+cur_time+'.model','wb'))
    cv_score(xgb_model,dtrain,ytrain)
    return xgb_model
# load data
def generate_results(dtrain,ytrain,dtest,xgb_model,test_ids,label_name):
    xgb_model.fit(dtrain,ytrain)
    pred_test = xgb_model.predict(dtest)
    sub = pd.DataFrame({'id':test_ids,'y':pred_test})
    return sub

test_ids = df_test['id']
for cur_label in label_name:
    fir_label = train_label[cur_label].values
    best_model = cross_validation(training,fir_label)
    sub = generate_results(training,fir_label,testing,best_model,test_ids,cur_label)
    cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
    sub.to_csv(cur_label+cur_time+'.csv',index=False)



