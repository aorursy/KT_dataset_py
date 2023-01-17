import numpy as np

import pandas as pd

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



# timing function

import time   

start = time.clock() #_________________ measure efficiency timing



input_folder='../input/'

train = pd.read_csv(input_folder + 'train.csv',encoding='utf8')[:10000]

test  = pd.read_csv(input_folder + 'test.csv',encoding='utf8')[:10000]



# lege opvullen

train.fillna(value='leeg',inplace=True)

test.fillna(value='leeg',inplace=True)



print("Original data: trainQ: {}, testQ: {}".format(train.shape, test.shape) )

end = time.clock()

print('open:',end-start)
def cleantxt(x):   

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    return x



train['question1']=train['question1'].map(cleantxt)

train['question2']=train['question2'].map(cleantxt)

test['question1']=test['question1'].map(cleantxt)

test['question2']=test['question2'].map(cleantxt)



train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist())

test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist())



count_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2) )

count_vectorizer.fit(train_qs.append(test_qs))  #Learn vocabulary and idf, return document freq list.

print('lengt dictionary',len(count_vectorizer.vocabulary_))



end = time.clock()

print('clean and make freq word dict:',end-start)
def splitter(dfQ,Dict):

    eq=[]

    di1=[]

    di2=[]

    for xi in range(0,len(dfQ)):

        q1words = dfQ.iloc[xi].question1.split()

        q2words = dfQ.iloc[xi].question2.split()

        equq1 = [w for w in q1words if w in q2words]

        difq1 = [w for w in q1words if w not in q2words] 

        difq2 = [w for w in q2words if w not in q1words ]

        eq.append(' '.join(equq1))

        di1.append(' '.join(difq1))

        di2.append(' '.join(difq2))

    count1_vectorizer = CountVectorizer(vocabulary=Dict, ngram_range=(1, 2),binary=True, min_df=1)

    count1_vectorizer.fit_transform(dfQ['question1'])  #Learn vocabulary and idf, return term-document matrix.

    freq1_term_matrix = count_vectorizer.transform(dfQ['question1'])

    count2_vectorizer = CountVectorizer(vocabulary=Dict, ngram_range=(1, 2),binary=True, min_df=1)

    count2_vectorizer.fit_transform(dfQ['question2'])

    freq2_term_matrix = count_vectorizer.transform(dfQ['question2']) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

    count3_vectorizer = CountVectorizer(vocabulary=Dict, ngram_range=(1, 2),binary=True, min_df=1)

    count3_vectorizer.fit_transform(di1)

    freq3_term_matrix = count_vectorizer.transform(di1) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

    count4_vectorizer = CountVectorizer(vocabulary=Dict, ngram_range=(1, 2),binary=True, min_df=1)

    count4_vectorizer.fit_transform(di2)

    freq4_term_matrix = count_vectorizer.transform(di2) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

    count5_vectorizer = CountVectorizer(vocabulary=Dict, ngram_range=(1, 2),binary=True, min_df=1)

    count5_vectorizer.fit_transform(eq)

    freq5_term_matrix = count_vectorizer.transform(eq) #Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform) This is equivalent to fit followed by transform

    tfidf1 = TfidfTransformer(norm="l2")

    tf1_idf_matrix = tfidf1.fit_transform(freq1_term_matrix)

    tfidf2 = TfidfTransformer(norm="l2")

    tf2_idf_matrix = tfidf2.fit_transform(freq2_term_matrix)

    tfidf3 = TfidfTransformer(norm="l2")

    tf3_idf_matrix = tfidf3.fit_transform(freq3_term_matrix)

    tfidf4 = TfidfTransformer(norm="l2")

    tf4_idf_matrix = tfidf4.fit_transform(freq4_term_matrix)

    tfidf5 = TfidfTransformer(norm="l2")

    tf5_idf_matrix = tfidf5.fit_transform(freq5_term_matrix)

    corr1=tf1_idf_matrix[:].dot(tf2_idf_matrix[:].T).diagonal().round(2)

    corr2=tf1_idf_matrix[:].dot(tf5_idf_matrix[:].T).diagonal().round(2)

    corr3=tf2_idf_matrix[:].dot(tf5_idf_matrix[:].T).diagonal().round(2)

    corr4=tf1_idf_matrix[:].dot(tf3_idf_matrix[:].T).diagonal().round(2)

    corr5=tf2_idf_matrix[:].dot(tf4_idf_matrix[:].T).diagonal().round(2)

    tf23e=corr2>corr3

    tf2345=(corr2+corr3)>(corr4+corr5)

    tf24=corr2>corr4

    tf35=corr3>corr5

    tf145=corr1>(corr4/2+corr5/2)

    

    return corr1,corr2,corr3,corr4,corr5,tf23e,tf2345,tf24,tf35,tf145





train['corr1'],train['corr2'],train['corr3'],train['corr4'],train['corr5'],train['tf23e'],train['tf2345'],train['tf24'],train['tf35'],train['tf145']=splitter(train,count_vectorizer.vocabulary_)

    

print(train.head(10))



test['corr1'],test['corr2'],test['corr3'],test['corr4'],test['corr5'],test['tf23e'],test['tf2345'],test['tf24'],test['tf35'],test['tf145']=splitter(test,count_vectorizer.vocabulary_)



print(test.head(10))



end = time.clock()

print('tfidf - corr:',end-start)         



    

    
import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import cross_validation, metrics

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4



def modelfit(alg, dtrain, predictors,predlabel,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[predlabel].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain[predlabel],eval_metric='auc')

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

        

    #Print model report:

    print("\nModel Report" )

    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[predlabel].values, dtrain_predictions) )

    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[predlabel], dtrain_predprob) )

                    

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')

    return alg



predictors = [x for x in train.columns if x not in ['id','question1','question2','is_duplicate','qid1','qid2']]



for di in range (5,20,3):

    # Set our parameters for xgboost

    for ci in range (1,2,2):

    # Set our parameters for xgboost

        print('maxdepth',di,'minchild',ci)

        xgb1 = XGBClassifier(

 learning_rate =0.1, n_estimators=1000, max_depth=di, min_child_weight=ci,

 gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',

 nthread=4, scale_pos_weight=1, seed=27)

        xgbmodel=modelfit(xgb1,train,predictors,'is_duplicate')



testcolumn = [x for x in test.columns if x not in ['id','question1','question2']]   

#print(test[testcolumn])
corrcolumns = [x for x in train.columns if x not in ['question1','question2']]



corr_mat=train[corrcolumns].corr()

corr_mat.head(15)


