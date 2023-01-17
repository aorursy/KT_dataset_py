!pip install vaderSentiment
import pandas as pd

import numpy as np

from scipy.sparse import hstack,csr_matrix

from itertools import combinations,chain

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk import pos_tag

from nltk.tokenize import RegexpTokenizer

toker = RegexpTokenizer(r'\w+')

from nltk.stem import snowball

st = snowball.SnowballStemmer('english')

wnl =  nltk.wordnet.WordNetLemmatizer()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

angram_range = (1,1)

#token_pattern=r'\w+'

tf_vectorizer = CountVectorizer( ngram_range=angram_range ,strip_accents='unicode',stop_words=stopWords)

tfidf_vectorizer = TfidfVectorizer(ngram_range=angram_range,strip_accents='unicode',stop_words=stopWords)

from scipy.stats import ttest_ind,mstats

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,auc,precision_recall_curve,f1_score

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier,LogisticRegression

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

from collections import Counter

import matplotlib.pyplot as plt 

import seaborn as sns
df_Train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_Test = pd.read_csv('../input/nlp-getting-started/test.csv')
df_Train.head()
df_Test.head()
print('number of tweets')

print(len(df_Train))

print('number of disaster tweets')

print(df_Train['target'].sum())
#unqiue counts

print('unqiue count id')

print(len(pd.unique(df_Train['id'])))

print('unqiue count keyword')

print(len(pd.unique(df_Train['keyword'])))

print('unqiue count location')

print(len(pd.unique(df_Train['location'])))

print('unqiue count target')

print(len(pd.unique(df_Train['target'])))
#nan counts

print('nan count id')

print(sum(pd.isna(df_Train['id'])))

print('nan count keyword')

print(sum(pd.isna(df_Train['keyword'])))

print('nan count location')

print(sum(pd.isna(df_Train['location'])))

print('nan count target')

print(sum(pd.isna(df_Train['target'])))
print(pd.unique(df_Train['keyword']))
#replace '%20' with spaces

df_Train.loc[:,'keyword'] = df_Train['keyword'].apply(lambda x: x.replace('%20',' ') if type(x)==str else x)

df_Test.loc[:,'keyword'] = df_Test['keyword'].apply(lambda x: x.replace('%20',' ') if type(x)==str else x)
#create hashtag indictor

df_Train['Hashtag_indicator']=df_Train['text'].apply(lambda x: 0  if str(x).find('#')==-1 else 1 )

df_Test['Hashtag_indicator']=df_Test['text'].apply(lambda x: 0  if str(x).find('#')==-1 else 1 )
print('number of tweets with hashtag')

print(df_Train['Hashtag_indicator'].sum())
#sentiment- neg sentiment could help model prediction

df_Train['sentiment'] = df_Train['text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

df_Test['sentiment'] = df_Test['text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
#tfidf vectorize the acutal text

tfidf_fit = tfidf_vectorizer.fit(df_Train['text'])



X2 = tfidf_fit.transform(df_Train['text'])

X4 = csr_matrix(df_Train['sentiment'].to_list()).transpose()

X6 = csr_matrix(df_Train['Hashtag_indicator'].to_list()).transpose()



X2_test = tfidf_fit.transform(df_Test['text'])

X4_test = csr_matrix(df_Test['sentiment'].to_list()).transpose()

X6_test =csr_matrix(df_Test['Hashtag_indicator'].to_list()).transpose()



#combine all features into one matrix

X=hstack([X2,X4,X6])

X_test=hstack([X2_test,X4_test,X6_test])

X=csr_matrix(X)

X_test =csr_matrix(X_test)

y = df_Train['target'].to_list()

y=np.array(y)
print('top 20 text words - avg tfidf score')

pd.DataFrame(X2.mean(axis=0),columns=tfidf_fit.get_feature_names()).T.sort_values(0,ascending=False)[:20].plot.bar()
df_Train_target1 =df_Train[df_Train['target']==1]

df_Train_target0 =df_Train[df_Train['target']==0]

L_target1_Pos = list(df_Train_target1.index)

L_target0_Pos = list(df_Train_target0.index)

X_target1 = X[L_target1_Pos,:]

X_target0 = X[L_target0_Pos,:]
print('avg sentiment for disaster:')

print(df_Train_target1['sentiment'].mean())

print('avg sentiment for non-disaster:')

print(df_Train_target0['sentiment'].mean())
print('avg Hashtag_indicator for disaster:')

print(df_Train_target1['Hashtag_indicator'].mean())

print('avg Hashtag_indicator for non-disaster:')

print(df_Train_target0['Hashtag_indicator'].mean())


#filter cols with t-test, might take some time

LtTest = [col for col in range(X.shape[1]) if ttest_ind(X_target0[:,col].toarray(), X_target1[:,col].toarray(), axis=0, equal_var=False, nan_policy='omit')[1]<=0.3]

X = X[:,LtTest]

X_test = X_test[:,LtTest]
#train models

classifier_tree = XGBClassifier(n_estimators=1200,n_jobs=-1,random_state=123)

classifier_svm = SVC(probability=True,gamma='scale',kernel='rbf')

classifier_linear = BaggingClassifier(LogisticRegression(solver='lbfgs'),n_estimators=10,random_state=123)



Lclassifiers = [classifier_linear,classifier_tree ]#

cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2)

##find best theshold from f1 score

#Threshold_pred = 0.5

for train, test in cv.split(X, y):

    X_train_cv  = X[train]

    X_test_cv = X[test]

    y_train_cv = y[train]

    y_test_cv = y[test]

    #X_trainT_cv = csr_matrix.transpose(X_train_cv)#X_train_cv.T

    

    print('over sample with smote')

    intKn = 5

    smoter = SMOTE(k_neighbors=intKn)

    X_train_cv, y_train_cv = smoter.fit_resample(X_train_cv, y_train_cv)

    

    X_train_cv=csr_matrix(X_train_cv)

    #train models in list

    print('fit model')

    Lfit_var_cv = [model.fit(X_train_cv, y_train_cv) for model in Lclassifiers]

    print('predict model')

    Lcv_probas_ = [predcitive_fit.predict_proba(X_test_cv)[:,1] for predcitive_fit in Lfit_var_cv]

    

    cv_probas_ = np.mean(Lcv_probas_,axis = 0)

    

    print('compute the ROC curve')

    fpr, tpr, thresholds = roc_curve(y_test_cv, cv_probas_)#[:, 1])##

    roc_auc = auc(fpr, tpr)

    

    print('compute the precision recall curve')

    precision, recall, thresholds_PR = precision_recall_curve(y_test_cv, cv_probas_)

    PR_auc =  auc(recall, precision)

    

    #make sure 50% is in tested list

    thresholds_PR=list(thresholds_PR)

    thresholds_PR.append(0.5)

    print('find % threshold that gives the highest f1 score')

    df_f1_PR = pd.DataFrame([[thresholds_,f1_score(y_test_cv,np.where(cv_probas_>thresholds_,1,0))]  for thresholds_ in thresholds_PR])

    df_f1_PR_max = df_f1_PR[df_f1_PR[1] == max(df_f1_PR[1])]

    #Threshold_pred = df_f1_PR_max[0].values[0]

    Threshold_pred=0.5

    cv_pred_var =  np.where(cv_probas_>Threshold_pred,1,0)

    Lcm = list(confusion_matrix(y_test_cv,cv_pred_var).ravel())

    print('__________________')

    df_f1_PR.columns = ['Thresholds','f1_score']

    df_f1_PR=df_f1_PR.sort_values('f1_score',ascending=False)

    print('top 5 thresholds and f1 score')

    print(df_f1_PR.head())

    print('bottom 5 thresholds and f1 score')

    print(df_f1_PR.tail())

    print('__________________')

    print('Threshold for predictions:')

    print(Threshold_pred)

    print('__________________')

    print('vailidation metrics:')

    print('__________________')

    print('ROC AUC')

    print(roc_auc)

    print('PR AUC')

    print(PR_auc)

    print('CM [tn,fp,fn,tp]:')

    print(Lcm)

    print('Accuracy:')

    print((Lcm[0]+Lcm[3])/sum(Lcm))

    print('Precision:')

    print(Lcm[3]/(Lcm[3]+Lcm[1]))

    print('Recall sensitivity:')

    print(Lcm[3]/(Lcm[3]+Lcm[2]))

    print('Specificity')

    print(Lcm[0]/(Lcm[0]+Lcm[1]))

    print('F1 Score ')

    print((2*Lcm[3])/((2*Lcm[3])+Lcm[1]+Lcm[2]))

    plt.close()

    mid_line = np.divide(list(range(0,101,1)),100)

    mid_line2=list(mid_line)

    mid_line2.reverse()

    roc_ax = sns.scatterplot(fpr, tpr)

    roc_ax.set(xlabel='fpr', ylabel='tpr', title='ROC AUC:'+str(roc_auc))

    plt.plot(mid_line, mid_line, color='r')

    plt.show()

    plt.close()

    PR_ax = sns.scatterplot(precision, recall)

    PR_ax.set(xlabel='precision', ylabel='recall', title='PR AUC:'+str(PR_auc))

    plt.plot(mid_line, mid_line2, color='r')

    plt.show()

    plt.close()

    df_his = pd.DataFrame([cv_probas_,y_test_cv]).T

    x_1 = df_his[df_his[1]==1.0][0].values

    x_0 = df_his[df_his[1]==0.0][0].values

    plt.clf()

    plt.hist(x_1, density=True, bins=30,color='r',label='target1',alpha = 0.5)#normed

    plt.hist(x_0, density=True, bins=30,color='b',label='target0',alpha = 0.5)#normed

    plt.axvline(x=Threshold_pred)

    plt.xlabel('Probability of prediction')

    plt.legend()
smoter = SMOTE(k_neighbors=intKn)

X_oversampled, y_oversampled = smoter.fit_resample(X, y)

Lfit_var = [model.fit(X_oversampled, y_oversampled) for model in Lclassifiers]

Lprobas = [predcitive_fit.predict_proba(X_test)[:,1] for predcitive_fit in Lfit_var]

probas_ = np.mean(Lprobas,axis = 0)

pred_var =  np.where(probas_>Threshold_pred,1,0)

df_Test['target'] = pred_var

df_Test[['id','target']].to_csv("submission.csv",index=False, header=True)

print('finshed')