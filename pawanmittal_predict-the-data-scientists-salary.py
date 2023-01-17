# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 8

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/Final_Train_Dataset.csv')

test=pd.read_csv('../input/Final_Test_Dataset.csv')

train.head()
train.company_name_encoded.nunique()

train.shape
train.salary.value_counts()
target=pd.get_dummies(train.salary)

#target
test.head()
# train.experience.apply(lambda x:x.split("-"))

train.job_type.value_counts()
df=train.copy()



df.job_description.fillna('None',inplace=True)

df.key_skills.fillna('None',inplace=True)

def exp1(val):

    v=val.split(" ")[0].split("-")

    v1,v2=int(v[0]),int(v[1])

    return (v1+v2)/2

df['exp']=df.experience.apply(exp1)

df['key_skill_len']=df.key_skills.apply(lambda x:len(x.split(",")))





df['exp_low']=df.experience.apply(lambda x:int(x.split("-")[0]))

df['exp_high']=df.experience.apply(lambda x:int(x.split("-")[1].split(" ")[0]))





df['jdescLen']=df.job_description.apply(len)

df['jdsgnLen']=df.job_desig.apply(len)

df['locationLen']=df.location.apply(lambda x:len(x.split(",")))



df.drop(['job_type','Unnamed: 0','experience'],axis=1,inplace=True)

target=df.salary



# target=pd.get_dummies(train.salary)
#target
df.head()
df.isnull().sum()

# df.columns
from sklearn.feature_extraction.text import TfidfVectorizer

cv_jb = TfidfVectorizer(ngram_range=(1,3),stop_words="english", analyzer='word')

jb =cv_jb.fit_transform(df['job_description'])



#print(cv_jb.get_feature_names())

#print(jb.shape)



cv_jd = TfidfVectorizer(ngram_range=(1,2),stop_words="english", analyzer='word')

jd =cv_jd.fit_transform(df['job_desig'])



cv_key = TfidfVectorizer(ngram_range=(1,3),stop_words="english", analyzer='word')

ks =cv_key.fit_transform(df['key_skills'])



cv_jbchar = TfidfVectorizer(ngram_range=(1,9),stop_words="english", analyzer='char')

jbchar =cv_jbchar.fit_transform(df['job_description'])

#print(cv_jbchar.get_feature_names())

print(jbchar.shape)



cv_jdchar = TfidfVectorizer(ngram_range=(1,8),stop_words="english", analyzer='char')

jdchar =cv_jdchar.fit_transform(df['job_desig'])



cv_keychar = TfidfVectorizer(ngram_range=(1,9),stop_words="english", analyzer='char')

kschar =cv_keychar.fit_transform(df['key_skills'])



# loc=np.array(df['location'].values)

# loc.append(np.array(test['location'].values),axis=0)

loc = np.append(df['location'].values, test['location'].values, axis=0)

loc
loc.shape
from sklearn.preprocessing import LabelEncoder

l=LabelEncoder()

l.fit(list(set(loc)))

df['location']=l.transform(df['location'])





tot_comp = np.append(df['company_name_encoded'].values, test['company_name_encoded'].values, axis=0)

l_comp=LabelEncoder()

l_comp.fit(list(set(tot_comp)))

df['company_name_encoded']=l_comp.transform(df['company_name_encoded'])



#df['company_name_encoded']

#set(loc)
df.columns

df.head()

#df[['location','company_name_encoded','exp','key_skill_len','jdescLen', 'jdsgnLen', 'locationLen']]
( jb, jd, ks,jbchar, jdchar, kschar)
from scipy.sparse import csr_matrix

from scipy import sparse

final_features = sparse.hstack((df[['location','company_name_encoded','exp','key_skill_len','jdescLen', 'jdsgnLen', 'locationLen']], jb, jd, ks,jbchar, jdchar, kschar)).tocsr()

# final_features = sparse.hstack((df[['location','company_name_encoded','exp','key_skill_len']], jb, jd, ks)).tocsr()
#final_features.todense()
dftest=test.copy()



dftest.job_description.fillna('None',inplace=True)

dftest.key_skills.fillna('None',inplace=True)

def exp1(val):

    v=val.split(" ")[0].split("-")

    v1,v2=int(v[0]),int(v[1])

    return (v1+v2)/2

dftest['exp']=dftest.experience.apply(exp1)

dftest['key_skill_len']=dftest.key_skills.apply(lambda x:len(x.split(",")))



dftest['exp_low']=dftest.experience.apply(lambda x:int(x.split("-")[0]))

dftest['exp_high']=dftest.experience.apply(lambda x:int(x.split("-")[1].split(" ")[0]))



jb_t =cv_jb.transform(dftest['job_description'])

jd_t =cv_jd.transform(dftest['job_desig'])

ks_t =cv_key.transform(dftest['key_skills'])



jb_tchar =cv_jbchar.transform(dftest['job_description'])

jd_tchar =cv_jdchar.transform(dftest['job_desig'])

ks_tchar =cv_keychar.transform(dftest['key_skills'])





dftest['location']=l.transform(dftest['location'])

dftest['company_name_encoded']=l_comp.transform(dftest['company_name_encoded'])



dftest['jdescLen']=dftest.job_description.apply(len)

dftest['jdsgnLen']=dftest.job_desig.apply(len)

dftest['locationLen']=dftest.location.astype(str).apply(lambda x:len(x.split(",")))

dftest.drop(['job_type','experience'],axis=1,inplace=True)





final_features_t = sparse.hstack((dftest[['location','company_name_encoded','exp','key_skill_len','jdescLen', 'jdsgnLen', 'locationLen']], jb_t, jd_t, ks_t,jb_tchar, jd_tchar, ks_tchar)).tocsr()

# final_features_t = sparse.hstack((dftest[['location','company_name_encoded','exp','key_skill_len']], jb_t, jd_t, ks_t)).tocsr()
final_features_t
#target
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train,X_val,y_train,y_val = train_test_split(final_features,target,test_size=0.25,random_state = 1994)
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

def train_model(classifier, train_X, train_y, test_X, test_y):

    # fit the training dataset on the classifier

    classifier.fit(train_X, train_y)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(test_X)

    

    return metrics.accuracy_score(predictions, test_y)
import xgboost

accuracy = train_model(xgboost.XGBClassifier(verbose=True), X_train.tocsc(), y_train, X_val.tocsc(),y_val)

print( "Xgb, Count Vectors: ", accuracy)




# lr=LogisticRegression(class_weight='balanced',C=2,random_state=1994,n_jobs=-1)

# lr.fit(X_train,y_train)

# lrpred=lr.predict(X_val)

# print(accuracy_score(y_val,lrpred))
# xgb=xgboost.XGBClassifier()



# # eval_set = [(X_train.tocsc(), y_train),(X_val.tocsc(), y_val)]

# # xgb.fit(X_train.tocsc(), y_train, eval_metric="mlogloss", eval_set=eval_set, verbose=True,early_stopping_rounds=50)

# # # make predictions for test data



# xgb.fit(X_train.tocsc(),y_train)

# xgbpred=xgb.predict(X_val.tocsc())

# print(accuracy_score(y_val,xgbpred))
# xgb=xgboost.XGBClassifier()



# # eval_set = [(X_train.tocsc(), y_train),(X_val.tocsc(), y_val)]

# # xgb.fit(X_train.tocsc(), y_train, eval_metric="mlogloss", eval_set=eval_set, verbose=True,early_stopping_rounds=50)

# # # make predictions for test data



# xgb.fit(X_train.tocsc(),y_train)

# xgbpred=xgb.predict(X_val.tocsc())

# print(accuracy_score(y_val,xgbpred))
# from sklearn.ensemble import RandomForestClassifier

# rf=RandomForestClassifier()

# rf.fit(X_train,y_train)

# rfpred=rf.predict(X_val)

# print(accuracy_score(y_val,rfpred))
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier



# mb=MultinomialNB(alpha=0.5)

# mb.fit(final_features,target)

# mbpred=mb.predict(final_features_t)



# lr=LogisticRegression(class_weight='balanced',C=2,random_state=1994,n_jobs=-1)

# lr.fit(final_features,target)

# lrpred=lr.predict(final_features_t)



xgb=xgboost.XGBClassifier(verbose=True,n_jobs=-1)

xgb.fit(final_features.tocsc(),target)

xgbpred=xgb.predict(final_features_t.tocsc())

# rf=RandomForestClassifier(n_estimators=200)

# rf.fit(final_features,target)

# rfpred=rf.predict(final_features_t)



# cb=CatBoostClassifier()

# cb.fit(final_features,target)

# cbpred=cb.predict(final_features_t)



xgbpred
s=pd.DataFrame({'company_name_encoded':test.company_name_encoded,'salary':xgbpred})

s.head()

s.to_excel('xgbbase9.xlsx',index=False)
# s['salary']=lrpred

# s.to_excel('lr.xlsx',index=False)



# s['salary']=rfpred

# s.to_excel('rf.xlsx',index=False)