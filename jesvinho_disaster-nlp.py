# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, metrics

import string

from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, classification

import time

import random
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
def model_evaluator(model,train_data,target_data,test_data):

    model.fit(train_data,target_data)

    score=model_selection.cross_val_score(model,train_data,target_data,cv=3,scoring='f1')

    return score
#exploring data

train.head()
#count vectorizing string-data

count_vectorizer=feature_extraction.text.CountVectorizer()

train_vectors=count_vectorizer.fit_transform(train.text[:])

test_vectors=count_vectorizer.transform(test.text[:])

target_vectors=train.target[:]
diff=train_vectors.mean(axis=0)-test_vectors.mean(axis=0)

sns.scatterplot(x=np.arange(diff.shape[1]),y=np.array(diff)[0])
#evaluating val score of count_vectrized Ridge_data

clf=linear_model.RidgeClassifier()

clf.fit(train_vectors,target_vectors)

score=model_selection.cross_val_score(clf,train_vectors,target_vectors,cv=5,scoring='f1')

a=score

print('cross validation score for count_vectorized data:',a.mean())
#predicting for test data

clf.fit(train_vectors,target_vectors)

test_preds=clf.predict(test_vectors)



#making final submission

a=pd.DataFrame({'id':test.id,'target':test_preds})

#a.to_csv('submission.csv',index=False)
# tf-idf processing the data

tfidf=feature_extraction.text.TfidfTransformer()

train_tfidf=tfidf.fit_transform(train_vectors)

test_tfidf=tfidf.transform(test_vectors)

clf=linear_model.RidgeClassifier()
train_tfidf.todense().shape
#evaluation tf_idf data with Ridge classifier (score improvement is observed)

clf=linear_model.RidgeClassifier()

clf.fit(train_tfidf,target_vectors)

score=model_selection.cross_val_score(clf,train_tfidf,target_vectors,cv=5,scoring='f1')

a=score

print('cross validation score for count_vectorized data:',a.mean())
#predicting for test data

clf.fit(train_tfidf,target_vectors)

test_preds=clf.predict(test_tfidf)



#making final submission

a=pd.DataFrame({'id':test.id,'target':test_preds})

#a.to_csv('submission.csv',index=False)
def text_cleaning(text):

    a=[char for char in text if char not in string.punctuation]

    a=''.join(a)

    a=a.split()

    #a=[c for c in a if c.lower() not in stopwords.words('english')]

    return  a
#text cleaning done

train['text'].apply(text_cleaning)
count_vectorizer=feature_extraction.text.CountVectorizer(analyzer=text_cleaning).fit(train['text'])

clf=linear_model.RidgeClassifier()
#vectorizing and tf_idf transforming again and evaluating

train_vectors=count_vectorizer.transform(train.text)

test_vectors=count_vectorizer.transform(test.text)

target_vectors=train.target[:]

tfidf=feature_extraction.text.TfidfTransformer()

train_tfidf=tfidf.fit_transform(train_vectors)

test_tfidf=tfidf.transform(test_vectors)
clf=linear_model.RidgeClassifier()

clf.fit(X_train,y_train)
accuracy=[]

f1_set=[]

for i in np.arange(-0.5,1.5,step=0.1):

    temp=pd.DataFrame(clf.decision_function(X_val))[0].apply(lambda x: 1 if x>i else 0)

    accuracy.append((temp.values==y_val.values).sum())

    

    tn, fp, fn, tp =confusion_matrix(y_val,temp).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)  

    f1_set.append(f1)

accuracy=np.array(accuracy)

f1_set=np.array(f1_set)
plt.scatter(np.arange(len(accuracy))*0.1-0.5,accuracy)
temp=pd.DataFrame(clf.decision_function(X_val))[0].apply(lambda x: 1 if x>-0.1 else 0)

print((temp.values==y_val.values).sum())
tn, fp, fn, tp =confusion_matrix(y_val,temp).ravel()

precision=tp/(tp+fp)

recall=tp/(tp+fn)

f1=(2*precision*recall)/(precision+recall)

print(f1)
#predicting for test data

clf.fit(train_tfidf,target_vectors)

test_preds=pd.DataFrame(clf.decision_function(test_tfidf))[0].apply(lambda x: 1 if x>0 else 0)



#making final submission

a=pd.DataFrame({'id':test.id,'target':test_preds})

a.to_csv('submission.csv',index=False)
test_preds.mean()
clf=tree.DecisionTreeClassifier(max_depth=180)

clf.fit(X_train,y_train)

score=model_selection.cross_val_score(clf,X_train,y_train,cv=5,scoring='f1')

a=score

print('cross validation score for count_vectorized data:',a.mean())
y_pred=clf.predict_proba(X_val)

predicted_probs=pd.DataFrame(y_pred)

predicted_probs[2]=predicted_probs[0].apply(lambda x: 0 if x>0.5 else 1)

y_prednew=predicted_probs[2].values

tn, fp, fn, tp =confusion_matrix(y_val,y_prednew).ravel()

precision=tp/(tp+fp)

recall=tp/(tp+fn)

f1=(2*precision*recall)/(precision+recall)

print("f1= ",f1)
sns.distplot(pd.DataFrame(y_pred)[1])
#predicting for test data

clf.fit(train_tfidf,target_vectors)

test_preds=clf.predict(test_tfidf)



#making final submission

a=pd.DataFrame({'id':test.id,'target':test_preds})

#a.to_csv('submission.csv',index=False)
clf.get_depth()
err=[]

for i in np.arange(5,411,step=5): 

    clf=tree.DecisionTreeClassifier(max_depth=i)

    clf.fit(train_tfidf,target_vectors)

    score=model_selection.cross_val_score(clf,train_tfidf,target_vectors,cv=5,scoring='f1')

    a=score

    print('cross validation score for count_vectorized data:',a.mean())

    err.append(a.mean())



err=np.array(err)

plt.scatter(x=range(len(err)),y=err)
plt.scatter(x=np.arange(len(err))*5+5,y=err,)
from sklearn.ensemble import RandomForestClassifier
X_train, X_val, y_train, y_val = train_test_split(train_tfidf,target_vectors, test_size=0.2, random_state=0)
clf=RandomForestClassifier(max_features="auto",n_estimators=400,oob_score=False,random_state=0)

clf.fit(X_train,y_train)
err=[]

for i in range(400):

    y_pred=clf.estimators_[i].predict(X_val)

    tn, fp, fn, tp =confusion_matrix(y_val,y_pred).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)

    err.append(f1)

    

err=np.array(err)
cum_err=[]

for i in range(len(err)):

    temp=sum(err[0:i]/(i+1))

    cum_err.append(temp)

    

cum_err=np.array(cum_err)
plt.scatter(x=np.arange(400)+1,y=cum_err)
clf=RandomForestClassifier(max_features="auto",n_estimators=40,oob_score=False,random_state=0)

clf.fit(X_train,y_train)
err=[]

for i in range(40):

    y_pred=clf.estimators_[i].predict(X_val)

    tn, fp, fn, tp =confusion_matrix(y_val,y_pred).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)

    err.append(f1)

    

err=np.array(err)



cum_err=[]

for i in range(len(err)):

    temp=sum(err[0:i]/(i+1))

    cum_err.append(temp)

    

cum_err=np.array(cum_err)



plt.scatter(x=np.arange(40)+1,y=cum_err)

plt.xlabel('number of trees')

plt.ylabel('f1-score')
depth=[]

for i in range(len(clf.estimators_)):

    depth.append(clf.estimators_[0].get_depth())

    

depth=np.array(depth)
err=[]

for i in np.arange(1,180):

    clf=RandomForestClassifier(max_depth=i,n_estimators=40,oob_score=False,random_state=0)

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_val)

    tn, fp, fn, tp =confusion_matrix(y_val,y_pred).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)

    err.append(f1)

    

err=np.array(err)
plt.scatter(x=np.arange(len(err))+1,y=err)

plt.xlabel('max_depth')

plt.ylabel('f1-score')
err=[]

for i in np.arange(1000,26000,step=1000):

    clf=RandomForestClassifier(max_depth=100,n_estimators=40,max_features=i,random_state=0)

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_val)

    tn, fp, fn, tp =confusion_matrix(y_val,y_pred).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)

    err.append(f1)

    

err=np.array(err)
plt.scatter(x=np.arange(len(err))+1,y=err)

plt.xlabel('max_features')

plt.ylabel('f1-score')
err.argmax()
ts=time.time()

clf=RandomForestClassifier(max_features="auto",n_estimators=1000,oob_score=False,random_state=0)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_val)

tn, fp, fn, tp =confusion_matrix(y_val,y_pred).ravel()

precision=tp/(tp+fp)

recall=tp/(tp+fn)

f1=(2*precision*recall)/(precision+recall)

print("f1= ",f1)

print("time taken", time.time()-ts)
f1_set=[]

for i in np.arange(0.3,0.9,step=0.01):

    y_pred=clf.predict_proba(X_val)

    predicted_probs=pd.DataFrame(y_pred)

    predicted_probs[2]=predicted_probs[0].apply(lambda x: 0 if x>i else 1)

    y_prednew=predicted_probs[2].values

    tn, fp, fn, tp =confusion_matrix(y_val,y_prednew).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)

    f1_set.append(f1)

    

f1_set=np.array(f1_set)
plt.scatter(np.arange(0.3,0.9,step=0.01),f1_set)
thres=f1_set.argmax()*0.01+0.3
clf=RandomForestClassifier(max_features="auto",n_estimators=1000,oob_score=False,random_state=0)

clf.fit(train_tfidf,target_vectors)

test_preds=clf.predict_proba(test_tfidf)

predicted_probs=pd.DataFrame(test_preds)

predicted_probs[2]=predicted_probs[0].apply(lambda x: 0 if x>thres else 1)

y_prednew=predicted_probs[2].values

#making final submission

a=pd.DataFrame({'id':test.id,'target':y_prednew})

#a.to_csv('submission.csv',index=False)
y_prednew.mean()
len(clf.estimators_)
from sklearn.ensemble import GradientBoostingClassifier

random.seed(0) 
ts=time.time()

clf = GradientBoostingClassifier(n_estimators=600, learning_rate=1.0,

                                 max_depth=1, random_state=0).fit(X_train, y_train)



print("time taken=",time.time()-ts)
f1_sets=[]

for i in np.arange(0.3,0.9,step=0.01):

    y_pred=clf.predict_proba(X_val)

    predicted_probs=pd.DataFrame(y_pred)

    predicted_probs[2]=predicted_probs[0].apply(lambda x: 0 if x>i else 1)

    y_prednew=predicted_probs[2].values

    tn, fp, fn, tp =confusion_matrix(y_val,y_prednew).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)

    f1_sets.append(f1)



f1_sets=np.array(f1_sets)
thres=f1_sets.argmax()*0.01+0.3
plt.scatter(np.arange(0.3,0.9,step=0.01),f1_sets)
thres
y_pred=clf.predict_proba(X_val)

predicted_probs=pd.DataFrame(y_pred)

predicted_probs[2]=predicted_probs[0].apply(lambda x: 0 if x>thres else 1)

y_prednew=predicted_probs[2].values

tn, fp, fn, tp =confusion_matrix(y_val,y_prednew).ravel()

precision=tp/(tp+fp)

recall=tp/(tp+fn)

f1=(2*precision*recall)/(precision+recall)

print("f1=",f1)
ts=time.time()

clf = GradientBoostingClassifier(n_estimators=600, learning_rate=1.0,

                                 max_depth=1, random_state=0).fit(train_tfidf, target_vectors)



print("time taken=",time.time()-ts)



y_pred=clf.predict_proba(test_tfidf)

predicted_probs=pd.DataFrame(y_pred)

predicted_probs[2]=predicted_probs[0].apply(lambda x: 0 if x>thres else 1)

y_prednew=predicted_probs[2].values

a=pd.DataFrame({'id':test.id,'target':y_prednew})

#a.to_csv('submission.csv',index=False)
abs(np.array(test_preds)-y_prednew).sum()
y_prednew
ts=time.time()

f1_sets=[]

for  i in np.arange(100,1000,step=100):

    clf = GradientBoostingClassifier(n_estimators=i, learning_rate=1.0,

                                 max_depth=1, random_state=0).fit(X_train, y_train)

    y_pred=clf.predict_proba(X_val)

    predicted_probs=pd.DataFrame(y_pred)

    predicted_probs[2]=predicted_probs[0].apply(lambda x: 0 if x>thres else 1)

    y_prednew=predicted_probs[2].values

    tn, fp, fn, tp =confusion_matrix(y_val,y_prednew).ravel()

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1=(2*precision*recall)/(precision+recall)

    f1_sets.append(f1)

    print("number of estimators of last run=",i)

print("time taken=",time.time()-ts)
f1_sets=np.array(f1_sets)
plt.scatter(np.arange(100,1000,step=100),f1_sets)
print("mean of training targets",y_train.mean())

print("mean of validation targets",y_val.mean())

print()