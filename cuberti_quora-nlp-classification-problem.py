import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from fastai.nlp import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import *
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
print("Training Data Size: " + str(train.shape))
print("Test Data Size: " + str(test.shape))
train.head()
train = train.sample(frac = 0.2, random_state = 42)
trn=train.iloc[:,1]
y = train.iloc[:,2]
x,x_test,y,y_test = train_test_split(trn, y,test_size = 0.2, random_state = 42)
def print_scores(y_test, pred):
    print("Accuracy: "+ str(accuracy_score(y_test, pred)))
    print("F1 Score: " + str(f1(y_test, pred)))

def f1(y_true, y_pred):
    tp = np.logical_and(y_pred == 1, y_true == 1).sum() #true positives
    fn = np.logical_and(y_pred == 0, y_true == 1).sum() #False Negatives
    fp = np.logical_and(y_pred == 1, y_true == 0).sum() #False Positives
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    return 2*((p*r)/(p+r))

#lets try it out
y_pred = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1])
y_true = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1])
print_scores(y_true, y_pred)
vec = CountVectorizer(tokenizer=tokenize) #Using the FastAI tokenizer
train_tdm = vec.fit_transform(x).sign()
test_tdm = vec.transform(x_test).sign()
train_tdm
m_logreg = LogisticRegression(C= 1e10, dual = True, max_iter = 1000) #C=1e-1 95% accuracy 
m_logreg.fit(train_tdm, y)
pred = m_logreg.predict(test_tdm)
print_scores(y_test, pred)
vec = CountVectorizer(ngram_range = (1,3),tokenizer=tokenize, max_features = 1000000)
train_tdm_ngram = vec.fit_transform(x).sign()
test_tdm_ngram = vec.transform(x_test).sign()
train_tdm
%%time
m_logreg = LogisticRegression(C= 1e10, dual = True, max_iter=1000) 
m_logreg.fit(train_tdm_ngram, y)
pred = m_logreg.predict(test_tdm_ngram)
print_scores(y_test, pred)
train_tdm_ngram
def pr(x,y,y_i):
    p= x[(y==y_i).values].sum(0)
    return (p+1)/((y==y_i).sum()+1)
r = np.log(pr(train_tdm_ngram,y,1)/pr(train_tdm_ngram,y,0))
b = np.log((y==1).mean()/ (y==0).mean())
pre_pred = test_tdm_ngram @ r.T + b
preds = pre_pred>0
print_scores(y_test, pd.DataFrame(preds).iloc[:,0])
x_nb = train_tdm_ngram.multiply(r)
m= LogisticRegression(C= 1e10, dual = True, max_iter=1000) 
m.fit(x_nb, y)
x_test_nb = test_tdm_ngram.multiply(r)
pred = m.predict(x_test_nb)
print_scores(y_test, pred)
sl = 2000
md = TextClassifierData.from_bow(train_tdm_ngram, y, test_tdm_ngram, y_test, sl)
learner = md.dotprod_nb_learner()
learner.fit(0.02, 2, wds=1e-6, cycle_len=1)
preds=learner.predict()
preds=pd.DataFrame(preds)[1]>0
print_scores(y_test, preds)
from keras.models import *
from keras.layers import *
import tensorflow as tf
import keras.backend as K
tf.Session()
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=len(vec.get_feature_names())))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics=['binary_accuracy'])
model.summary()
%%time
model.fit(train_tdm_ngram, y, epochs=3, batch_size = 64)
preds=model.predict(test_tdm_ngram)
pred=preds[:,0]>0.19
print_scores(y_test, pred)
test_text = test['question_text']
test_tdm = vec.transform(test_text).sign() #maintain same VEC structure 
final_pred=model.predict(test_tdm)[:,0]
my_submission = pd.DataFrame({'qid': test.qid, 'prediction': final_pred.astype(int)})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()

