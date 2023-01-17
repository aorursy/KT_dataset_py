import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os


from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
#1: unreliable
#0: reliable
train=pd.read_csv('../input/fake-news-dataset/train.csv')
test=pd.read_csv('../input/fake-news-dataset/test.csv')
        
SUBMIT_FILE = '../input/fake-news-dataset/submit.csv'
PREDICTED_SUBMIT = '/kaggle/working/submission.csv'
MODEL_PATH_ML='/kaggle/working/'
test.info()
test['label']='t'
train.info()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import recall_score,precision_score

#data prep
test=test.fillna(' ')
train=train.fillna(' ')
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+train['text']

#tfidf
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(train['total'].values)
tfidf = transformer.fit_transform(counts)

targets = train['label'].values
test_counts = count_vectorizer.transform(test['total'].values)
test_tfidf = transformer.fit_transform(test_counts)

#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)

Extr = ExtraTreesClassifier(n_estimators=5,n_jobs=4)
Extr.fit(X_train, y_train)
print('Accuracy of ExtrTrees classifier on training set: {:.2f}'
     .format(Extr.score(X_train, y_train)))
print('Accuracy of Extratrees classifier on test set: {:.2f}'
     .format(Extr.score(X_test, y_test)))
pickle.dump(Extr, open(os.path.join(MODEL_PATH_ML,'Extr.bin'), 'wb'))
from sklearn.tree import DecisionTreeClassifier

Adab= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
Adab.fit(X_train, y_train)
print('Accuracy of Adaboost classifier on training set: {:.2f}'
     .format(Adab.score(X_train, y_train)))
print('Accuracy of Adaboost classifier on test set: {:.2f}'
     .format(Adab.score(X_test, y_test)))
pickle.dump(Adab, open(os.path.join(MODEL_PATH_ML,'Adab.bin'), 'wb'))
X_train.shape
Rando= RandomForestClassifier(n_estimators=5)

Rando.fit(X_train, y_train)
print('Accuracy of randomforest classifier on training set: {:.2f}'
     .format(Rando.score(X_train, y_train)))
print('Accuracy of randomforest classifier on test set: {:.2f}'
     .format(Rando.score(X_test, y_test)))
pickle.dump(Rando, open(os.path.join(MODEL_PATH_ML,'Rando.bin'), 'wb'))

NB = MultinomialNB()
NB.fit(X_train, y_train)
print('Accuracy of NB  classifier on training set: {:.2f}'
     .format(NB.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'
     .format(NB.score(X_test, y_test)))
pickle.dump(NB, open(os.path.join(MODEL_PATH_ML,'NB.bin'), 'wb'))


model = xgb.XGBClassifier(objective="binary:logistic",random_state=7)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          
eval_metric='auc',
early_stopping_rounds=100)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
pickle.dump(model, open(os.path.join(MODEL_PATH_ML,'xgb.bin'), 'wb'))
df_submit = pd.read_csv(SUBMIT_FILE).fillna("none").reset_index(drop=True)
df_submit = df_submit.reset_index(drop=True)
y_test= df_submit['label']
loaded_model = pickle.load(open("/kaggle/working/xgb.bin", "rb"))
y_pred_xgb = loaded_model.predict(test_tfidf)
print(recall_score(y_test, y_pred_xgb)*100)
print(precision_score(y_test, y_pred_xgb)*100)

loaded_model_NB = pickle.load(open("/kaggle/working/NB.bin", "rb"))
y_pred_NB = loaded_model.predict(test_tfidf)
print(recall_score(y_test, y_pred_NB)*100)
print(precision_score(y_test, y_pred_NB)*100)

loaded_model_Rando = pickle.load(open("/kaggle/working/Rando.bin", "rb"))
y_pred_Rando = loaded_model.predict(test_tfidf)
print(recall_score(y_test, y_pred_Rando)*100)
print(precision_score(y_test, y_pred_Rando)*100)

loaded_model_Extr = pickle.load(open("/kaggle/working/Extr.bin", "rb"))
y_pred_Extr = loaded_model.predict(test_tfidf)
print(recall_score(y_test, y_pred_Extr)*100)
print(precision_score(y_test, y_pred_Extr)*100)

loaded_model_Adab = pickle.load(open("/kaggle/working/Adab.bin", "rb"))
y_pred_Adab = loaded_model.predict(test_tfidf)
print(recall_score(y_test, y_pred_Adab)*100)
print(precision_score(y_test, y_pred_Adab)*100)



# df_submit['predicted'] =y_pred
# df_submit.to_csv(PREDICTED_SUBMIT)


tfidf.shape