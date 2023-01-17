import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/nlp-getting-started/train.csv',delimiter=',')

test=pd.read_csv('../input/nlp-getting-started/test.csv',delimiter=',')
train.head()
train.tail()
train.describe()
train.info()
real=train[train['target']==1]

unreal=train[train['target']==0]
real
unreal
print('percentage of real tweets: ',len(real)/len(train)*100)
sns.countplot(x=train['target'])
train.keyword.value_counts()
plt.figure(figsize=(20,10))

sns.pairplot(train,hue='target')
train.keyword.unique()
output=train['target'].values
output
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()

target_convert=vectorizer.fit_transform(train['text'])
target_convert.shape
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier()

classifier.fit(target_convert,output)
from sklearn.naive_bayes import MultinomialNB

NB_classifier=MultinomialNB()
NB_classifier.fit(target_convert,output)
x=target_convert

y=output
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25)
NB_classifier.fit(x_train,y_train)
from sklearn.metrics import classification_report, confusion_matrix
y_pred_train=NB_classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred_train)

sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred_train))
test.head()
test.info()
test_vector=vectorizer.transform(test['text'])
print(test_vector.toarray())
test.head()
encoded=pd.DataFrame(test_vector.toarray())
encoded.shape
result=NB_classifier.predict(encoded)
result_random_forest=classifier.predict(encoded)
result
final=pd.DataFrame({'id':test['id'],'target':result})

final_random=pd.DataFrame({'id':test['id'],'target':result_random_forest})
final_random
final
final.to_csv('kaggle_submission.csv',index=False)
final.to_csv('kaggle_submission_randomforest.csv',index=False)