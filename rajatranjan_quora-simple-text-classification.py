import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
train_df.head()
train_df.target.value_counts()
train_df.question_text.sample(10).values
train_df['length']=train_df.question_text.apply(lambda x: len(x))
train_df.head()
test_df['length']=test_df.question_text.apply(lambda x: len(x))
test_df.head()
print('Average questions length in train is {0:.0f}.'.format(train_df.length.mean()))
print('Average questions length in test is {0:.0f}.'.format(test_df.length.mean()))
print()
print('Maximum questions length in train is {0:.0f}.'.format(train_df.length.max()))
print('Maximum questions length in test is {0:.0f}.'.format(test_df.length.max()))
print()
print('Minimum questions length in train is {0:.0f}.'.format(train_df.length.min()))
print('Minimum questions length in test is {0:.0f}.'.format(test_df.length.min()))
print()
print('Average word length of questions in train is {0:.0f}.'.format(np.mean(train_df['question_text'].apply(lambda x: len(x.split())))))
print('Average word length of questions in test is {0:.0f}.'.format(np.mean(test_df['question_text'].apply(lambda x: len(x.split())))))
print('Max word length of questions in train is {0:.0f}.'.format(np.max(train_df['question_text'].apply(lambda x: len(x.split())))))
print('Max word length of questions in test is {0:.0f}.'.format(np.max(test_df['question_text'].apply(lambda x: len(x.split())))))
print('Average character length of questions in train is {0:.0f}.'.format(np.mean(train_df['question_text'].apply(lambda x: len(x)))))
print('Average character length of questions in test is {0:.0f}.'.format(np.mean(test_df['question_text'].apply(lambda x: len(x)))))
import re
import nltk
from nltk.corpus import stopwords
def clean_text(raw_text):
    raw_text=raw_text.strip()
    try:
        no_encoding=raw_text.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        no_encoding = raw_text
    letters_only = re.sub("[^a-zA-Z]", " ",no_encoding) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 
train_df['clean_ques']=train_df.question_text.apply(clean_text)
train_df.sample(10)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix,classification_report,f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# nb = MultinomialNB()
ft=FeatureUnion([('ct', CountVectorizer(analyzer='char',ngram_range=(1,5),max_df=0.9)),('ct2', CountVectorizer(analyzer='word',ngram_range=(1,4),max_df=0.9))])
pipeline = Pipeline([
    ('bow',ft),  # strings to token integer counts
    ('clf', LogisticRegression(solver='saga', class_weight='balanced', C=0.45,max_iter=250, verbose=1))
])
pipeline.get_params().keys()
pipeline.fit(train_df['clean_ques'].values,train_df.target)
import gc
gc.collect()
test_df['clean_ques']=test_df.question_text.apply(clean_text)
pr=pipeline.predict(test_df['clean_ques'].values)
sub=pd.DataFrame({'qid':test_df.qid,'prediction':pr})
sub.prediction.value_counts()
sub.to_csv('submission.csv',index=False)