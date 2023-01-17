# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re
import pandas as pd
pd.set_option("display.max_columns",None)
train_df= pd.read_csv("../input/quora-insincere-questions-classification/train.csv",index_col=False)
test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv",index_col=False)
test_id= test_df['qid']

target = train_df['target']
train_df = train_df.drop(['target'],axis=1)

df = pd.concat([train_df,test_df],axis=0)
df.head()
df.tail()
df.isnull().sum()
df.shape

import spacy
nlp =spacy.load('en_core_web_lg')

df['question_text']=df['question_text'].apply(lambda x: x.lower())
df['question_text']=df['question_text'].apply(lambda x: re.sub("[^a-z A-Z 0-9-]+",'',x))
df['question_text']=df['question_text'].apply(lambda x: " ".join(x.split()))

import unicodedata
def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('utf-8','ignore')
    return x

# df['question_text']=df['question_text'].apply(remove_accented_chars(str(df['question_text'])))

#Lemmatization
def make_to_base(x):
    x_list = []
    docs=nlp.pipe(x, batch_size=32, disable=["parser", "ner"])
    for doc in docs:
        lemma= [tok.lemma_ for tok in doc]
#        print(lemma)
        if lemma=='-PRON-' or lemma=='be':
            lemma=lemma.text
        x_list.append(lemma)
    # return "".join(str(x_list))
    # return "".join(str(x_list))
    return x_list

# x=['hi makes would john','dadsa']
#make_to_base(x)

df['question_text'] = (make_to_base(df['question_text']))

df['question_text']=df['question_text'].apply(lambda x: re.sub("[^a-z A-Z]+",'',str(x)))
df.head()

test_df.shape[0]
x = df.iloc[:train_df.shape[0],1]
test = df.iloc[train_df.shape[0]:,1]
y = target
x.shape,test.shape,y.shape
train_df.shape,test_df.shape,target.shape
# y = df.iloc[:,2]

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,shuffle=True)
train_x.shape, test_x.shape
from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(ngram_range=(1,4),min_df=3,max_df=0.9,strip_accents='unicode',use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words='english')
tfidf = TfidfVectorizer(min_df=3,max_df=0.9,strip_accents='unicode',use_idf=True,smooth_idf=True,sublinear_tf=True)
tfidf.fit(train_x)
train_x= tfidf.transform(train_x)
test_x= tfidf.transform(test_x)
test= tfidf.transform(test)

from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.5, max_iter=40)
classifier = LogisticRegression(max_iter=800)

classifier.fit(train_x, train_y)

classifier.score(train_x, train_y), classifier.score(test_x, test_y)

predicted= classifier.predict(test)
predicted[1]


sample_df=pd.read_csv('../input/quora-insincere-questions-classification/sample_submission.csv',index_col=False)
sample_df.head(2)

df_sub=pd.DataFrame(predicted,columns=sample_df.columns[1:])
df_sub.head(2)
df_sub1=pd.DataFrame(test_id)
df_sub1.head(2)

final_sub=pd.concat([df_sub1,df_sub],axis=1)
final_sub.to_csv('submission.csv',index=False)
