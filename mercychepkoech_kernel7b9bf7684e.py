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
import seaborn as sns

train_df=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_df=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
print('rows in train dataset:',train_df.shape[0])
print('rows in test dataset:',test_df.shape[0])

print('rows in sample dataset:',sample_df.shape[0])
train_df.isnull().sum()
from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist
test_df.isnull().sum()
train_df.head()
train_df.info()
train_df.columns

test_df.columns
test_df.head()
grouped= train_df.groupby('target')

for target_type, group in grouped:

    print("Target type:",target_type )

    cnt = 0

    for ind, row in group.iterrows():

        print(row["text"])

        cnt += 1

        if cnt == 5:

            break

    print("\n")
df=train_df.append(test_df)

print("Missing values:",(df.target.isnull().sum()))
train_df.head()

test_df.head()
test_df.info()
train_df['text_length']=train_df['text'].apply(len)
train_df.head()
train_df.text_length.describe()
train_df[train_df['text_length']==157]
train_df[train_df['text_length']==157]['text'].iloc[0]
sns.boxplot(x='target',y='text_length',data=train_df,palette='rainbow')
import string

from nltk.corpus import stopwords

def text_process(text):

    #check for punc

    nopunc=[char for char in text if char not in string.punctuation]

    #joins to form a string

    nopunc=''.join(nopunc)

    #remove stopwords

    splited=nopunc.split()

    return[word for word in nopunc.split() if word.lower()not in stopwords.words('english')]
train_df.head()
train_df['text'].head()
train_df['text'].head().apply(text_process)
#vectorization

from sklearn.feature_extraction.text import CountVectorizer
#bag of words

bow_transformer=CountVectorizer(analyzer=text_process).fit(train_df['text'])
len(bow_transformer.vocabulary_)
text4=train_df['text'][3]
text4
bow4=bow_transformer.transform([text4])
print(bow4)
print(bow_transformer.get_feature_names()[22635])
#transforming our text

text_bow=bow_transformer.transform(train_df['text'])
text_bow.shape
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(text_bow)
tfidf4=tfidf_transformer.transform(bow4)
print(tfidf4)
text_tfidf=tfidf_transformer.transform(text_bow)
print(text_tfidf)
#classifying using bayes

from sklearn.naive_bayes import MultinomialNB
tweet_detect_model=MultinomialNB().fit(text_tfidf,train_df['target'])
tweet_detect_model.predict(tfidf4)
tweet_detect_model.predict(text_tfidf)
from sklearn.model_selection import train_test_split
#train_test_split()

text_train, text_test, target_train, target_test = train_test_split(train_df['text'], train_df['target'], test_size=0.3, random_state=101)
from sklearn.pipeline import Pipeline

#pipeline is used to combine every NLP steps
pipeline=Pipeline([

    ('bow',CountVectorizer(analyzer=text_process)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB()),

])
pipeline.fit(text_train,target_train)
predict=pipeline.predict(text_test)
from sklearn.metrics import classification_report
classification_report(predict,target_test)