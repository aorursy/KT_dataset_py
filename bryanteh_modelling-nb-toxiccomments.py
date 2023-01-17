# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

df_test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Size of dataset: ',df.shape)

print(df.columns)
display(df.describe())
print('Count of cases positive:')

positive = []

positive_perc = []

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for i in label_cols:

    print(i,'-',len(df[df[i]==1]),',',str(round(100*len(df[df[i]==1])/len(df),2))+'%')

    positive.append(len(df[df[i]==1]))

    positive_perc.append(len(df[df[i]==1])/len(df))
temp = df

temp['length'] = temp.comment_text.apply(lambda x: len(x))

temp['length'] = temp.length.apply(lambda x: np.log(x))

plt.figure(figsize=(20,16))

import seaborn as sns

plt.style.use('fivethirtyeight')



print('Comments that are flagged tend to have shorter length')



label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for i in range(len(label_cols)):

    plt.subplot(2,3,i+1)

    sns.kdeplot(temp[temp[label_cols[i]]==0]['length'],label='ntf')

    sns.kdeplot(temp[temp[label_cols[i]]==1]['length'],label='flagged')

    plt.title(label_cols[i])
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for j in range(len(label_cols)):

    print(label_cols[j],'comments:')

    for i in range(len(df)):

        print(str(i+1),':',list(df[df[label_cols[j]]==1].comment_text)[i],'\n')

        if i == 1: break
from nltk.corpus import wordnet

from collections import Counter



def get_part_of_speech(word):

  probable_part_of_speech = wordnet.synsets(word)

  pos_counts = Counter()

  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )

  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )

  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )

  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )

  

  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]

  return most_likely_part_of_speech
print('checking')

print(list(df[df[label_cols[j]]==1].comment_text)[0])

print(list(df[df[label_cols[j]]==1].comment_text)[1])

print(list(df[df[label_cols[j]]==1].comment_text)[2])
import re

import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



print('BEFORE:\n',list(df[df[label_cols[j]]==1].comment_text)[0],'\n')



cleaned = re.sub('\W+',' ',list(df[df[label_cols[j]]==1].comment_text)[0]) 

print('AFTER clearing non-words:\n',cleaned,'\n')



lowered = cleaned.lower()

print('AFTER setting lowercase:\n',lowered,'\n')



tokenized = word_tokenize(lowered)

print('AFTER tokenizing:\n',tokenized,'\n')



filtered_stopwords = [w for w in tokenized if not w in stopwords.words('english')]

print('AFTER removing stopwords:\n',filtered_stopwords)



lemmatizer = WordNetLemmatizer()

lemmatized = [lemmatizer.lemmatize(i,get_part_of_speech(i)) for i in filtered_stopwords]

print('AFTER lemmatizing:\n',lemmatized)



regroup = " ".join(lemmatized)

print('AFTER regrouping:\n',regroup)
lemmatizer = WordNetLemmatizer()



def text_processing(text_list):

    processed_text = []

    for text in text_list:

        cleaned = re.sub('\W+',' ',text)

        lowered = cleaned.lower()

        tokenized = word_tokenize(lowered)

        filtered_stopwords = [w for w in tokenized if not w in stopwords.words('english')]

        lemmatized = [lemmatizer.lemmatize(i,get_part_of_speech(i)) for i in filtered_stopwords]

        processed_text.append(" ".join(lemmatized))

    return processed_text
from datetime import datetime

from datetime import timedelta

import time



before = datetime.now()

processed_text = text_processing(df['comment_text'][0:1000])

after = datetime.now()

time_delta = after - before

seconds = time_delta.total_seconds()

minutes = seconds/60

print(1000,'rows takes:',round(minutes,4),'minutes')

print(len(df)+len(df_test),'rows takes:',round((len(df)+len(df_test))*minutes/1000,4),'minutes')
processed_text = text_processing(df['comment_text'])

test_processed_text = text_processing(df_test['comment_text'])
no_positive = len(df[df['toxic'] == 1])

negative_indices = df[df['toxic']==0].index

random_negative_indices = np.random.choice(negative_indices,no_positive, replace=False)

positive_indices = df[df['toxic'] == 1].index

under_sample_indices = np.concatenate([positive_indices,random_negative_indices])

under_sample = df.loc[under_sample_indices]



retained_processed_text = []

for i in under_sample.index:

    retained_processed_text.append(processed_text[i])
#split train document with X_Test_y_test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(retained_processed_text, under_sample.toxic, test_size = 0.2, random_state = 6)

print(len(x_train))

print(len(x_test))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_creator = TfidfVectorizer(max_df=0.60)

x_train_tfidf = tfidf_creator.fit_transform(x_train)

x_test_tfidf = tfidf_creator.transform(x_test)
#not used for now CountVectorizer

# option = 2 #1, 2

# if option == 1:

#     from sklearn.feature_extraction.text import CountVectorizer

#     counter = CountVectorizer()

#     counter.fit(processed_text) 

#     counter.vocabulary_

#     vectorized_text = counter.transform(processed_text).toarray()

# else:

#     pass
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(x_train_tfidf,np.array(y_train))
y_pred = classifier.predict(x_train_tfidf)

classifier.predict_proba(x_train_tfidf)



from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

print('Train dataset:')

 

result = pd.DataFrame(confusion_matrix(y_train, y_pred))

display(result)

print('accuracy:\t',round(accuracy_score( y_train , y_pred),2))

print('recall:\t\t',round(recall_score( y_train , y_pred),2))

print('precision:\t',round(precision_score( y_train , y_pred),2))

print('f1:\t\t',round(f1_score( y_train , y_pred),2))
y_pred = classifier.predict(x_test_tfidf)

classifier.predict_proba(x_test_tfidf)



from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix



print('Test dataset:')

result = pd.DataFrame(confusion_matrix(y_test, y_pred))

display(result)

print('accuracy:\t',round(accuracy_score( y_test , y_pred),2))

print('recall:\t\t',round(recall_score( y_test , y_pred),2))

print('precision:\t',round(precision_score( y_test , y_pred),2))

print('f1:\t\t',round(f1_score( y_test , y_pred),2))
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

submissions = df_test[['id']]



#6 nb models to predict 6 flags

for i in label_cols:

    

    #undersampling for each nb model

    no_positive = len(df[df[i] == 1])

    negative_indices = df[df[i] == 0].index

    random_negative_indices = np.random.choice(negative_indices,no_positive, replace=False)

    positive_indices = df[df[i] == 1].index

    under_sample_indices = np.concatenate([positive_indices,random_negative_indices])

    under_sample = df.loc[under_sample_indices]



    retained_processed_text = []

    for j in under_sample.index:

        retained_processed_text.append(processed_text[j])

    

    #vectorized for each nb model

    tfidf_creator = TfidfVectorizer(max_df=0.60,binary=True)

    x_train_tfidf = tfidf_creator.fit_transform(retained_processed_text)

    x_test_tfidf = tfidf_creator.transform(test_processed_text)

        

    #nb model for each flags

    classifier = MultinomialNB()

    classifier.fit(x_train_tfidf,np.array(under_sample[i]))

    y_pred = classifier.predict(x_test_tfidf)

    submissions[i] = y_pred.reshape(-1,1)

    print('done for',i)

    

display(submissions.head(15))
submissions.to_csv('submission.csv',index=False)