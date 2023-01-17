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
from nltk.corpus import stopwords
from nltk import word_tokenize
import matplotlib.pyplot as plt
import wordcloud
import string
sms = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')
sms.head(10)
sms.columns=['label','text','A','B','C']
sms.drop(['A','B','C'],axis=1,inplace=True)
sms.head(6)
sms.groupby('label').describe()
pd.value_counts(sms['label'])
sns.countplot(x=sms['label'],data=sms)
sms['label']=sms['label'].map({'ham':0,'spam':1}).astype(int)
sms.head(10)
sms.describe()
stop_words=set(stopwords.words("english"))
sms['text'].iloc[5571]
#words
def text_preprocess(text):
    words=text.lower().split()
    actual_list=[]
    for word in words:
        if word not in stop_words:
            if word not in string.punctuation:
                actual_list.append(word)
    return(' '.join(actual_list))            
sms['text'].head().apply(text_preprocess)
data_ham  = sms[sms['label'] == 0].copy()
data_spam = sms[sms['label'] == 1].copy()
def show_wordcloud(data_spam_or_ham, header):
    text = ' '.join(data_spam_or_ham['text'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='lightgrey',
                    colormap='viridis', width=800, height=600).generate(text)
    
    plt.figure(figsize=(12,6), frameon=True)
    plt.imshow(fig_wordcloud)  
    plt.axis('off')
    plt.title(header, fontsize=20 )
    plt.show()
show_wordcloud(data_spam, "SPAM Words")
show_wordcloud(data_ham, "HAM Words")
from sklearn.feature_extraction.text import CountVectorizer
Cnt_Vector = CountVectorizer(analyzer=text_preprocess)
SMS_TEXTS=Cnt_Vector.fit_transform(sms['text'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(SMS_TEXTS, sms['label'], test_size=0.20, random_state=0)
print(X_train)
# Create and train the naive Bayes classifier
# The multinomial Naive Bayes classifier is suitable for classification with discrete features .We shall explore others as we go ahead

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)
# Evaluate the model on the training data set

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train, pred))
print()
print('Confusion Matrix:\n',confusion_matrix(y_train, pred))
print()
print('Model Accuracy : ',accuracy_score(y_train, pred))
#Evaluate the model on the test data set

pred = classifier.predict(X_test)
print(classification_report(y_test, pred))
print()
print('Confusion Matrix:\n',confusion_matrix(y_test, pred))
print()
print('Model Accuracy : ',accuracy_score(y_test, pred))
test=['Your number as 10000']
x = Cnt_Vector.transform(test)
classifier.predict(x)[0]
