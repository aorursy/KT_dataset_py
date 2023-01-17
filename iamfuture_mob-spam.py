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
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sn
import wordcloud

warnings.filterwarnings('ignore')


data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',usecols=['v1','v2'],encoding='latin-1')

print(data.shape)
#data[data.columns[data.isnull().any()]].isnull().sum() - to find null values
#therefore use only 2 columns
data.head()
#making data column readable
data.columns = ['Category','Message']
#finding the categories present
print(data['Category'].value_counts())
#we have less spam compared to ham (4825:747)
# Ratio = 6:1
#Using colab, following installation is required
#seems similar for kaggle XD
!pip install contextualSpellCheck
!pip install -U spacy
!python -m pip install transformers
import nltk
nltk.download()
import spacy
import contextualSpellCheck

nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)
data['character_count'] = data['Message'].str.len()
print(data[['Message','character_count','Category']].head())
#from this we can say that it's not necessary that spam has more char count
#finding numerics in the message
data['numerics'] = data['Message'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(data[['Message','numerics','Category']].sample(10))
#we can see that numbers present with spacing are generally in spam messages
#One of our analysis but not very strong as even ham can have numbers
#finding symbols
#data['symbol'] = data['Message'].apply(lambda x: len([x for x in x.split() if x.find('\\')]))
#data[['Message','symbol','Category']].sample(10)
#we cannot predict from the symbols, hence we discard it
#making text readable
data['Message'] = data['Message'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Message'].head()
#removing punctuations - can cause problem
data['Message'] = data['Message'].str.replace('[^\w\s]','')
data['Message'].head()
#correcting the text using TextBlob, not very accurate
#DO NOT TRY USING CPU

#from textblob import TextBlob

#for messages in data['Message']:
#  correct_text = TextBlob(messages)
#  print(correct_text.correct())
data['Message'][0]
data.hist(column='character_count',by='Category',figsize=(10,8))
plt.show()

#to get an overview of complete data we add bins and xlim to the figure
data.hist(column='character_count',by='Category',bins=60,figsize=(10,8))
plt.xlim(-60,2000)
plt.show()

#We can see that one of our data is positively skewed whereas other is 
#negatively skewed
data['text_length'] = data['Message'].apply(lambda x: len(x) - x.count(" "))
bins = np.linspace(0, 300, 40)
plt.hist(data['text_length'],bins)
plt.title("Text Length Distribution")
plt.show()
data['spammed_data'] = data['Category'].map({'spam':1, 'ham':0}).astype(int)
ham_data = data[data['spammed_data'] == 0].copy()
spam_data = data[data['spammed_data'] == 1].copy()
def make_wordcloud(data_type,title):
  text = ' '.join(data_type['Message'].astype(str).tolist())
  stopwords = set(wordcloud.STOPWORDS)

  fig_wordcloud = wordcloud.WordCloud(stopwords = stopwords,width=800,height=600).generate(text)
  plt.figure(figsize=(10,7))
  plt.imshow(fig_wordcloud)
  plt.axis("off")
  plt.title(title,fontsize=24)
  plt.show()
make_wordcloud(ham_data,"Ham Message")
make_wordcloud(spam_data,"Spam Message")
#tokenizing text

def make_tokens(text):
  token = re.split('\W+',text)
  return token

data['tokenized_message'] = data['Message'].apply(lambda row: make_tokens(row))
data.head()

#removing stopwords

stopwords = nltk.corpus.stopwords.words('english')

def rem_stop(text):
  clean_text = [word for word in text if word not in stopwords]
  return clean_text

data['Clean_message'] = data['tokenized_message'].apply(lambda row: rem_stop(row))
data.head()
# using PorterStemmer to stem our words
#reason for taking STEMMER is that the prediction should also be able to 
#pick up the HAM messages or else it will classify it as spam
porter = nltk.PorterStemmer()

def stemmed(text):
  stemmed_text = [porter.stem(word) for word in text]
  return stemmed_text

data['Stemmed_message'] = data['Clean_message'].apply(lambda row: stemmed(row))
data[['Clean_message','Stemmed_message']].head()
#making the stemmed words join to sentence

def final_message(text):
  final_message = " ".join([word for word in text])
  return final_message

data['Final_message'] = data['Stemmed_message'].apply(lambda row: final_message(row))
data.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_model = TfidfVectorizer()
#fit the model
tfidf_vector = tfidf_model.fit_transform(data.Final_message)
#convert vector to array
tfidf_data = pd.DataFrame(tfidf_vector.toarray())
print(tfidf_data)
#now we have put values to the words with Tfidf
#length of text can have an impact on our data
#adding text_len to our data

final_da = pd.concat([data['text_length'],tfidf_data],axis=1)
final_da.head()
#calculating the precission and recall
from sklearn.metrics import precision_recall_fscore_support as pfscore
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(final_da,data['Category'],test_size=0.25)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
from sklearn.ensemble import RandomForestClassifier

Rf = RandomForestClassifier(n_estimators=60,max_depth=None,n_jobs=-1)
rf_model = Rf.fit(X_train,Y_train)
rf_pred = rf_model.predict(X_test)
prec,recall,fscore,support = pfscore(Y_test,rf_pred,pos_label='spam',average='binary')
#we have only 2 categories here
print('Precision:{}\nRecall:{}\nAccuracy:{}'.format(round(prec,4),round(recall,4),
                                                    round((rf_pred==Y_test).sum()/len(rf_pred),4)))
print(sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10])
