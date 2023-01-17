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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
false = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
true.head()
false.head()
true['category'] = 1
false['category'] = 0
true.head()
df = pd.concat([true,false]) 
df.isna().sum()
df.title.count()
df.subject.value_counts()
df['text'] = df['text'] + " " + df['title'] + " " + df['subject']
del df['title']
del df['subject']
del df['date']
df.head()
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
stop
stemmer = PorterStemmer()
def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)    
df.text = df.text.apply(stem_text)
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df.text))
plt.imshow(wc , interpolation = 'bilinear')
x_train,x_test,y_train,y_test = train_test_split(df.text,df.category)
cv=CountVectorizer(min_df=0,max_df=1,ngram_range=(1,2))
#transformed train reviews
cv_train_reviews=cv.fit_transform(x_train)
#transformed test reviews
cv_test_reviews=cv.transform(x_test)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
model = Sequential()
model.add(Dense(units = 100 , activation = 'relu' , input_dim = cv_train_reviews.shape[1]))
model.add(Dense(units = 50 , activation = 'relu'))
model.add(Dense(units = 25 , activation = 'relu'))
model.add(Dense(units = 10 , activation = 'relu'))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.fit(cv_train_reviews,y_train , epochs = 5)
from joblib import dump, load
dump(model, 'CNN.model') 

pred = (model.predict(cv_test_reviews) > 0.9999)
#for i in range(len(pred)):
 #   if(pred[i] ==1):
  #      pred[i] = 1
   # else:
    #    pred[i] = 0
#0.7
accuracy_score(pred,y_test)
#0.8
accuracy_score(pred,y_test)
#0.9999
accuracy_score(pred,y_test)
#cv_report = classification_report(y_test,pred,target_names = ['0','1'])
#print(cv_report)
#0.7
cm_cv = confusion_matrix(pred,y_test)
cm_cv
#0.8
cm_cv = confusion_matrix(pred,y_test)
cm_cv
#0.9999
cm_cv1 = confusion_matrix(pred,y_test)
cm_cv1
cm_cv2 = confusion_matrix(y_test,pred)
cm_cv2
#cm_cv = pd.DataFrame(cm_cv, index=[0,1], columns=[0,1])
#cm_cv.index.name = 'Actual'
#cm_cv.columns.name = 'Predicted'
#0.7
plt.figure(figsize = (10,10))
sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')
# confusion_matrix(pred,y_test)


                     #Actual Negative    #Actual Positive                          #y_test(second variable)
#Predicted Negative   True Negative       False Negative
#Predicted Positive   False Positive      True Positive
    
    
    
#pred(first variable)
#0.8
plt.figure(figsize = (10,10))
sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')
#0.9999
plt.figure(figsize = (10,10))
sns.heatmap(cm_cv1,cmap= "Blues",annot = True, fmt='')