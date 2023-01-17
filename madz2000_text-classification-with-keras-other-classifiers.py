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

from nltk.stem import LancasterStemmer,WordNetLemmatizer

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from string import punctuation

from nltk import pos_tag

from nltk.corpus import wordnet

import keras

from keras.layers import Dense

from keras.models import Sequential

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("../input/hackereartheffectivenessofstddrugs/train.csv")

test_df = pd.read_csv("../input/hackereartheffectivenessofstddrugs/test.csv")
train_df.head()
train_df.isna().sum()
train_df['text'] = train_df['review_by_patient'] + ' ' + train_df['use_case_for_drug'] + ' ' + train_df['name_of_drug']

del train_df['review_by_patient']

del train_df['use_case_for_drug']

del train_df['name_of_drug']

del train_df['drug_approved_by_UIC']

del train_df['patient_id']
train_df.head()
train_df.effectiveness_rating.value_counts()
# Max and Min Rating

min_rating = train_df.effectiveness_rating.min()

max_rating = train_df.effectiveness_rating.max()

min_rating , max_rating
def scale_rating(rating):

    # Sacling from (1,10) to (0,5) and then replacing 0,1,2 in ratings with 0 (poor) and 3,4,5 with 1 (good).

    rating -= min_rating

    rating = rating/(max_rating - 1)

    rating *= 5

    rating = int(round(rating,0))

    if(int(rating) == 0 or int(rating) == 1 or int(rating) == 2):

        return 0

    else:

        return 1

    
train_df.effectiveness_rating = train_df.effectiveness_rating.apply(scale_rating)
train_df.effectiveness_rating.value_counts()
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def get_simple_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            pos = pos_tag([i.strip()])

            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))

            final_text.append(word.lower())

    return " ".join(final_text)
train_df.text = train_df.text.apply(lemmatize_words)
train_df.head()
test_df.head()
plt.figure(figsize = (20,20)) # Poor Reviews

wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(train_df[train_df.effectiveness_rating == 0].text)))

plt.imshow(wc,interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Good Reviews

wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(train_df[train_df.effectiveness_rating == 1].text)))

plt.imshow(wc,interpolation = 'bilinear')
x_train,x_test,y_train,y_test = train_test_split(train_df.text , train_df.effectiveness_rating)
cv=CountVectorizer(min_df=0,max_df=1,ngram_range=(1,3))

#transformed train reviews

cv_train_reviews=cv.fit_transform(x_train)

#transformed test reviews

cv_test_reviews=cv.transform(x_test)



print('BOW_cv_train:',cv_train_reviews.shape)

print('BOW_cv_test:',cv_test_reviews.shape)
model = Sequential()

model.add(Dense(units = 100 , activation = 'relu' , input_dim = cv_train_reviews.shape[1]))

model.add(Dense(units = 75 , activation = 'relu'))

model.add(Dense(units = 50 , activation = 'relu'))

model.add(Dense(units = 25 , activation = 'relu'))

model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model.summary()
model.fit(cv_train_reviews,y_train , epochs = 5)
pred = model.predict(cv_test_reviews)
for i in range(len(pred)):

    if(pred[i] > 0.5):

        pred[i] = 1

    else:

        pred[i] = 0
accuracy_score(pred,y_test)
cv_report = classification_report(y_test,pred,target_names = ['Poor Review','Good Review'])

print(cv_report)
cm_cv = confusion_matrix(y_test,pred)

cm_cv
cm_cv = pd.DataFrame(cm_cv, index=[0,1], columns=[0,1])

cm_cv.index.name = 'Actual'

cm_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')
mnb = MultinomialNB()

mnb.fit(cv_train_reviews,y_train)
mnb_pred = mnb.predict(cv_test_reviews)

accuracy_score(y_test , mnb_pred)
cv_report = classification_report(y_test,mnb_pred,target_names = ['Poor Review','Good Review'])

print(cv_report)
cm_cv = confusion_matrix(y_test,mnb_pred)

cm_cv
cm_cv = pd.DataFrame(cm_cv, index=[0,1], columns=[0,1])

cm_cv.index.name = 'Actual'

cm_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')
rf = RandomForestClassifier(max_depth = 20 , random_state = 0)

rf.fit(cv_train_reviews,y_train)
rf_pred = rf.predict(cv_test_reviews)

accuracy_score(y_test,rf_pred)
cv_report = classification_report(y_test,rf_pred,target_names = ['Poor Review','Good Review'])

print(cv_report)
cm_cv = confusion_matrix(y_test,rf_pred)

cm_cv
cm_cv = pd.DataFrame(cm_cv, index=[0,1], columns=[0,1])

cm_cv.index.name = 'Actual'

cm_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))

sns.heatmap(cm_cv,cmap= "Blues",annot = True, fmt='')