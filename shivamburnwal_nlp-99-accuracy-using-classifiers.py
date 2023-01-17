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
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud,STOPWORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense
df = pd.read_csv("/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv")
df.head()
df.isna().sum()
# deleting columns.
del df['author_flair_text']
del df['removed_by']
del df['total_awards_received']
del df['awarders']
del df['id']
del df['created_utc']
del df['full_link']
df.head()
df.isna().sum()
# taking care of nan in title
df.title.fillna(" ",inplace = True)
df['text'] = df['title'] + ' ' + df['author']
del df['title']
del df['author']
df.head()
df.over_18.replace([True, False], [1, 0], inplace=True)
df.over_18.value_counts()
over18text_false = df[df.over_18 == 0.0].text
over18text_true = df[df.over_18 == 1.0].text
train_text = df.text.values[:100000]
test_text = df.text.values[100000:]
train_category = df.over_18[:100000]
test_category = df.over_18[100000:]
plt.figure(figsize = (20,20))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(over18text_true)))
plt.imshow(wc,interpolation = 'bilinear')
plt.figure(figsize = (20,20))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(over18text_false)))
plt.imshow(wc,interpolation = 'bilinear')
# Getting the most frequently used words from wordcloud
text_true = wc.process_text(str(" ".join(over18text_true)))
text_true
text_true = sorted(text_true.items(),key = lambda kv:(kv[1], kv[0]))
ans_true = []
for i in text_true:
    ans_true.append(i[0])
ans_true [:5] 
predictions = []
for i in test_text:
    x = i.split()
    for j in x:
        if j in ans_true:
            predictions.append(1)
            break
        else:
            predictions.append(0)
            break
len(predictions)
count = 0
for i in range(len(predictions)):
    test_category = list(test_category)
    if(predictions[i] == int(test_category[i])):
        count += 1
print(count)
accuracy = (count/len(predictions))*100
accuracy
print("Accuracy using WordCloud is : ", accuracy , "%")
stop = set(stopwords.words('english'))
punctuation = list(punctuation)
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
# Function to clean our text.
lemmatizer = WordNetLemmatizer()
def clean_review(text):
    clean_text = []
    for w in word_tokenize(text):
        if w.lower() not in stop:
            pos = pos_tag([w])
            new_w = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))
            clean_text.append(new_w)
    return clean_text

def join_text(text):
    return " ".join(text) 
df.text = df.text.apply(clean_review)
df.text = df.text.apply(join_text)
df.head()
train_text, test_text, train_category, test_category = train_test_split(df.text, df.over_18, random_state=0)
train_text.shape, test_text.shape, train_category.shape, test_category.shape
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,2))
#transformed train reviews
cv_train_reviews=cv.fit_transform(train_text)
#transformed test reviews
cv_test_reviews=cv.transform(test_text)

print('cv_train:',cv_train_reviews.shape)
print('cv_test:',cv_test_reviews.shape)
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))
#transformed train reviews
tv_train_reviews=tv.fit_transform(train_text)
#transformed test reviews
tv_test_reviews=tv.transform(test_text)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)
lr=LogisticRegression()

# Fitting the model
lr_cv=lr.fit(cv_train_reviews,train_category)
lr_tfidf=lr.fit(tv_train_reviews,train_category)

# Predicting for model
lr_cv_predict=lr_cv.predict(cv_test_reviews)
lr_tfidf_predict=lr_tfidf.predict(tv_test_reviews)
# Getting Score

lr_cv_score=accuracy_score(test_category,lr_cv_predict)
print("lr_cv_score :",lr_cv_score)

lr_tfidf_score=accuracy_score(test_category,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)
print(classification_report(test_category,lr_cv_predict))
print(classification_report(test_category,lr_tfidf_predict))
plot_confusion_matrix(lr_cv, cv_test_reviews, test_category, cmap="Accent", values_format = '')
plot_confusion_matrix(lr_tfidf, tv_test_reviews, test_category, cmap="Accent", values_format = '')
#training the model
nb=MultinomialNB()

# Fitting the model
nb_cv=nb.fit(cv_train_reviews,train_category)
nb_tfidf=nb.fit(tv_train_reviews,train_category)

# Predicting for model
nb_cv_predict=nb_cv.predict(cv_test_reviews)
nb_tfidf_predict=nb_tfidf.predict(tv_test_reviews)
# Getting Score

nb_cv_score=accuracy_score(test_category,nb_cv_predict)
print("nb_cv_score :",nb_cv_score)

nb_tfidf_score=accuracy_score(test_category,nb_tfidf_predict)
print("nb_tfidf_score :",nb_tfidf_score)
print(classification_report(test_category,nb_cv_predict))
print(classification_report(test_category,nb_tfidf_predict))
plot_confusion_matrix(nb_cv, cv_test_reviews, test_category, cmap="Blues", values_format = '')
plot_confusion_matrix(nb_tfidf, tv_test_reviews, test_category, cmap="Blues", values_format = '')
#svc=SVC()
# Fitting the model
#svc_cv=svc.fit(cv_train_reviews,train_category)
#svc_tfidf=svc.fit(tv_train_reviews,train_category)

# Predicting for model
#svc_cv_predict=nb_cv.predict(cv_test_reviews)
#svc_tfidf_predict=nb_tfidf.predict(tv_test_reviews)
# Getting Score

#svc_cv_score=accuracy_score(test_category,svc_cv_predict)
#print("svc_cv_score :", svc_cv_score)

#svc_tfidf_score=accuracy_score(test_category,svc_tfidf_predict)
#print("svc_tfidf_score :", svc_tfidf_score)
#print(classification_report(test_category,svc_cv_predict))
#print(classification_report(test_category,svc_tfidf_predict))
#plot_confusion_matrix(svc_cv, cv_test_reviews, test_category, cmap="Blues", values_format = '')
#plot_confusion_matrix(svc_tfidf, tv_test_reviews, test_category, cmap="Blues", values_format = '')
model = Sequential()
model.add(Dense(units = 100, activation = 'relu' , input_dim = cv_train_reviews.shape[1]))
model.add(Dense(units = 20, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()
history = model.fit(cv_train_reviews,train_category , epochs=1, batch_size = 512, validation_data=(cv_test_reviews,test_category))
predictions = model.predict_classes(cv_test_reviews)
predictions[:5]
print(classification_report(test_category, predictions))
cm = confusion_matrix(test_category,predictions)
cm = pd.DataFrame(cm , index=['0','1'] , columns=['0','1'])
plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor='black', annot=True, fmt='')
