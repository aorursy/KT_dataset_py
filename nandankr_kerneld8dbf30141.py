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
import os
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix , roc_auc_score , classification_report,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
sentim_data=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
sentim_data.head(10)
sentim_data.info()

sentim_data.dropna(inplace=True)
sns.countplot(x='sentiment', data=sentim_data)
sentim_data.isnull().sum()
vs=sentim_data['sentiment'].value_counts()
vs
senti_nuteral=(sentim_data['sentiment']=='neutral').value_counts()
senti_pos=(sentim_data['sentiment']=='positive').value_counts()

senti_neg=(sentim_data['sentiment']=='negetive').value_counts()


data=list([senti_nuteral,senti_pos,senti_neg])
plt.violinplot(data)
senti_nuteral
sentim_data['no._textword']=sentim_data['text'].apply(lambda x:len(str(x).split()))
sentim_data['no._sel_text']=sentim_data['selected_text'].apply(lambda x:len(str(x).split()))
sentim_data.head()
sentim_data['word_diff']=sentim_data['no._textword']-sentim_data['no._sel_text']
sentim_data.head()
y1=sentim_data['no._textword']
y2=sentim_data['no._sel_text']
x=sentim_data['word_diff']
fig=plt.figure(figsize=(30,15))
plt.scatter(y1,y2)
plt.scatter(y1,x)
plt.legend(['y1','y2',x],loc='best')
plt.show()
temp = sentim_data.groupby('sentiment').count()['text']
temp
labels=['neutral','positive','negative']
size=[temp.neutral,temp.positive,temp.negative]
fig=plt.figure(figsize=(10,5))
my_explode = (0, 0.1, 0)
ax1=plt.subplot()
ax1.pie(size,labels=labels,autopct='%1.1f%%',shadow=True,startangle=15,explode=my_explode)
len_tweet = sentim_data['text'].apply(lambda x:len(str(x)))
len_selected = sentim_data['selected_text'].apply(lambda x:len(str(x)))
len_sentiment=sentim_data['sentiment'].apply(lambda x:len(str(x)))
fig=plt.figure(figsize=(30,15))
sns.distplot(len_tweet,norm_hist=True,color='orange')
sns.distplot(len_selected,norm_hist=True,color='yellow')

pcal=pd.get_dummies(sentim_data['sentiment'])
pcal
sentim_data=pd.concat([sentim_data,pcal],axis=1)
sentim_data.head()
def cleanTxt(text):
    text=re.sub(r'@[A-Za-z0-9]+','',text)
    text=re.sub(r'#','',text)
    text=re.sub(r'RT[\s]+','',text)
    return text

sentim_data['selected_text']=sentim_data['selected_text'].apply(cleanTxt)
sentim_data
     
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getpolarity(text):
    return TextBlob(text).sentiment.polarity
sentim_data['Subjectivity']=sentim_data['selected_text'].apply(getsubjectivity)
sentim_data['polarity']=sentim_data['selected_text'].apply(getpolarity)
sentim_data.head(15)
plt.figure(figsize=(10,5))
# for i in range(0,sentim_data.shape[0]):
plt.scatter(sentim_data['polarity'],sentim_data['Subjectivity'],marker='4',color='orange')
    
plt.title('sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.grid()
plt.show()
# here you see the majority of the sentiment is greater than 0
sentim_data.hist(bins=50 , figsize=(20,15))

corr_matrix=sentim_data.corr()
corr_matrix['polarity'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes =["polarity" , "positive" , "Subjectivity" , "no._sel_text","neutral","negative"]
scatter_matrix(sentim_data[attributes], figsize=(18,12))
sample_data=[['Good case for the money !','positive'],
            ['Do not waste your money.','negetive'],
            ['Good product. Really good.Love it!','positive']]
small_sample=pd.DataFrame(sample_data,columns=['selected_text','sentiment'])
count_vec=CountVectorizer(binary=False, stop_words='english', ngram_range=(1,1),max_features=50000)
count_vec.fit(small_sample.selected_text)
small_tranformed=count_vec.transform(small_sample.selected_text)
small_sample
from pandas import DataFrame
print(DataFrame(small_tranformed.A,columns=count_vec.get_feature_names()).to_string())
type(small_tranformed)
small_tranformed.A
tfidf=TfidfTransformer(use_idf=True)
tfidf.fit(small_tranformed)
small_tfidf=tfidf.transform(small_tranformed)
print(DataFrame(small_tfidf.A,columns=count_vec.get_feature_names()).to_string())
lr=LogisticRegression(penalty='l2',C=.8,random_state=21)
text_classifier=Pipeline([
    ('vectorizer',CountVectorizer(binary=False,stop_words='english',ngram_range=(1,2))),
    ('tfidf',TfidfTransformer(use_idf=True,)),
    ('clf',lr),
])

X_train, X_test, y_train, y_test = train_test_split(sentim_data.selected_text, sentim_data.sentiment, test_size=0.35, random_state=4)
text_classifier.fit(X_train,y_train)
X_test[0:4]

y_test[0:4]
predicted_test = text_classifier.predict(X_test)
predicted_proba_test = text_classifier.predict_proba(X_test)



from sklearn import metrics
# for training data
predicted_train = text_classifier.predict(X_train)

y_train = y_train.astype('category')
print(metrics.classification_report(y_train, predicted_train,
    labels=y_train.cat.categories.tolist()))

metrics.confusion_matrix(y_train, predicted_train)


# for testing data

predicted_test = text_classifier.predict(X_test)

y_test = y_test.astype('category')
print(metrics.classification_report(y_test, predicted_test,
    labels=y_test.cat.categories.tolist()))

metrics.confusion_matrix(y_test, predicted_test)
text_classifier.predict_proba(['Dogs love us !'])
corpus=sentim_data['selected_text'].values
X=count_vec.fit_transform(corpus)
X.shape
tfidf=TfidfTransformer()
X=tfidf.fit_transform(X)
X.shape
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=sentim_data['sentiment'].values
y=le.fit_transform(y)
y.shape
from keras import models
from keras.layers import Dense
model=models.Sequential()
model.add(Dense(16,activation="relu",input_shape=(X.shape[1],)))
model.add(Dense(16,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
X_val=X[:5000]
X_train=X[5000:]

y_val=y[:5000]
y_train=y[5000:]
X_train.shape ,y_train.shape
hist=model.fit(X,y,batch_size=128,epochs=6,validation_data=(X_val,y_val))
result=hist.history
plt.plot(result['val_accuracy'],label='val_accuracy')
plt.plot(result['accuracy'],label='Train_accuracy')
plt.legend()
plt.show()
plt.plot(result['val_loss'],label='val_loss')
plt.plot(result['loss'],label='Train_loss')
plt.legend()
plt.show()
model.evaluate(X_val,y_val)
test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test.shape
test.head()
test['selected_text']=test['text'].apply(cleanTxt)
test.head()
X_test=test['selected_text']
X_test.shape
X_test=count_vec.transform(X_test)
X_test=tfidf.transform(X_test)
X_test.shape
print(X_test[0])
y_pred=model.predict(X_test)
# y_pred[y_pred>0.5]=1
# y_pred=y_pred.astype('int')
y_pred.shape
ids=test['textID']
ids.shape
my_submission = pd.DataFrame({'textID': test.textID.values, 'selected_text': test.selected_text})
my_submission


import csv
my_submission.to_csv('submission.csv', index=False)

