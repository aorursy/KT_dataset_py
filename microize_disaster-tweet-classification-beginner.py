import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from langdetect import detect_langs
from textblob import TextBlob

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
display(train.head(10))
display(test.sample(5))
submission
tr_rows,tr_cols=train.shape
te_rows,te_cols=test.shape
print("Rows in train data is",tr_rows,"\nRows in train data",te_rows)
train.id.nunique()
train=train.set_index('id')
train.sample(5)
train.isnull().sum()
train.info()
train.target.value_counts(normalize=True)
disaster_df=train[train.target==1]
non_disaster_df=train[train.target==0]
disaster_df_kw=set(disaster_df.keyword)
non_disaster_df_kw=set(non_disaster_df.keyword)
disaster_df
dis_kw=len(disaster_df_kw)
ndis_kw=len(non_disaster_df_kw)
print(dis_kw,ndis_kw)
common_kw=disaster_df_kw.intersection(non_disaster_df_kw)
common_kw_no=len(common_kw)
print("Number of common keywords are{}".format(common_kw_no))
print(common_kw)
disaster_tweet=''
for x in disaster_df.text:
    disaster_tweet=disaster_tweet+str(x)
disaster_tweet_wc=WordCloud(background_color='white',stopwords=STOPWORDS).generate(disaster_tweet)
plt.imshow(disaster_tweet_wc)
plt.axis('off')
plt.show()
non_disaster_tweet=''
for x in non_disaster_df.text:
    non_disaster_tweet=non_disaster_tweet+str(x)
non_disaster_tweet_wc=WordCloud(background_color='white').generate(non_disaster_tweet)
plt.imshow(non_disaster_tweet_wc)
plt.axis('off')
plt.show()
X_train,X_test,y_train,y_test=train_test_split(train.text,train.target,test_size=0.20,random_state=3)
x_sh=X_train.shape[0]
y_sh=X_test.shape[0]
print('Number of data in training set:',x_sh,'Number of data in CV set:',y_sh)
count_vect=CountVectorizer()
c_X_train=count_vect.fit_transform(X_train)
c_X_test=count_vect.transform(X_test)
mult_nb=MultinomialNB()
mult_nb.fit(c_X_train,y_train)
y_predict=mult_nb.predict(c_X_test)
acc_score=accuracy_score(y_test,y_predict)
conf_score=confusion_matrix(y_test,y_predict)
print('Accuracy Score:',acc_score,'\nconfusion Matrix\n',conf_score)
c_X_t=count_vect.transform(test['text'])
y_predict1=mult_nb.predict(c_X_t)
test['target']=y_predict1
test['target'].to_csv('submission.csv')