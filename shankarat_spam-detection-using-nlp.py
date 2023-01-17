import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
sms = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

sms.head()
sms.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

sms.rename(columns={'v1':'label','v2':'message'},inplace=True)
print ('Shape = >',sms.shape)
print ('ham and spam counts','\n',sms.label.value_counts())
print ('spam ratio = ', round(len(sms[sms['label']=='spam']) / len(sms.label),2)*100,'%')

print ('ham ratio  = ', round(len(sms[sms['label']=='ham']) / len(sms.label),2)*100,'%')
sms['length'] = sms.message.str.len()

sms.head(2)
sms['label'].replace({'ham':0,'spam':1},inplace=True)
sms['message'] = sms['message'].str.lower()
# Replace email addresses with 'email'

sms['message'] = sms['message'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',

                                 'emailaddress')



# Replace URLs with 'webaddress'

sms['message'] = sms['message'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',

                                  'webaddress')



# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)

sms['message'] = sms['message'].str.replace(r'£|\$', 'moneysymb')

    

# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'

sms['message'] = sms['message'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',

                                  'phonenumber')



    

# Replace numbers with 'numbr'

sms['message'] = sms['message'].str.replace(r'\d+(\.\d+)?', 'numbr')
# Remove punctuation

sms['message'] = sms['message'].str.replace(r'[^\w\d\s]', ' ')



# Replace whitespace between terms with a single space

sms['message'] = sms['message'].str.replace(r'\s+', ' ')



# Remove leading and trailing whitespace

sms['message'] = sms['message'].str.replace(r'^\s+|\s+?$', '')
import string

import nltk

from nltk.corpus import  stopwords



stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])



sms['message'] = sms['message'].apply(lambda x: ' '.join(

    term for term in x.split() if term not in stop_words))
sms['clean_length'] = sms.message.str.len()

sms.head()
print ('Origian Length', sms.length.sum())

print ('Clean Length', sms.clean_length.sum())
f,ax = plt.subplots(1,2,figsize = (10,5))



sns.distplot(sms[sms['label']==1]['length'],bins=20,ax=ax[0],label='Spam messages distribution',color='r')

ax[0].set_xlabel('Spam sms length')

ax[0].legend()



sns.distplot(sms[sms['label']==0]['length'],bins=20,ax=ax[1],label='ham messages distribution')

ax[1].set_xlabel('ham sms length')

ax[1].legend()



plt.show()
f,ax = plt.subplots(1,2,figsize = (10,5))



sns.distplot(sms[sms['label']==1]['clean_length'],bins=20,ax=ax[0],label='Spam messages distribution',color='r')

ax[0].set_xlabel('Spam sms length')

ax[0].legend()



sns.distplot(sms[sms['label']==0]['clean_length'],bins=20,ax=ax[1],label='ham messages distribution')

ax[1].set_xlabel('ham sms length')

ax[1].legend()



plt.show()
#Getting sense of loud words in spam

from wordcloud import WordCloud





spams = sms['message'][sms['label']==1]

spam_cloud = WordCloud(width=600,height=400,background_color='white',max_words=50).generate(' '.join(spams))

plt.figure(figsize=(10,8),facecolor='b')

plt.imshow(spam_cloud)

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#Getting sense of loud words in ham 



hams = sms['message'][sms['label']==0]

spam_cloud = WordCloud(width=600,height=400,background_color='white',max_words=50).generate(' '.join(hams))

plt.figure(figsize=(10,8),facecolor='k')

plt.imshow(spam_cloud)

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



tf_vec = TfidfVectorizer()

naive = MultinomialNB()



features = tf_vec.fit_transform(sms['message'])



X = features

y = sms['label']
X_train,x_test,Y_train,y_test = train_test_split(X,y,random_state=42)

naive.fit(X_train,Y_train)

y_pred= naive.predict(x_test)



print ('Final score = > ', accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_test,y_pred)



ax=plt.subplot()

sns.heatmap(conf_mat,annot=True,ax=ax,linewidths=5,linecolor='r',center=0)

ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels')

ax.set_title('Confusion matrix')

ax.xaxis.set_ticklabels(['ham','spam'])

ax.yaxis.set_ticklabels(['ham','spam'])

plt.show()