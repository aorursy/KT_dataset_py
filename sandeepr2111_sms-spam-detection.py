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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
sms_df=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
sms_df.head()
sms_df.shape
sms_df.isnull().sum()
sms_df.isnull().mean()*100
sms_df.dropna(how='any',axis=1,inplace=True)
sms_df.head()
sms_df.columns=['Tag','Message']
sms_df.describe()
sms_df.groupby('Tag').describe()
sms_df.info()
sms_df['Tag'].unique()
sms_df['Tag']=np.where(sms_df['Tag']=='spam',1,0)
sms_df.head()
sms_df.describe()
sms_df['Tag'].mean()*100
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
X_train,X_test,y_train,y_test=train_test_split(sms_df['Message'],sms_df['Tag'],random_state=0)
# CountVectorizer

count=CountVectorizer().fit(X_train)

X_train_Count=count.transform(X_train)

X_test_count=count.transform(X_test)



#TfidfVectorizer



Tfid=TfidfVectorizer().fit(X_train)

X_train_Tfid=Tfid.transform(X_train)

X_test_Tfid=Tfid.transform(X_test)
clf_count=MultinomialNB(alpha=0.1)

clf_count.fit(X_train_Count,y_train)

pred=clf_count.predict(X_test_count)

print('ROC score by applying Countvectorizer:',roc_auc_score(y_test,pred))
clf_Tfid=MultinomialNB(alpha=0.1)

clf_Tfid.fit(X_train_Tfid,y_train)

pred=clf_Tfid.predict(X_test_Tfid)

print('ROC score by applying TfidfVectorizer:',roc_auc_score(y_test,pred))
feature_names=np.array(count.get_feature_names())

count_coefficients=clf_count.coef_[0].argsort()



print('Smallest 20  Count vectorizer coefficients:\n')

print(feature_names[count_coefficients[:20]])

print('\n\n')

print('Largest 20  Count vectorizer coefficients:\n')

print(feature_names[count_coefficients[-21:-1]])
feature_names=np.array(Tfid.get_feature_names())

Tfid_coefficients=clf_Tfid.coef_[0].argsort()



print('Smallest 20  Tfid vectorizer coefficients:\n')

print(feature_names[Tfid_coefficients[:20]])

print('\n\n')

print('Largest 20  Tfid vectorizer coefficients:\n')

print(feature_names[Tfid_coefficients[-21:-1]])
from sklearn.dummy import DummyClassifier



dummy=DummyClassifier(strategy='prior').fit(X_train_Count,y_train)
dummy_predict=dummy.predict(X_test_count)

roc_auc_score(y_test,dummy_predict)
X_train_Count.shape
sum1=X_train_Count.sum(axis=0)
len(count.get_feature_names())
sum1.shape
# Countvectorizer Features



data=[]



for col,features in enumerate(count.get_feature_names()):

    data.append([features,sum1[0,col]])

    

feature_data=pd.DataFrame(data,columns=['Feature','Score'])

feature_data.sort_values(by='Score',inplace=True)



print('20 features with lowest score')



print(feature_data.head(20).sort_values(by='Score',ascending=False))



print('20 features with highest score')

print(feature_data.tail(20).sort_values(by='Score',ascending=False))
#Tfidf



data1=[]



Tf_sum=X_train_Tfid.sum(axis=0)



for col,features in enumerate(Tfid.get_feature_names()):

    data1.append([features,Tf_sum[0,col]])

    

feature_data=pd.DataFrame(data1,columns=['Feature','Score'])

feature_data.sort_values(by='Score',inplace=True)



print('20 features with lowest score')

print('\n')



print(feature_data.head(20).sort_values(by='Score',ascending=False))

print('\n\n')

print('20 features with highest score')

print('\n')

print(feature_data.tail(20).sort_values(by='Score',ascending=False))
sms_df['Message_length']=sms_df['Message'].apply(lambda x:len(x))

sms_df.head()
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



example_sentence='This is an example showing of stop word filteration.'

words=word_tokenize(example_sentence)

print('Before applying Stopwords:\n\n{}'.format(words))

stop_words=set(stopwords.words('english'))



w=[]

for i in words:

    if i not in stop_words:

        w.append(i)

print('\n\n')

print('After applying Stopwords:\n\n{}'.format(w))
sms_df['Message_stop']=sms_df['Message'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words ]))
sms_df['Message_length_stop']=sms_df['Message_stop'].apply(lambda x:len(x))

sms_df.head()
sms_df.drop(['Message','Message_length'],axis=1,inplace=True)
sms_df.head()
plt.figure(figsize=(7,5))

sns.countplot(x='Tag',data=sms_df)

plt.title('Total Count of Spam and ham message\n 1=Spam and 0=ham',size=15)
sms_df['Number_count']=sms_df['Message_stop'].apply(lambda x:len(''.join([n for n in x if n.isdigit()])))
sms_df.head(6)
sms_df['Message_stop'][5] # 6th message contains exactly 4 digits
sms_df[sms_df['Tag']==1].describe()
sms_df[sms_df['Tag']==0].describe()
fig,ax=plt.subplots(1,2,figsize=(15,5))



msg_length_spam=sms_df.loc[sms_df['Tag']==1,'Message_length_stop']

msg_length_ham=sms_df.loc[sms_df['Tag']==0,'Message_length_stop']



sns.distplot(msg_length_spam,ax=ax[0],color='r')

ax[0].set_title('Distribution of Message length of Spam',fontsize=14)





sns.distplot(msg_length_ham,ax=ax[1],color='b')

ax[1].set_title('Distribution of Message length of ham',fontsize=14)

plt.show()
fig,ax=plt.subplots(1,2,figsize=(15,5))



number_count_spam=sms_df.loc[sms_df['Tag']==1,'Number_count']

number_count_ham=sms_df.loc[sms_df['Tag']==0,'Number_count']



sns.distplot(number_count_spam,ax=ax[0],color='r')

ax[0].set_title('Distribution of numbers length of Spam',fontsize=14)







# try:

#     sns.distplot(number_count_ham,ax=ax[1],color='b')

# except RuntimeError as re:

#     if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):

#         sns.distplot(number_count_ham,ax=ax[1],color='b', kde_kws={'bw': 0.1})

#         ax[1].set_title('Distribution of numbers length of ham',fontsize=14)

#     else:

#         raise re



sns.distplot(number_count_ham,ax=ax[1],color='b', kde_kws={'bw': 0.1})

ax[1].set_title('Distribution of numbers length of ham',fontsize=14)

plt.show()
X=sms_df['Message_stop']

y=sms_df['Tag']



X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
# # CountVectorizer

count=CountVectorizer(min_df=5,ngram_range=[3,6],analyzer='char').fit(X_train)

X_train_Count=count.transform(X_train)

X_test_Count=count.transform(X_test)



#TfidfVectorizer



Tfid=TfidfVectorizer(min_df=5,ngram_range=[3,6],analyzer='char').fit(X_train)

X_train_Tfid=Tfid.transform(X_train)

X_test_Tfid=Tfid.transform(X_test)
clf=MultinomialNB(alpha=0.1)

clf.fit(X_train_Count,y_train)

train_pred=clf.predict(X_train_Count)

print('ROC score of Training by applying Countvectorizer:',roc_auc_score(train_pred,y_train))

pred_count=clf.predict(X_test_Count)

print('ROC score of Testing by applying Countvectorizer:',roc_auc_score(y_test,pred_count))
# Predictions of MultinomialNB using CountVectorization



x=['do you have plans for weekend?, let us meet at our usual place',

  'Hii, you are our lucky customer, you have won 100000000 Rs, Please provide your account details we will transfer the amount',

  'Your account is freezed please provide your account details to unfreeze the account',

  'Hi, Pooja you have been selected for the First round of interview with Wipro, you need to visit our campus on Next Monday']

data=pd.Series(x)

trans=count.transform(data)

clf.predict(trans)
clf=MultinomialNB(alpha=0.1)

clf.fit(X_train_Tfid,y_train)

train_pred=clf.predict(X_train_Tfid)

print('ROC score of Training by applying TFidVectorizer:',roc_auc_score(train_pred,y_train))

pred_tfid=clf.predict(X_test_Tfid)

print('ROC score of Testing by applying TFidVectorizer:',roc_auc_score(y_test,pred_tfid))
# Predictions of MultinomialNB using TfidVectorization



x=['do you have plans for weekend?, let us meet at our usual place',

  'Hii, you are our lucky customer, you have won 100000000 Rs, Please provide your account details we will transfer the amount',

  'Your account is freezed please provide your account details to unfreeze the account',

  'Hi, Pooja you have been selected for the First round of interview with Wipro, you need to visit our campus on Next Monday for interview process']

data=pd.Series(x)

trans=Tfid.transform(data)

clf.predict(trans)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
print('confusion matrix ')

conf_count=confusion_matrix(y_test,pred_count)

print('Confusion matrix for classifier using Countvectorizer:\n\n{}'.format(conf_count))

conf_tfid=confusion_matrix(y_test,pred_tfid)

print('Confusion matrix for classifier using TFidVectorizer:\n\n{}'.format(conf_tfid))
print('classification report')

conf_report=classification_report(y_test,pred_count)

print('Classification report for classifier using Countvectorizer:\n\n{}'.format(conf_report))

conf_report=classification_report(y_test,pred_tfid)

print('Classification report for classifier using TFidVectorizer:\n\n{}'.format(conf_report))
from sklearn.linear_model import LogisticRegression



#Countvectorizer

log_clf=LogisticRegression(C=100)

log_clf.fit(X_train_Count,y_train)

train_pred=log_clf.predict(X_train_Count)

print(roc_auc_score(train_pred,y_train))

log_pre=log_clf.predict(X_test_Count)

print(roc_auc_score(y_test,log_pre))
#Tfid



log_clf=LogisticRegression(C=100)

log_clf.fit(X_train_Tfid,y_train)

train_pred=log_clf.predict(X_train_Tfid)

print(roc_auc_score(train_pred,y_train))

log_pre=log_clf.predict(X_test_Tfid)

print(roc_auc_score(y_test,log_pre))
from sklearn.tree import DecisionTreeClassifier



#Count

tree_clf=DecisionTreeClassifier(max_depth=3)

tree_clf.fit(X_train_Count,y_train)

train_pred=tree_clf.predict(X_train_Count)

print(roc_auc_score(train_pred,y_train))

test_pred=tree_clf.predict(X_test_Count)

print(roc_auc_score(test_pred,y_test))
#Tfid

tree_clf=DecisionTreeClassifier(max_depth=2)

tree_clf.fit(X_train_Tfid,y_train)

train_pred=tree_clf.predict(X_train_Tfid)

print(roc_auc_score(train_pred,y_train))

test_pred=tree_clf.predict(X_test_Count)

print(roc_auc_score(test_pred,y_test))