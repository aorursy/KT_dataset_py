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
sms = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

sms.head()
sms = sms.rename(columns = {'v1' : 'label', 'v2':'message'})

sms.head()
sms.label.value_counts()
sms.isnull().any()
#Converting label to numeric variable

sms['label'] = sms.label.map({'ham':0,'spam':1})

sms.head()
#splitting data into train and test

x = sms.message

y = sms.label



from sklearn.model_selection import train_test_split



train_x, test_x, train_y, test_y = train_test_split(x,y)

print(train_x.shape)

print(test_x.shape)

print(train_y.shape)

print(test_y.shape)
from sklearn.feature_extraction.text import CountVectorizer



#instantiate the vector

vect = CountVectorizer(max_df=0.90,min_df = 0.001,stop_words='english',strip_accents = 'unicode')
#learn training data vocabulary and convert it into document term matrix

train_x_dtm = vect.fit_transform(train_x)

train_x_dtm
#Now transform testing data(using fitted vocabulary) into jdocument term matrix

test_x_dtm = vect.transform(test_x)

test_x_dtm
#Store Token names

train_x_token = vect.get_feature_names()

print(train_x_token[-50:])
#view train_x_dtm matrix

train_x_dtm.toarray()
train_x_token_count = np.sum(train_x_dtm.toarray(),axis=0)

train_x_token_count
#Now create a dataframe of tokens with their counts



pd.DataFrame({'token': train_x_token,'count':train_x_token_count})
#Create seperate data frames for ham and spam

sms_ham = sms[sms.label==0]

sms_spam = sms[sms.label==1]
#learn the vocabulary of all messages

vect.fit(sms.message)

tokens = vect.get_feature_names()
#create document term matrix for ham and spam

ham_dtm = vect.transform(sms_ham.message)

spam_dtm = vect.transform(sms_spam.message)
#Count the token of ham messages 

ham_token_count = np.sum(ham_dtm.toarray(), axis=0)

spam_token_count = np.sum(spam_dtm.toarray(),axis=0)
#create dataframe of token with seperate ham and spam counts

df = pd.DataFrame({'token': tokens,'ham_count': ham_token_count, 'spam_count': spam_token_count})

df
#lets add 1 to ham and spam count to avoid dividing by zero

df['ham_count'] = df['ham_count'] + 1

df['spam_count'] = df['spam_count'] + 1
df['spam_ratio'] = df['spam_count']/df['ham_count']

df.sort_values('spam_ratio')
#The multinomial Naive Bayes classifier is suitable for classification with discrete features



from sklearn.naive_bayes import MultinomialNB



nb = MultinomialNB()

nb.fit(train_x_dtm,train_y)
# make class predictions for test_x_dtm

pred_y = nb.predict(test_x_dtm)
#calculating accuracy

from sklearn import metrics



print(metrics.accuracy_score(test_y,pred_y))
#Confusion matrix



cm = metrics.confusion_matrix(test_y, pred_y)

cm
#ROC score

print(metrics.roc_auc_score(test_y, pred_y))
# print message text for the false positives

test_x[test_y < pred_y]
# print message text for the false negatives

test_x[test_y > pred_y]