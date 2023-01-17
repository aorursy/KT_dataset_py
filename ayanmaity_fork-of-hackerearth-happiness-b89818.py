# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train_hp.csv')

train_df.head()
train_df = train_df.drop(['User_ID'],axis=1)
train_df.head()
train_df.Browser_Used.unique()
train_df['Browser_Used'][train_df['Browser_Used']=='InternetExplorer'] = 'Internet Explorer'

train_df['Browser_Used'][train_df['Browser_Used']=='IE'] = 'Internet Explorer'

train_df['Browser_Used'][train_df['Browser_Used']=='Google Chrome'] = 'Chrome'

train_df['Browser_Used'][train_df['Browser_Used']=='Mozilla'] = 'Firefox'

train_df['Browser_Used'][train_df['Browser_Used']=='Mozilla Firefox'] = 'Firefox'
train_df.Browser_Used.unique()
train_df.Device_Used.unique()
train_df.head()
ip_addresses = train_df.Browser_Used.unique()

ip_dict1 = dict(zip(ip_addresses, range(len(ip_addresses))))

train_df = train_df.replace(ip_dict1)



ip_addresses = train_df.Device_Used.unique()

ip_dict2 = dict(zip(ip_addresses, range(len(ip_addresses))))

train_df = train_df.replace(ip_dict2)
train_df.head()
train_df['Browser_Used'] = train_df['Browser_Used']+1

train_df['Device_Used'] = train_df['Device_Used']+1

train_df.head()
ip_addresses = train_df.Is_Response.unique()

ip_dict3 = dict(zip(ip_addresses, range(len(ip_addresses))))

train_df = train_df.replace(ip_dict3)
train_df.head()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS

c_vector = TfidfVectorizer(stop_words = stop_words,min_df=.00001,lowercase=1)
X_train_counts = c_vector.fit_transform(train_df['Description'].values)
X_train_counts
from sklearn.naive_bayes import MultinomialNB

clf1 = MultinomialNB()

target = train_df['Is_Response']

clf1.fit(X_train_counts,target)
clf1.score(X_train_counts,target)
X_vec = train_df.as_matrix(['Browser_Used','Device_Used'])
X_vec.shape
##from sklearn.neural_network import MLPClassifier 

##clf2 = MLPClassifier(max_iter=250,solver='sgd',hidden_layer_sizes=(10,10),verbose=1,random_state=10)

clf2 = MultinomialNB()

target = train_df['Is_Response']

clf2.fit(X_vec,target)
clf2.score(X_vec,target)
muti_proba = clf1.predict_proba(X_train_counts)

nn_proba = clf2.predict_proba(X_vec)

X_final = np.hstack((muti_proba,nn_proba))
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='sgd',max_iter=250,random_state=20,verbose=1,hidden_layer_sizes=(10,10))

clf.fit(X_final,target)
clf.score(X_final,target)
from sklearn import svm

clf_svm = svm.SVC(verbose=1,random_state=30,tol=.0001)

clf_svm.fit(X_final,target)
clf_svm.score(X_final,target)
test_df = pd.read_csv('../input/test_hp.csv')
test_df.head()
X_test_counts = c_vector.transform(test_df['Description'].values)
test_df.Browser_Used.unique()
test_df['Browser_Used'][test_df['Browser_Used']=='InternetExplorer'] = 'Internet Explorer'

test_df['Browser_Used'][test_df['Browser_Used']=='IE'] = 'Internet Explorer'

test_df['Browser_Used'][test_df['Browser_Used']=='Google Chrome'] = 'Chrome'

test_df['Browser_Used'][test_df['Browser_Used']=='Mozilla'] = 'Firefox'

test_df['Browser_Used'][test_df['Browser_Used']=='Mozilla Firefox'] = 'Firefox'

test_df.Browser_Used.unique()
ip_dict2
test_df = test_df.replace(ip_dict1)

test_df = test_df.replace(ip_dict2)
test_df.head()
test_df['Browser_Used'] = test_df['Browser_Used']+1

test_df['Device_Used'] = test_df['Device_Used']+1

test_df.head()
X_vec_t = test_df.as_matrix(['Browser_Used','Device_Used'])

multi_proba_t = clf1.predict_proba(X_test_counts)

gauss_proba_t = clf2.predict_proba(X_vec_t)

X_final_t = np.hstack((multi_proba_t,gauss_proba_t))
Y_test = clf.predict(X_final_t)

Y_test
Y_test.sum()
Y_test.shape
sub = pd.DataFrame()

sub['User_ID'] = test_df['User_ID']

sub['Is_Response'] = Y_test

sub
dict = {1:'happy',0:'not_happy'}

sub = sub.replace(dict)

sub
sub.to_csv('output.csv')