# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

def warn(*args, **kwargs):

    pass

warnings.warn = warn



from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from scipy.sparse import csr_matrix, vstack, hstack

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train_hp.csv')

test_df = pd.read_csv('../input/test_hp.csv')

print(train_df.shape)

train_df.head()
train_df.drop(['User_ID'],axis=1, inplace=True)
train_df.Browser_Used.unique()
train_df['Browser_Used'][train_df['Browser_Used']=='InternetExplorer'] = 'IE'

train_df['Browser_Used'][train_df['Browser_Used']=='Internet Explorer'] = 'IE'

train_df['Browser_Used'][train_df['Browser_Used']=='Google Chrome'] = 'Chrome'

train_df['Browser_Used'][train_df['Browser_Used']=='Mozilla'] = 'Firefox'

train_df['Browser_Used'][train_df['Browser_Used']=='Mozilla Firefox'] = 'Firefox'



test_df['Browser_Used'][test_df['Browser_Used']=='InternetExplorer'] = 'IE'

test_df['Browser_Used'][test_df['Browser_Used']=='Internet Explorer'] = 'IE'

test_df['Browser_Used'][test_df['Browser_Used']=='Google Chrome'] = 'Chrome'

test_df['Browser_Used'][test_df['Browser_Used']=='Mozilla'] = 'Firefox'

test_df['Browser_Used'][test_df['Browser_Used']=='Mozilla Firefox'] = 'Firefox'
train_df.Device_Used.unique()
test_df.Device_Used.unique()
lable = LabelEncoder()

train_df['Browser_Used'] = lable.fit_transform(train_df['Browser_Used'])

test_df['Browser_Used'] = lable.transform(test_df['Browser_Used'])

list(lable.classes_)
list(lable.inverse_transform([0,1,2,3,4,5]))
lable2 = LabelEncoder()

train_df['Device_Used'] = lable2.fit_transform(train_df['Device_Used'])

test_df['Device_Used'] = lable2.transform(test_df['Device_Used'])
list(lable2.classes_)
list(lable2.inverse_transform([0,1,2]))
lable3 = LabelEncoder()

train_df['Is_Response'] = lable3.fit_transform(train_df['Is_Response'])

list(lable3.classes_)
list(lable3.inverse_transform([0,1]))
y = train_df['Is_Response']
from sklearn.feature_extraction.text import CountVectorizer

c_vector = CountVectorizer()

X_train_counts = c_vector.fit_transform(train_df['Description'].values)

X_test_counts = c_vector.transform(test_df['Description'].values)
X_train_vec = train_df.as_matrix(['Browser_Used','Device_Used'])

X_test_vec = test_df.as_matrix(['Browser_Used','Device_Used'])
X_train_vec_sparce = csr_matrix(X_train_vec, dtype = int)

X = hstack((X_train_counts, X_train_vec_sparce))



X_test_vec_sparce = csr_matrix(X_test_vec, dtype = int)

X_final_test = hstack((X_test_counts, X_test_vec_sparce))
X_train, X_val, y_train, y_val = train_test_split(X, y , test_size = 0.33, random_state = 42)
modelNB = MultinomialNB()

modelNB.fit(X_train, y_train).score(X_val, y_val)
confusion_matrix(y_true = y_train, y_pred=modelNB.predict(X_train))
confusion_matrix(y_true = y_val, y_pred=modelNB.predict(X_val))
modelLR = LogisticRegressionCV(Cs=20, cv = 3)

modelLR.fit(X_train, y_train).score(X_val, y_val)
confusion_matrix(y_true = y_train, y_pred=modelLR.predict(X_train))
confusion_matrix(y_true = y_val, y_pred=modelLR.predict(X_val))
y_pred = modelLR.predict(X_final_test)

y_pred
sub = pd.DataFrame()

sub['User_ID'] = test_df['User_ID']

sub['Is_Response'] = y_pred

dict = {1:'happy',0:'not_happy'}

sub = sub.replace(dict)

sub