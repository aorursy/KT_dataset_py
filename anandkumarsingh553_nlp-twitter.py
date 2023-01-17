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
train_df=pd.read_csv('../input/nlp-getting-started/train.csv')

test_df=pd.read_csv('../input/nlp-getting-started/test.csv')
print(train_df.shape)

train_df.head()
print(test_df.shape)

test_df.head()
train_df.drop(columns=['id',"keyword",'location'],inplace=True,axis=0)

train_df.head()
test_df.drop(columns=['id','keyword','location'],inplace=True)

test_df.head()
from sklearn.model_selection import train_test_split

x=train_df['text']

y=train_df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()

x_vector=vectorizer.fit_transform(x_train)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=500)

lr.fit(x_vector,y_train)

pred_y=lr.predict(vectorizer.transform(x_test))
from sklearn.metrics import accuracy_score

score_linear=accuracy_score(y_test,pred_y)

score_linear
from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB(alpha=1)

nb.fit(x_vector,y_train)

y_pred=nb.predict(vectorizer.transform(x_test))

score_nb=accuracy_score(y_test,y_pred)

score_nb
from sklearn.svm import SVC

svm=SVC(C=10,gamma=0.01,kernel='rbf')

svm.fit(x_vector,y_train)

y_pred=svm.predict(vectorizer.transform(x_test))

score_svm=accuracy_score(y_test,y_pred)

score_svm
prediction=lr.predict(vectorizer.transform(test_df['text']))

submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission['target']=prediction

submission
submission.to_csv('submission.csv',index=False)