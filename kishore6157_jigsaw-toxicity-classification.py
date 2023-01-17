# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import zipfile

from zipfile import ZipFile

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
tfidf = TfidfVectorizer()

X=tfidf.fit_transform(train['comment_text'])

Y=np.where(train['target']>=0.5,1,0)
test_X = tfidf.transform(test["comment_text"])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.333, random_state=42)
lr=LogisticRegression(random_state=42, solver='sag',n_jobs=-1)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(x_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print(classification_report(y_test, y_pred))
predictions = lr.predict_proba(test_X)[:,1:]

predictions=pd.DataFrame(predictions)
ids=test['id']

ids=pd.DataFrame(ids)
submission=pd.concat([ids,predictions],axis=1)

submission['prediction']=submission.iloc[:,1:]

submission=submission.drop([0],axis=1)

submission.to_csv("submission.csv",index=False)