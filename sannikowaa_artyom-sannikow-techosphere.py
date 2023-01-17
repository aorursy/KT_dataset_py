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
import numpy as np 

import pandas as pd

import time

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv('../input/train.csv')

train = train.dropna(how="any").reset_index(drop=True)



test = pd.read_csv('../input/test.csv')

test = test.dropna(how="any").reset_index(drop=True)
len(test)
train.head()
train['question1'] = train['question1'].fillna('')

train['question2'] = train['question2'].fillna('')



test['question1'] = test['question1'].fillna('')

test['question2'] = test['question2'].fillna('')
def to_lowercase(pd_series):

    return pd_series.str.lower()
train['question1'] = to_lowercase(train['question1'])

train['question2'] = to_lowercase(train['question2'])



test['question1'] = to_lowercase(test['question1'])

test['question2'] = to_lowercase(test['question2'])
import nltk

from nltk.corpus import stopwords



stop_words = set(stopwords.words("english"))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
print(stop_words)
def rm_stop_words(pd_series, stop_words):

    return pd_series.apply(lambda x: "".join([item+" " for item in x.split(" ") 

                                       if item not in stop_words])[0:-2])
a = 0.165 / 0.37

b = (1 - 0.165) / (1 - 0.37)
train["question1"] = rm_stop_words(train["question1"],stop_words)

train['question2']= rm_stop_words(train['question2'],stop_words)

test['question1']= rm_stop_words(test['question1'],stop_words)

test['question2']= rm_stop_words(test['question2'],stop_words)
train.head()
from sklearn.feature_extraction.text import CountVectorizer
len(test)
Word_Extractor = CountVectorizer(analyzer='char', ngram_range=(1,2), binary=True, lowercase=True)

Word_Extractor.fit(pd.concat((train.ix[:,'question1'],train.ix[:,'question2'])).unique())
question_1 = Word_Extractor.transform(train.ix[:,'question1'])

question_2 = Word_Extractor.transform(train.ix[:,'question2'])
X = -(question_1 != question_2).astype(int)

y = np.array(train.ix[:,'is_duplicate'])
len(X)
len(train)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 111, 20), 'max_depth':range(10,21,5)}

rf = RandomForestClassifier()

model = GridSearchCV(rf,parameters)

model.fit(X[:20000,:],y[:20000])
print(model.best_params_)
model = RandomForestClassifier(n_estimators = 50, max_depth = 15, class_weight={1: a, 0: b})

model.fit(X,y)
test_question_1 = Word_Extractor.transform(test.ix[:,'question1'])

test_question_2 = Word_Extractor.transform(test.ix[:,'question2'])

X_test = -(test_question_1 != test_question_2).astype(int)
size(X_test)
testPredictions = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame()

submission['test_id'] = test['test_id']

submission['is_duplicate'] = testPredictions

submission.to_csv('submission.csv', index=False)