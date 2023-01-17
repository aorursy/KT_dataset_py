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
%matplotlib inline
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
train=pd.read_csv('/kaggle/input/bnbr-kenya/Train_health.csv')
test=pd.read_csv('/kaggle/input/bnbr-kenya/Test_health.csv')
sub = pd.read_csv('/kaggle/input/sample-submission/ss_health.csv')
print('Train shape:',train.shape,'and number of null values are:',train.isnull().sum())
train.head()
print('Test shape:',test.shape,'and number of null values are:',test.isnull().sum())
test.head()
train['label'].value_counts()
cate = train['label']
cate.head()
from sklearn.preprocessing import OneHotEncoder
cate_one = pd.get_dummies(data=cate)
cate_one.head()
train_df =train.join(cate_one)
train_df.head()
train_df = train_df.drop(['label'], axis=1)
train_df.head()
train_df.shape
!pip install --upgrade transformers
!pip install simpletransformers
# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
train_df.shape, test.shape, sub.shape
label = ['Depression','Alcohol','Suicide','Drugs']
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['Depression','Alcohol','Suicide','Drugs']

train_df.fillna(' ')
test.fillna(' ')

#train = pd.read_csv('../input/train.csv').fillna(' ')
#test = pd.read_csv('../input/test.csv').fillna(' ')

train_df_text = train['text']
test_text = test['text']
all_text = pd.concat([train_df_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features= 1050)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_df_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features= 1050 )
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_df_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'ID': test['ID']})
for class_name in class_names:
    train_target = train_df[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)