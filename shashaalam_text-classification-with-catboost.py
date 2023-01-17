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
from catboost import Pool, CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import os
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
tweet.head()
tweet.keyword.value_counts()
tweet.location.value_counts()
x=tweet.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')

cat_features = ['keyword','location']
text_features = ['text']

tweet.isna().sum()
tweet['keyword'].value_counts()
tweet['keyword'] = np.where(tweet['keyword'].isna()==True,'UNK',tweet['keyword'])
tweet['location'] = np.where(tweet['location'].isna()==True,'UNK',tweet['location'])
col = ['target','id']
X = tweet.drop(col,axis=1)
y = tweet['target']
X_train, X_test, y_train, y_test = train_test_split(X,y.values,test_size=0.15)
train_pool = Pool(
    X_train, 
    y_train, 
    cat_features=cat_features, 
    text_features=text_features, 
    feature_names=list(X_train)
)
valid_pool = Pool(
    X_test, 
    y_test,
    cat_features=cat_features, 
    text_features=text_features, 
    feature_names=list(X_train)
)

catboost_params = {
    'iterations': 10000,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    'task_type': 'GPU',
    'early_stopping_rounds': 5000,
    'use_best_model': True,
    'verbose': 5000
}

model = CatBoostClassifier(**catboost_params)
model.fit(train_pool, eval_set=valid_pool)


from sklearn.metrics import classification_report,accuracy_score
pred = model.predict(X_test)
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))
pred
test1 = test.drop('id',axis=1)
test1.isna().sum()
test1['keyword'] = np.where(test1['keyword'].isna()==True,'UNK',test1['keyword'])
test1['location'] = np.where(test1['location'].isna()==True,'UNK',test1['location'])
pred1 = model.predict(test1)
sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
pred1.shape
sub.columns
sub.target.value_counts()

sub['target'] = pred1

sub.head()
sub.to_csv('submission.csv', index=False)
