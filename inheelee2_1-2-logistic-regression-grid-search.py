# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import json
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
data = pd.read_csv("/kaggle/input/dfc615e/en_data.csv", sep=",", encoding="ms949")
sample = pd.read_csv("/kaggle/input/dfc615e/en_sample.csv", sep=",", encoding="ms949")
dev_file = open('/kaggle/input/friends3/friends_test3.json', encoding="utf-8")
dev_data = json.load(dev_file)
train_file = open('/kaggle/input/friends3/friends_train3.json', encoding="utf-8")
train_data = json.load(train_file)
test_file = open('/kaggle/input/friends3/friends_dev3.json', encoding="utf-8")
test_data = json.load(test_file)
data.head()
df_dev = pd.DataFrame(columns=['annotation', 'emotion', 'speaker', 'utterance'])
df_train = pd.DataFrame(columns=['annotation', 'emotion', 'speaker', 'utterance'])
df_test = pd.DataFrame(columns=['annotation', 'emotion', 'speaker', 'utterance'])

for i in range(len(dev_data)):
    df_dev = pd.concat([df_dev, pd.DataFrame(dev_data[i])])

for i in range(len(train_data)):
    df_train = pd.concat([df_train, pd.DataFrame(train_data[i])])

for i in range(len(test_data)):
    df_test = pd.concat([df_test, pd.DataFrame(test_data[i])])

df_dev = df_dev.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_data = data.sort_values("id").reset_index(drop=True)
len(df_dev), len(df_train), len(df_test), len(df_data)
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 랭커스터 스태머의 사용
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

# Lemmatization 음소표기법
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))
df_dev['words'] = ''
df_train['words'] = ''
df_test['words'] = ''
df_data['words'] = ''

len(df_dev), len(df_train), len(df_test), len(df_data)
def comment_to_words(data):
    # 1. 영어가 아닌 문자는 공백으로 변환
    data = re.sub('[^a-zA-Z]', ' ', data)
    
    # 2. 소문자로 변환
    lowerdata = data.lower()
    
    # 3. 문자열로 변환
    words = lowerdata.split()
    
    # 4. 불용어 제거
    words = [w for w in words if not w in stops]
    
    # 5. 어간추출
    stemming_words = [stemmer.stem(w) for w in words]
    
    # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    words = ' '.join(stemming_words)
    return words
# 개발 데이터셋 전처리
for i in range(0, len(df_dev)):
    df_dev.loc[i, 'words'] = comment_to_words(df_dev.loc[i, 'utterance'])

# 훈련 데이터셋 전처리
for i in range(0, len(df_train)):
    df_train.loc[i, 'words'] = comment_to_words(df_train.loc[i, 'utterance'])    

# 테스트 데이터셋 전처리
for i in range(0, len(df_test)):
    df_test.loc[i, 'words'] = comment_to_words(df_test.loc[i, 'utterance'])    
    
# 테스트 데이터셋 전처리
for i in range(0, len(df_test)):
    df_data.loc[i, 'words'] = comment_to_words(df_data.loc[i, 'utterance'])    
y_info = [['neutral', 0],
          ['surprise', 1],
          ['non-neutral', 2],
          ['joy', 3],
          ['sadness', 4],
          ['anger', 5],
          ['disgust', 6]]

y_info = [['neutral', 'neutral'],
          ['surprise', 'surprise'],
          ['non-neutral', 'non-neutral'],
          ['joy', 'joy'],
          ['sadness', 'sadness'],
          ['anger', 'anger'],
          ['disgust', 'disgust']]

df_y = pd.DataFrame(y_info, columns=['emotion', 'Y'])
df_dev = pd.merge(df_dev, df_y, on=['emotion'])
df_train = pd.merge(df_train, df_y, on=['emotion'])
df_test = pd.merge(df_test, df_y, on=['emotion'])
df_train.head()
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import backend as K
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__max_df': (0.25, 0.5),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'clf__C': (0.1, 1, 10),
}
len(df_train['words']), len(df_train['Y']), len(df_test['words']), len(df_test['Y']), 
X_train,  _, y_train, _ = train_test_split(df_train['words'], df_train['Y'], train_size=0.9999)
X_data = df_data['words']
X_train[:5]
X_test, _, y_test, _ = train_test_split(df_test['words'], df_test['Y'], train_size=0.9999)
len(X_train), len(y_train), len(X_test), len(y_test)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print('best score: %0.3f' % grid_search.best_score_)
print('best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t %s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
print('Confusion Matrix:', confusion_matrix(y_test, predictions))
print('*************************************************************')
print('Classification Report:')
print(classification_report(y_test, predictions,target_names=df_y['emotion']))
results = grid_search.predict(X_data)
sample['Predicted'] = results
len(sample), len(results)
sample['Predicted'].value_counts()
sample[['Id', 'Predicted']].to_csv("/kaggle/working/output2_logisticRegrsion.csv", sep=",", encoding="ms949", index=False)
sample.tail()
