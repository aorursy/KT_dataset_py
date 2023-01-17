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
df_last = pd.read_csv("/kaggle/input/dfc615e/en_data.csv", sep=",", encoding="ms949")
sample = pd.read_csv("/kaggle/input/dfc615e/en_sample.csv", sep=",", encoding="ms949")
dev_file = open('/kaggle/input/friends2/friends_test2.json', encoding="utf-8")
dev_data = json.load(dev_file)
train_file = open('/kaggle/input/friends2/friends_train2.json', encoding="utf-8")
train_data = json.load(train_file)
test_file = open('/kaggle/input/friends2/friends_dev2.json', encoding="utf-8")
test_data = json.load(test_file)
df_last.head()
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
df_last['words'] = ''

len(df_dev), len(df_train), len(df_test), len(df_last)
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

# LAST 데이터셋 전처리
for i in range(0, len(df_test)):
    df_last.loc[i, 'words'] = comment_to_words(df_last.loc[i, 'utterance'])    
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
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tk = Tokenizer(df_train['words'], lower=True, split=" ")
tk.fit_on_texts(df_train['words'])

# 튜토리얼과 다르게 파라메터 값을 수정
# 파라메터 값만 수정해도 캐글 스코어 차이가 많이 남
vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 2, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 3),
                             max_features = 20000
                            )
from multiprocessing import Pool
import numpy as np

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라메터를 꺼냄
    workers = kwargs.pop('workers')
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    pool = Pool(processes=workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    # 작업 결과를 합쳐서 반환
    return pd.concat(list(result))

pipeline = Pipeline([
    ('vect', vectorizer),
])
#train_data_features = pipeline.fit_transform(df_train['words'].values)
train_data_features = pipeline.fit_transform(df_train['utterance'].values)
train_data_features.shape
vocab = vectorizer.get_feature_names()

# 벡터화 된 피처를 확인해 봄
dist = np.sum(train_data_features, axis=0)

for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns=vocab)
# 랜덤포레스트 분류기를 사용
forest = RandomForestClassifier(n_estimators = 50, n_jobs = -1, random_state=0)
forest
df_train.head()
%%time
traindata = pd.DataFrame(train_data_features.toarray(), columns=vocab)
forest.fit(traindata.values, df_train['Y'])
%time 
score = np.mean(cross_val_score(forest,
                                train_data_features,
                                df_train['Y'],
                                cv=10))
score
trainpredict = forest.predict(train_data_features.toarray())
trainpredict[-10:]
#test_data_features = pipeline.transform(df_test['words'].values)
test_data_features = pipeline.transform(df_test['utterance'].values)
test_data_features = test_data_features.toarray()

test_data_features
%time 

score = np.mean(cross_val_score(forest,
                                test_data_features,
                                df_test['Y'],
                                cv=10))
score
testdata = pd.DataFrame(test_data_features, columns=vocab)
testpredict = forest.predict(testdata.values)
pd.Series(testpredict).value_counts()
df_test['Y'].value_counts()
df_test.head()
%%time
last_data_features = pipeline.transform(df_last['utterance'].values)
last_data_features = last_data_features.toarray()

%%time
lastdata = pd.DataFrame(last_data_features, columns=vocab)
lastpredict = forest.predict(lastdata.values)
sample['Predicted'] = lastpredict
sample['Predicted'].value_counts()
# vectorizer = CountVectorizer(analyzer = 'word', 
#                              tokenizer = None,
#                              preprocessor = None, 
#                              stop_words = stops, 
#                              min_df = 4, # 토큰이 나타날 최소 문서 개수
#                              ngram_range=(1, 4),
#                              max_features = 10000
#                             )

vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = nltk.word_tokenize,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 2, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 4),
                             max_features = 10000
                            )
vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 2, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 4),
                             max_features = 10000
                            )
%%time

pipeline = Pipeline([
    ('vect', vectorizer),
])

train_data_features = pipeline.fit_transform(df_train['utterance'].values)
vectorizer._validate_vocabulary()
vocab = vectorizer.get_feature_names()

# 벡터화 된 피처를 확인해 봄
dist = np.sum(train_data_features, axis=0)

for tag, count in zip(vocab, dist):
    print(count, tag)
    
# 랜덤포레스트 분류기를 사용
forest = RandomForestClassifier(n_estimators = 25, n_jobs = -1, random_state=1)
data = pd.DataFrame(train_data_features.toarray(), columns=vocab)

# 데이터 학습
print("데이터 학습")
forest.fit(data.values, df_train['Y'])

score = np.mean(cross_val_score(forest,
                                train_data_features,
                                df_train['Y'],
                                cv=10))

# 정확도 출력 
print("정확도 학습")
score
%%time
test_data_features = pipeline.transform(df_test['utterance'].values)
test_data_features = test_data_features.toarray()
test = pd.DataFrame(test_data_features, columns=vocab)
testpredict = forest.predict(test.values)
score = np.mean(cross_val_score(forest,
                                test_data_features,
                                df_test['Y'],
                                cv=10))
score
pd.Series(testpredict).value_counts()
df_test['Y'].value_counts()
from sklearn.metrics import confusion_matrix
import pylab as plt
import seaborn as sns

cm = confusion_matrix(testpredict, df_test['Y'])
sns.heatmap(cm, cmap=sns.light_palette(
    "navy", as_cmap=True), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

%%time
last_data_features = pipeline.transform(df_last['utterance'].values)
last_data_features = last_data_features.toarray()
lasts = pd.DataFrame(last_data_features, columns=vocab)
lastpredict = forest.predict(lasts.values)
pd.Series(lastpredict).value_counts()
sample['Predicted'] = lastpredict
sample.to_csv("/kaggle/working/result2randomforest.csv", sep=",", index=False)
