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
naverdata = pd.read_csv("/kaggle/input/dfc615k/ko_data.csv", sep=",", encoding="ms949")
naversample = pd.read_csv("/kaggle/input/dfc615k/ko_sample.csv", sep=",", encoding="ms949")
moviecomments = pd.read_csv("/kaggle/input/naverdata3/naverfile3.csv", sep=",")
moviecomments = moviecomments[moviecomments['POINT'].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])]
moviecomments
moviecomments['POINT'] = moviecomments['POINT'].astype(str).astype(int)
moviecomments['Y'] = -1
moviecomments.loc[moviecomments['POINT']>=9, "Y"] = 1
moviecomments.loc[moviecomments['POINT']<=4, "Y"] = 0
moviecomments = moviecomments[moviecomments['Y']>=0]
moviecomments.head()
moviecomments['Y'].value_counts().plot(kind = 'bar')
print(moviecomments.groupby('Y').size().reset_index(name = 'count'))

print(moviecomments.isnull().sum())
moviecomments = moviecomments.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(moviecomments.isnull().values.any()) # Null 값이 존재하는지 확인
print(len(moviecomments))
moviecomments.head()
naverdata.head()
moviecomments['COMMENT'] = moviecomments['COMMENT'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
naverdata['Sentence'] = naverdata['Sentence'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
moviecomments['COMMENT'].replace('', np.nan, inplace=True)
print(moviecomments.isnull().sum())
naverdata['Sentence'].replace('', np.nan, inplace=True)

naverdata['Sentence'].replace(np.nan, '', inplace=True)


print(naverdata.isnull().sum())
moviecomments = moviecomments.dropna(how = 'any')
print(len(moviecomments))
data1 = moviecomments['COMMENT'].tolist()
data2 = naverdata['Sentence'].tolist()
data = [*data1, *data2]
size = len(data1)
len(data1), len(data2), len(data)
#!pip install --upgrade pip
!pip install konlpy
from konlpy.tag import Okt
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

#moviecomments = moviecomments.reset_index(drop=True)
data[:5]
%%time
X = []

moviecomments["WORDS"] = ''

# for i in range(0, len(moviecomments)):
#     sentence = moviecomments.loc[i, "COMMENT"]
#     sentence = okt.morphs(sentence, stem=True) # 토큰화
#     X.append([word for word in sentence if not word in stopwords]) # 불용어 제거    

for i in range(0, len(data)):
    sentence = str(data[i])
    try:        
        sentence = okt.morphs(sentence, stem=True) # 토큰화
    except:
        print('Error', i)
    X.append([word for word in sentence if not word in stopwords]) # 불용어 제거    
    
len(X)
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
vocab_size = total_cnt - rare_cnt + 1 # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
print('단어 집합의 크기 :',vocab_size)
len(X)
tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X)

X_train = tokenizer.texts_to_sequences(X[:5000])
X_test = tokenizer.texts_to_sequences(X[5000:size])
X_last = tokenizer.texts_to_sequences(X[size:])
y_train = np.array(moviecomments['Y'][:5000])
y_test = np.array(moviecomments['Y'][5000:size])
y_last = np.array(moviecomments['Y'][size:])
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
max_len = 30
below_threshold_len(max_len, X_train)
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
X_last = pad_sequences(X_last, maxlen = max_len)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
len(X_train), len(y_train)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
len(X_test), len(y_test)
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
Xtestprediction = loaded_model.predict(X_test)
len(Xtestprediction), len(y_test)
def to_bool(s):
    return 1 if s > 0.5 else 0
from sklearn.metrics import confusion_matrix
import pylab as plt
import seaborn as sns

cm = confusion_matrix([ to_bool(x) for x in Xtestprediction], y_test)
sns.heatmap(cm, cmap=sns.light_palette(
    "navy", as_cmap=True), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()


predicted = loaded_model.predict(X_last)
len(predicted)
naversample['Predicted'] =  [ to_bool(x) for x in predicted]
naversample[['Id', 'Predicted']].to_csv("/kaggle/working/output1_lstm.csv", sep=",", encoding="ms949", index=False)

