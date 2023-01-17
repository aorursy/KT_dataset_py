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
import nltk

from tqdm import tqdm

import numpy as np

import pandas as pd

import re

# from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import itertools

import matplotlib.pyplot as plt

# nltk.download('punkt')
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
모든전처리들 = []
def 결측값제거(DATA):

    """

    결측값을 데이터에서 제거해주는 함수. row를 날려버린다. 

    날린 row의 index를 출력한다

    index를 다시 맞춰준다. 

    사용법은 DATA = 결측값제거(DATA)

    """

    null_columns=DATA.columns[DATA.isnull().any()]

    null_list = DATA[DATA.isnull().any(axis=1)][null_columns].index

    for i in null_list:

        print("결측값 row=", i,"제거!")

        DATA = DATA.drop(i, axis=0)

    DATA.reset_index(inplace=True)

    print("Empty Data Remove")

    return DATA
모든전처리들.append(결측값제거)
def 애스터리스크를_퍽으로_바꾸기(DATA):

    """

    욕설을 아마 ****로 처리한 것 같은데, 데이터에서 나오는 모든 ****를 다 fuck로 바꿔준다.

    

    """

    tmp = DATA.columns

    for i in ["text", "selected_text"]:

        try:

            DATA[i] = DATA[i].str.replace("****", "FUCK", regex=False)

        except:

            pass

    print("**** → fuck")

    return DATA
모든전처리들.append(애스터리스크를_퍽으로_바꾸기)
def URL제거기(DATA):

    """

    문장에 URL이 있으면 찾아서 URLWASHERE 라는 단어로 바꿔준다

    """

    #URL 제거 정규표현식

    URL_Rex = "(https?):\/\/([a-zA-Z0-9-\.\/~]+)+"

    targets = ["text", "selected_text"]

    for i in targets:

        try:

            DATA[i] = DATA[i].str.replace(URL_Rex, "URLWASHERE", regex=True )

        except:

            pass

    print("Replace All URL's to URLWASHERE")

    return DATA
모든전처리들.append(URL제거기)
def 알파벳만남기기(DATA):

    """알파벳이 아닌 글자를 다 날린다."""

    targets = ["text", "selected_text"]

    for i in targets:

        try:

            DATA[i] = DATA[i].str.replace("[^a-zA-Z]", " ")

        except:

            pass

    print("Remove all Characters EXCEPT Alphabet")

    return DATA
모든전처리들.append(알파벳만남기기)
def 소문자화(DATA):

    """다 소문자"""

    targets = ["text", "selected_text"]

    for i in targets:

        try:

            DATA[i] = DATA[i].str.lower()

        except:

            pass

    print("Lower Case")

    return DATA
모든전처리들.append(소문자화)
def 세개연속된문자처리(DATA):

    """연속된 글자는 최대 2개 까지만 허용한다. coool -> cool """

    consecutive_characters = ["aaa","bbb","ccc","ddd","eee","fff","ggg","hhh","iii","jjj","kkk","lll","mmm",

                              "nnn","ooo","ppp","qqq","rrr","sss","ttt","uuu","vvv","www","xxx","yyy","zzz"]



    targets = ["text", "selected_text"]

    for i in targets:

        try:

            for j in consecutive_characters:

                DATA[i] = DATA[i].str.replace(j, j[:1])

            for j in consecutive_characters:

                DATA[i] = DATA[i].str.replace(j, j[:1])

            for j in consecutive_characters:

                DATA[i] = DATA[i].str.replace(j, j[:1])

            for j in consecutive_characters:

                DATA[i] = DATA[i].str.replace(j, j[:1])

            for j in consecutive_characters:

                DATA[i] = DATA[i].str.replace(j, j[:1])

            for j in consecutive_characters:

                DATA[i] = DATA[i].str.replace(j, j[:1])



        except:

            pass

    

    print("Remove consecutive characters more than 3. (eg:cooooooooool → cool) ")

    return DATA
모든전처리들.append(세개연속된문자처리)
def 감성원핫인코딩(DATA):

    tmp = pd.concat([DATA, pd.get_dummies(DATA.sentiment)], axis=1)

    print("Generate One-Hot Encoded Sentiment column")

    return tmp 
모든전처리들.append(감성원핫인코딩)
def 종합전처리기(DATA):

    for i in 모든전처리들:

        DATA = i(DATA)

    return DATA
train = 종합전처리기(train)
test = 종합전처리기(test)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train["text"])
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
tokenizer = Tokenizer(vocab_size) 

tokenizer.fit_on_texts(train["text"])
def 토큰화(DATA):

    DATA["token"] = tokenizer.texts_to_sequences(DATA["text"])

    return DATA
train = 토큰화(train)

test = 토큰화(test)
def 토큰화과정에서제거당한열제거기(DATA):

    no_token_list = DATA[DATA["token"].str.len()==0].index

    for i in no_token_list:

        DATA = DATA.drop(i, axis=0)

    DATA.reset_index(inplace=True)       

    print("토큰화 해서 사라진 값 ", i,"제거!")

    return DATA
train = 토큰화과정에서제거당한열제거기(train)
test = 토큰화과정에서제거당한열제거기(test)
def 패딩된토큰만들기(DATA):

    DATA["padding_token"]= pad_sequences(DATA['token'], maxlen=25).tolist()

    return DATA
train = 패딩된토큰만들기(train)

test = 패딩된토큰만들기(test)
from tensorflow.keras.layers import Embedding, Dense, LSTM

from tensorflow.keras.models import Sequential

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model = Sequential()

model.add(Embedding(vocab_size, 100)) 

model.add(LSTM(128))

model.add(Dense(3, activation='softmax'))
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=False)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'] )
y = train[["negative","neutral","positive"]]



# y = np.array(y.values, dtype=object)

y = np.array(y.values)



y[:3]
X = train["padding_token"]



# X = np.array(X.values.tolist(), dtype=object)

X = np.array(X.values.tolist())

X
import os

if "best_model.h5" in os.listdir():

    model = load_model('best_model.h5') # 경고! 전처리 다시 했으면 이 파일을 삭제해줘야 재학습함. 있으면 말고. 

else:

    history = model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2, callbacks=[mc], workers=0)

    
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):

    # Looking up words in dictionary

    words = [reverse_word_map.get(letter) for letter in list_of_indices]

    return(words)
def 셀렉티드생성기(DATA):

    """

    예시문장 = 단어1 단어2 단어3 단어4 단어5 단어6 단어7 

    이 있으면,

    확률평가(예시문장) = (부정확률, 중립확률, 긍정확률)

    이 나온다.

    

    이 중, 예시문장에서 i~j 번 단어를 뽑은 예시문장 tmp를 만들고 tmp를 확률평가한 뒤,

    (i는 1에서7 사이의 정수, j는 i+1에서 7 사이의 정수)

    

    전체 문장의 감성이 부정일 경우, 최대 부정 확률을 나타내는 i,j 값을 찾는다.

    전체 문장의 감성이 긍정일 경우, 최대 긍정 확률을 나타내는 i,j 값을 찾는다.

    전체 문장의 감성이 중립일 경우, 최대 중립 확률을 나타내는 i,j 값을 찾는다.

    

    이 단어i ~ 단어j 가 센티멘트를 나타내는 selected_text 이다!

    

    """

    total_selected_text = []

    for 개별트윗 in tqdm(DATA["padding_token"].tolist()):

        셀렉티드 = [x for x in 개별트윗 if x is not 0]

        개별트윗문장평가값 =  model.predict([개별트윗])

        

        

        for i in range(24):

            if 개별트윗[i] ==0:

                continue

            for j in range(i+1, 25, 3):

                패딩된개별트윗부분 = pad_sequences([개별트윗[i:j]], maxlen=25).tolist()

                if max(패딩된개별트윗부분[0]) == 0: # None만 있으면 제낀다.

                    continue

                predict = model.predict(패딩된개별트윗부분)

                inx = predict[0].argmax()

                

                if inx == 개별트윗문장평가값[0].argmax(): # 그 부분의 평가가 문장 전체의 평가랑 일치하며

                    if np.max(predict[0]) >= np.max(개별트윗문장평가값[0]): # 그 부분이 더 높은 평가값을 보일 경우

                        셀렉티드 = 개별트윗[i:j]

        

        

        결과텍스트 = '"'+" ".join(sequence_to_text(셀렉티드))+'"'

        결과텍스트 = 결과텍스트.replace("urlwashere","")

        total_selected_text.append(결과텍스트)

        

    DATA["selected_text"] = total_selected_text

    return DATA            
submission_file = 셀렉티드생성기(test)
submission_file
submission_file=submission_file.drop(["level_0","index", "text","sentiment", "negative","neutral", "positive", "token", "padding_token" ], axis=1)

submission_file
submission_file.to_csv("submission.csv")