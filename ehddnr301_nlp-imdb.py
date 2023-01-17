# IMDB 데이터셋 로드
from keras.datasets import imdb

# Numpy, Pandas, Matplotlib 로드
import numpy as np # 파이썬에서 수치를 다루기 위한 모듈
import pandas as pd # 파이썬에서 table을 다루기 위한 모듈
import matplotlib.pyplot as plt # 파이썬에서 그림을 그리기 위한 모듈
import tensorflow as tf
import os
%matplotlib inline

import pandas as pd 
import numpy as np 



# 모델을 위한 모듈
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout, GRU, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from tensorflow.keras.models import load_model

import re
NUM_WORDS= 20000
BATCH_SIZE = 1024
EPOCHS=10
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS) # 이미 토큰화 되어있습니다.
(unique,counts) = np.unique(y_train, return_counts=True)
print(unique, counts)

word_dict = imdb.get_word_index() # word_dict는 'word' : idx 형태입니다.
reverse_word_dict = {k:v for v, k in word_dict.items()} # idx : 'word' 형태로 바꿔주기 위함입니다.
for i in x_train[0]: # 첫번째 문장의 토큰화된 단어들을 가져와서 reverse_word_dict 를 통해서 무슨단어인지 알아냅니다.
    
    print(reverse_word_dict[i], end = " ") # end parameter 는 개행 하지않고 이어붙일때 사용합니다.
print("\n")
print(x_train[0])

print('maximun length : {}'.format(max(len(l) for l in x_train))) # format 으로 {} 안의 내용을 바꿀수 있다.
print('average length : {}'.format(sum(map(len, x_train))/len(x_train)))
# map(f, iterable)은 함수(f)와 반복 가능한(iterable) 자료형을 입력으로 받는다. 
# map은 입력받은 자료형의 각 요소를 함수 f가 수행한 결과를 묶어서 돌려주는 함수이다.

plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length')
plt.ylabel('number')
plt.show()

max_len = 500
vocab_size = NUM_WORDS #  imdb 데이터가 가지고 있는 단어의 개수이다. | num_words를 20,000으로 했기 때문에 총 단어 개수는 20,000개이다.
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
# pad_sequences 를 활용
# 문장의 최대 길이는 500, 더 길면 잘리고 더 짧으면 패딩처리
def create_model(type=""):
    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    # 우리가 imdb 데이터에 사용되는 총 20,000개의 문장을 단어 하나 하나마다 100차원의 임베딩하겠다는 레이어를 뜻합니다.
    # 여기서 임베딩 되는게 밀집 표현(Dense Representation) 인건지?
    if(type == 'GRU'):
        model.add(GRU(100))
    elif(type == 'LSTM'):
        model.add(LSTM(100))
    else:
        model.add(SimpleRNN(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

model = create_model('LSTM')
model.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[es, mc])

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.title('Training')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()
plt.show()
loaded_model = load_model('./GRU_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))
# loaded_model 은 loss 와 acc 를 리턴하는데 acc 를 출력하기위해 [1] 추가하였습니다.
def sentiment_predict(new_sentence):
  # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
  new_sentence = re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()

  # 정수 인코딩
  encoded = []
  for word in new_sentence.split():
    # 단어 집합의 크기를 10,000으로 제한.
    try :
      if word_dict[word] <= 10000:
        encoded.append(word_dict[word]+3)
      else:
    # 10,000 이상의 숫자는 <unk> 토큰으로 취급.
        encoded.append(0)
    # 단어 집합에 없는 단어는 <unk> 토큰으로 취급.
    except KeyError:
      encoded.append(0)

  pad_new = pad_sequences([encoded], maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
temp_str = "This movie was just way too overrated. The fighting was not professional and in slow motion. I was expecting more from a 200 million budget movie. The little sister of T.Challa was just trying too hard to be funny. The story was really dumb as well. Don't watch this movie if you are going because others say its great unless you are a Black Panther fan or Marvels fan."

sentiment_predict(temp_str)
temp_str = " I was lucky enough to be included in the group to see the advanced screening in Melbourne on the 15th of April, 2012. And, firstly, I need to say a big thank-you to Disney and Marvel Studios. \
Now, the film... how can I even begin to explain how I feel about this film? It is, as the title of this review says a 'comic book triumph'. I went into the film with very, very high expectations and I was not disappointed. \
Seeing Joss Whedon's direction and envisioning of the film come to life on the big screen is perfect. The script is amazingly detailed and laced with sharp wit a humor. The special effects are literally mind-blowing and the action scenes are both hard-hitting and beautifully choreographed."

sentiment_predict(temp_str)
