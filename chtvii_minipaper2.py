import jieba

import gensim

import json

import numpy as np

from keras.models import Model,Sequential

from keras.layers import InputLayer,Embedding,LSTM,Dense,TimeDistributed,SimpleRNN

from keras.optimizers import SGD,Adam,Adadelta,RMSprop
with open('../input/songci/songci.txt','rb') as load_file:

    songs = json.load(load_file)

    lyrics = []

    for song in songs:

        for i in range(len(song['paragraphs'])):

            lyrics.append(song['paragraphs'][i].split('。'))
raw_text = []

raw_lyric = []

for lyric in lyrics:

    sentence = lyric[0].split('，')

    words = []

    for each_sentence in sentence:

        words.extend([word for word in each_sentence])

    new_words = []

    for word in words:

        new_words.extend(word.split('、'))

    new_words.extend('\n')

    raw_text.append(new_words)

    raw_lyric.extend(new_words)
modelx = gensim.models.word2vec.Word2Vec(raw_text, size=300,min_count=7)

modelx.save("w2v.model")

modelx = gensim.models.Word2Vec.load("./" + "w2v.model")

all_word_vector = modelx[modelx.wv.vocab]
modelx.most_similar('花',topn=10)
vocab_inv = list(modelx.wv.vocab)

vocab = {x:index for index,x in enumerate(vocab_inv)}
def build_matrix(text, vocab, length, step):

    M = []

    for word in text:

        index = vocab.get(word) # 提取word的value

        if (index is None):

            M.append(len(vocab)) # 为了防止搜不到词系统崩溃，统一将搜不到的词标记为第len(vocab)+1位

        else:

            M.append(index)

    num_sentences = len(M) // length

    M = M[: num_sentences * length] # 使M可被seq_len整除

    M = np.array(M)

    X = []

    Y = []

    for i in range(0, len(M) - length, step):

        X.append(M[i : i + length])

        Y.append([[x] for x in M[i + 1 : i + length + 1]])

    return np.array(X), np.array(Y)
seq_length = 7



X, Y = build_matrix(raw_lyric, vocab, seq_length, seq_length)

print(X.shape)

print(Y.shape)

print(X[0])

print(Y[0])
model = Sequential()

model.add(InputLayer(input_shape=(None, )))

model.add(Embedding(input_dim=len(vocab),output_dim=300,trainable=True,weights=[all_word_vector]))

model.add(LSTM(units=300,return_sequences=True)) # True输出所有结果

model.add(TimeDistributed(Dense(units=len(vocab)+1,activation="softmax")))

model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy')

model.summary()
model.fit(X,Y,batch_size=512,epochs=100,verbose=1)
st = '梦里不知身是客\n'

print(st,end = "")



vocab_inv.append('')



for i in range(150):

    X_sample = np.array([[vocab.get(x, len(vocab)) for x in st]]) # .get(x,default)指查找x的value，若未找到则返出default的值

    # pdt = (-model.predict(X_sample)[:, -1: , :][0][0]).argsort()[:4]

    out = model.predict(X_sample) # out.shape = (1,seq_length,4001)

    out_2 = out[:,-1:,:] # out_2.shape = (1，1，4001)

    out_3 = out_2[0][0] # out_3.shape = (4001,)

    out_4 = (-out_3).argsort() # ().argsort自小到大排序后 显示编号 >>>>> 该段代码即显示概率最高的四个字的序号

    pdt = out_4[:4]

    if pdt[0] == '\n' :

        ch = vocab_inv(pdt[0])

    else:

        ch = vocab_inv[np.random.choice(pdt)]

    print(ch, end='')

    st = st + ch