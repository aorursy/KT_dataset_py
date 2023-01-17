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

            lyrics.append(song['paragraphs'][i])
print(lyrics[:100])
raw_text = []

raw_text_w2v = []

for lyric in lyrics:

    if not lyric[-1] == '。':

        continue

    if '、' in lyric:

        continue

    if '□' in lyric:

        continue

    lyric = lyric + '\n'

    raw_text.extend([word for word in lyric])

    raw_text_w2v.append([word for word in lyric])
print(raw_text[:1000])

print(raw_text_w2v[:10])
modelx = gensim.models.word2vec.Word2Vec(raw_text_w2v, size=300, min_count=1)

modelx.save("w2v.model")

modelx = gensim.models.Word2Vec.load("./" + "w2v.model")
modelx.most_similar('玉', topn=10)
all_word_vector = modelx[modelx.wv.vocab]

all_word_vector = np.append(all_word_vector, [np.zeros(300)], axis = 0)

print(all_word_vector.shape)
vocab_inv = list(modelx.wv.vocab)

vocab = {x:index for index,x in enumerate(vocab_inv)}



print(vocab_inv[:10])
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
seq_length = 4



X, Y = build_matrix(raw_text, vocab, seq_length, seq_length)

print(X.shape)

print(Y.shape)

print(X[0])

print(Y[0])
model = Sequential()

model.add(InputLayer(input_shape=(None, )))

model.add(Embedding(input_dim=len(vocab)+1,output_dim=300,trainable=True,weights=[all_word_vector]))

model.add(LSTM(units=300,return_sequences=True)) # True输出所有结果

model.add(TimeDistributed(Dense(units=len(vocab)+1,activation="softmax")))

model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy')

model.summary()
model.fit(X,Y,batch_size=512,epochs=40,verbose=1)
st = '今宵酒醒何处，'

print(st,end = "")



vocab_inv.append('')



for i in range(200):

    X_sample = np.array([[vocab.get(x, len(vocab)) for x in st]]) # .get(x,default)指查找x的value，若未找到则返出default的值

    out = model.predict(X_sample) # out.shape = (1,seq_length,4001)

    out_2 = out[:,-1:,:] # out_2.shape = (1，1，4001)

    out_3 = out_2[0][0] # out_3.shape = (4001,)

    out_4 = (-out_3).argsort() # ().argsort自小到大排序后 显示编号 >>>>> 该段代码即显示概率最高的四个字的序号

    pdt = out_4[:3]

    #if pdt[0] == '\n' or pdt[0] == ',':

    #    ch = vocab_inv(pdt[0])

    #else:

    pb = [out_3[index] for index in pdt]

    # print([(vocab_inv[index], out_3[index] / sum(pb)) for index in pdt])

    if vocab['，'] in pdt:

        ch = '，'

    else:

        ch = vocab_inv[np.random.choice(pdt, p=pb/sum(pb))]

    print(ch, end='')

    st = st + ch