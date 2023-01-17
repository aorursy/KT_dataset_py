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
print(songs[199])
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
print(raw_text[0:100])
modelx = gensim.models.word2vec.Word2Vec(raw_text_w2v, size=300, min_count=1)

modelx.save("w2v.model")

modelx = gensim.models.Word2Vec.load("./" + "w2v.model")

all_word_vector = modelx[modelx.wv.vocab]

all_word_vector = np.append(all_word_vector, [np.zeros(300)], axis = 0)

print(all_word_vector.shape)
modelx.most_similar('花', topn=10)
vocab_inv = list(modelx.wv.vocab)

vocab = {x:index for index,x in enumerate(vocab_inv)}
def build_matrix(text, vocab, length, step):

    M = []

    for word in text:

        index = vocab.get(word)

        if (index is None):

            M.append(len(vocab)) 

        else:

            M.append(index)

    num_sentences = len(M) // length

    M = M[: num_sentences * length] 

    M = np.array(M)

    X = []

    Y = []

    for i in range(0, len(M) - length, step):

        X.append(M[i : i + length])

        Y.append([[x] for x in M[i + 1 : i + length + 1]])

    return np.array(X), np.array(Y)
seq_length = 4



X, Y = build_matrix(raw_text, vocab, seq_length, seq_length)



print("第150个输入矩阵：",X[150])

print("第150个输出矩阵：\n",Y[150])
model = Sequential()

model.add(InputLayer(input_shape=(None, )))

model.add(Embedding(input_dim=len(vocab)+1,output_dim=300,trainable=True,weights=[all_word_vector]))

# 词嵌入层，input_dim表示词典的大小，每个词对应一个output_dim维的词向量，weights为先前训练好的词向量

model.add(LSTM(units=300,return_sequences=True)) 

# LSTM层，语言处理层，输出形状为(seq_length,300)

model.add(TimeDistributed(Dense(units=len(vocab)+1,activation="softmax")))

# 输出的300维向量需要经过一次线性变换（也就是Dense层）转化为len(vocab)+1维的向量，用softmax变换将其转化为概率分布，第i维表示下一个时刻的词是i号单词的概率

model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy')

# 优化器为Adam，损失函数为交叉熵

model.summary()
model.fit(X,Y,batch_size=512,epochs=40,verbose=1)
st = '昨夜雨疏风骤，'

print(st,end = "")



vocab_inv.append('')



i = 0

while i < 100:

    X_sample = np.array([[vocab.get(x, len(vocab)) for x in st]]) 

    out = model.predict(X_sample) 

    out_2 = out[:,-1:,:] 

    out_3 = out_2[0][0]

    out_4 = (-out_3).argsort() 

    pdt = out_4[:3]

    pb = [out_3[index] for index in pdt]

    if vocab['，'] in pdt:

        ch = '，'

    else:

        ch = vocab_inv[np.random.choice(pdt, p=pb/sum(pb))]

    print(ch, end='')

    st = st + ch

    if vocab[ch] != len(vocab) and ch != '，' and ch != '。' and ch != '\n':

        i += 1