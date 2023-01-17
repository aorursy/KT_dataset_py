PLACEHOLDER_CORPUS_FILE_PATH = "../input/poetry.txt"

PLACEHOLDER_EMBEDDING_DIM = 128

PLACEHOLDER_LSTM_DIM = 128

PLACEHOLDER_MODEL_PATH = "/kaggle/working/lstm_poerty.h5"
## Init



# 如果没有安装 keras 和 tensorflow 库

# 请使用 pip install keras tensorflow 安装

import itertools

import jieba

import numpy as np

from collections import Counter

from keras.models import Model

from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed

from keras.optimizers import SGD, Adam, Adadelta



# 建立词汇表，为每种字赋予唯一的索引

def build_vocab(text, vocab_lim):

    word_cnt = Counter(itertools.chain(*text))

    vocab_inv = [x[0] for x in word_cnt.most_common(vocab_lim)]

    vocab_inv = list(sorted(vocab_inv))

    vocab = {x: index for index, x in enumerate(vocab_inv)}

    return vocab, vocab_inv



# 处理输入文本文件

def process_file(file_name, use_char_based_model):

    raw_text = []

    with open(file_name, "r") as f:

        for line in f:

            if (use_char_based_model):

                raw_text.extend([str(ch) for ch in line])

            else:

                raw_text.extend([word for word in jieba.cut(line)])

    return raw_text



# 格式化文本，建立词矩阵

def build_matrix(text, vocab, length, step):

    M = []

    for word in text:

        index = vocab.get(word)

        if index is None:

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

        Y.append(M[i + length])

    return np.array(X), np.array(Y)



raw_text = process_file(PLACEHOLDER_CORPUS_FILE_PATH, True)

vocab, vocab_inv = build_vocab(raw_text, 4000)
## Model Config



seq_length = 5

X, Y = build_matrix(raw_text, vocab, seq_length, 1)
## Run



# 构建模型

inputs = Input(shape=(None, ))

embedding = Embedding(input_dim=len(vocab) + 1, output_dim=PLACEHOLDER_EMBEDDING_DIM, trainable=True)(inputs)

lstm1 = LSTM(units=PLACEHOLDER_LSTM_DIM, return_sequences=False)(embedding)

outputs = Dense(units=len(vocab) + 1, activation='softmax')(lstm1)

model = Model(inputs=inputs, outputs=outputs)



# 编译模型

model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy')



# 输出模型报告

model.summary()
model.fit(X, Y, batch_size=512, epochs=20, verbose=1)

model.save(PLACEHOLDER_MODEL_PATH)
PLACEHOLDER_START_TEXT = "明月松间照" # 限制为5个字

PLACEHOLDER_TOPN = 4

PLACEHOLDER_GEN_LEN = 202

PLACEHOLDER_MODEL_PATH = "/kaggle/working/lstm_poerty.h5"
## Init



# 如果没有安装 keras 和 tensorflow 库

# 请使用 pip install keras tensorflow 安装

import itertools

import jieba

import numpy as np

from collections import Counter

from keras.models import Model, load_model

from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed

from keras.optimizers import SGD, Adam, Adadelta



# 建立词汇表，为每种字赋予唯一的索引

def build_vocab(text, vocab_lim):

    word_cnt = Counter(itertools.chain(*text))

    vocab_inv = [x[0] for x in word_cnt.most_common(vocab_lim)]

    vocab_inv = list(sorted(vocab_inv))

    vocab = {x: index for index, x in enumerate(vocab_inv)}

    return vocab, vocab_inv



# 处理输入文本文件

def process_file(file_name, use_char_based_model):

    raw_text = []

    with open(file_name, "r") as f:

        for line in f:

            if (use_char_based_model):

                raw_text.extend([str(ch) for ch in line])

            else:

                raw_text.extend([word for word in jieba.cut(line)])

    return raw_text



# 格式化文本，建立词矩阵

def build_matrix(text, vocab, length, step):

    M = []

    for word in text:

        index = vocab.get(word)

        if index is None:

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

        Y.append(M[i + length])

    return np.array(X), np.array(Y)





model = load_model(PLACEHOLDER_MODEL_PATH)

raw_text = process_file(PLACEHOLDER_CORPUS_FILE_PATH, True)

vocab, vocab_inv = build_vocab(raw_text, 4000)
## Model Config



st = PLACEHOLDER_START_TEXT
## Run



print(st, end='')

vocab_inv.append(' ')

for i in range(PLACEHOLDER_GEN_LEN):

    X_sample = np.array([[vocab.get(x, len(vocab)) for x in st]])

    pdt = (-model.predict(X_sample))[0].argsort()[:PLACEHOLDER_TOPN]

    if vocab_inv[pdt[0]] == '，' or vocab_inv[pdt[0]] == '。' or vocab_inv[pdt[0]] == '\n':

        ch = vocab_inv[pdt[0]]

    else:

        ch = vocab_inv[np.random.choice(pdt)]

    print(ch, end='')

    if len(st) == seq_length:

        st = st[1 :] + ch

    else:

        st = st + ch