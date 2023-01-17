import os

import random

import spacy

import numpy as np

import math

import itertools

from tqdm import tqdm

from collections import Counter

from scipy.sparse import csr_matrix, coo_matrix

from keras.utils import to_categorical

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Embedding, LSTM, SimpleRNN, Bidirectional

from keras.preprocessing.sequence import pad_sequences

root = '../input'

random.seed(0)
def read_imdb(data_path, folder):

    result = []

    for label in ['pos', 'neg']:

        folder_path = os.path.join(data_path, folder, label)

        for file in tqdm(os.listdir(folder_path)):

            with open(os.path.join(folder_path, file), 'r', encoding='utf8') as f:

                text = f.read()

                text = text.replace('<br /><br />', ' ').lower()

            result.append([text, int(label == 'pos')])

    random.shuffle(result)

    return result



data_path = os.path.join(root, 'aclimdb/aclImdb')

train_data = read_imdb(data_path, 'train')

test_data = read_imdb(data_path, 'test')

print(len(train_data), len(test_data))

print(train_data[0])
def tokenize(texts):

    """

    @param texts: 训练文本构成的列表，每个元素都是一个训练文本

    @return: 分词结果，二维列表

    """

    result = []

    nlp = spacy.load('en_core_web_sm')

    tokenizer = nlp.tokenizer

    for text in tqdm(texts):

        doc = tokenizer(text)

        text = [token.text.lower() for token in doc]

        result.append(text)

    return result



train_X = tokenize([text for text, _ in train_data])

train_y = [label for _, label in train_data]

test_X = tokenize([text for text, _ in test_data])

test_y = [label for _, label in test_data]



print(train_X[0])

print(train_y[0])
def evaluate(labels, predictions):

    TP, FP, FN = 0, 0, 0

    for y, y_hat in zip(labels, predictions):

        TP += (y == 1 and y_hat == 1)

        FP += (y == 0 and y_hat == 1)

        FN += (y == 1 and y_hat == 0)

    P = TP / (TP + FP)

    R = TP / (TP + FN)

    F1 = 2 * P * R / (P + R)

    return P, R, F1
labels = [0, 0, 1, 1, 1]

predictions = [0, 1, 0, 1, 1]

P, R, F1 = evaluate(labels, predictions)

print('%.2f%%, %.2f%%, %.2f%%' % (P * 100, R * 100, F1 * 100))
def read_senti_dict(data_path, filename):

    with open(os.path.join(data_path, filename), 'r') as f:

        words = [word.strip().lower() for word in f]

    return set(words)



sentiment_dict_path = os.path.join(root, 'sentimentdictionary/Hu and Liu Sentiment Lexicon')

pos_file, neg_file = 'positive-words.txt', 'negative-words.txt'

pos_words = read_senti_dict(sentiment_dict_path, pos_file)

neg_words = read_senti_dict(sentiment_dict_path, neg_file)



print(list(pos_words)[: 10])

print(list(neg_words)[: 10])

print(len(pos_words))

print(len(neg_words))
def predict_dictionary(test_X, pos_words, neg_words):

    result = []

    for text in tqdm(test_X):

        score = 0

        for token in text:

            score += (token.lower() in pos_words)  # step 1

            score -= (token.lower() in neg_words)  # step 2

        result.append(int(score > 0))  # step 3

    return result
predictions = predict_dictionary(test_X, pos_words, neg_words)

precision, recall, F1 = evaluate(test_y, predictions)

print('precision: %.2f%%' % (precision * 100))

print('recall: %.2f%%' % (recall * 100))

print('F1: %.2f%%' % (F1 * 100))
def get_vocab(texts):

    tokens = set(itertools.chain(*texts))  # step 1

    idx2token = ['<pad>', '<unk>'] + list(tokens)  # step 2

    token2idx = {token: idx for idx, token in enumerate(idx2token)}  # step 3

    return idx2token, token2idx



idx2token, token2idx = get_vocab(train_X)

print('Vocab size: %d' % len(idx2token))
def cal_bow(texts, token2idx, use_tf):

    value, col_ind, row_ptr = [], [], [0]

    for text in texts:

        counter = Counter(text)  # step 1

        value += [v if use_tf else 1 for v in counter.values()]  # step 2

        col_ind += [token2idx[k] for k in counter.keys()]  # step 2

        row_ptr.append(len(value))  # step 3

    return csr_matrix((value, col_ind, row_ptr), dtype=np.float32)
def cal_parameter(train_X, train_y, token2idx):

    """

    @param train_X: 训练数据的评论文本

    @param train_y: 训练数据的情感标签

    @param token2idx: 从单词到索引的映射

    @return: pr_0: Pr(xi=1|y=0), shape: (vocab_size,)

             pr_1: Pr(xi=1|y=1), shape: (vocab_size,)

             pr_y: Pr(y=1)

    """

    train_X = cal_bow(train_X, token2idx, use_tf=False)

    train_y = np.array(train_y)

    neg_idx = train_y == 0

    pos_idx = train_y == 1

    pr_0 = (train_X[neg_idx].sum(0) + 1) / (neg_idx.sum() + 2)  # step 1

    pr_1 = (train_X[pos_idx].sum(0) + 1) / (pos_idx.sum() + 2)  # step 2

    pr_0 = np.array(pr_0).reshape(-1)

    pr_1 = np.array(pr_1).reshape(-1)

    pr_y = pos_idx.sum() / len(train_y)  # step 3

    return pr_0, pr_1, pr_y



pr_0, pr_1, pr_y = cal_parameter(train_X, train_y, token2idx)

print(pr_0.shape)

print(pr_1.shape)

print(pr_y.shape)
def predict_naive_bayes(pr_0, pr_1, pr_y, test_X, token2idx):

    """

    @param pr_0: Pr(xi=1|y=0), shape: (vocab_size,)

    @param pr_1: Pr(xi=1|y=1), shape: (vocab_size,)

    @param pr_y: Pr(y=1)

    @param test_X: 测试数据文本，二维列表

    @param token2idx: 从单词到索引的映射

    @return: 预测结果，列表

    """

    result = []

    

    def cal_vector(text, token2idx):

        result = np.zeros(len(token2idx))

        for token in set(text):

            if token not in token2idx:

                continue

            result[token2idx[token]] = 1

        return result

    

    for text in tqdm(test_X):

        x = cal_vector(text, token2idx)  # step 1

        p0 = (np.log(pr_0) * x + np.log(1 - pr_0) * (1 - x)).sum() + np.log(1 - pr_y)  # step2

        p1 = (np.log(pr_1) * x + np.log(1 - pr_1) * (1 - x)).sum() + np.log(pr_y)  # step 2

        result.append(int(p1 > p0))  # step 3

    return result
predictions = predict_naive_bayes(pr_0, pr_1, pr_y, test_X, token2idx)

P, R, F1 = evaluate(test_y, predictions)

print('Precision: %.2f%%' % (P * 100))

print('Recall: %.2f%%' % (R * 100))

print('F1: %.2f%%' % (F1 * 100))
def create_model(vocab_size, embed_size, hidden_size):

    model = Sequential()

    model.add(Embedding(vocab_size, embed_size))

    model.add(Bidirectional(LSTM(hidden_size)))

    model.add(Dense(2, activation='softmax'))

    model.compile('adam', 'categorical_crossentropy')

    return model
model = create_model(vocab_size=len(token2idx), embed_size=100, hidden_size=64)

model.summary()
def train(model, train_X, train_y, token2idx, maxlen, epochs, batch_size):

    train_X = [[token2idx[token] for token in text] for text in train_X]  # step 1

    train_X = pad_sequences(train_X, maxlen=maxlen, padding='post', value=token2idx['<pad>'])  # step 2

    train_y = to_categorical(np.array(train_y), num_classes=2)  # step 3

    model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs)  # step 4
def predict_RNN(model, test_X, token2idx, maxlen, batch_size):

    test_X = [[token2idx[token] if token in token2idx else token2idx['<unk>'] for token in text] for text in test_X]  # step 1

    test_X = pad_sequences(test_X, maxlen=maxlen, padding='post')  # step 2

    predictions = model.predict_classes(test_X, batch_size=batch_size)  # step 3

    predictions = predictions.reshape(-1).tolist()

    return predictions
def run(train_X, train_y, test_X, test_y, embed_size, hidden_size, maxlen, epochs, batch_size, token2idx):

    model = create_model(len(token2idx), embed_size, hidden_size)

    train(model, train_X, train_y, token2idx, maxlen, epochs, batch_size)

    predictions = predict_RNN(model, test_X, token2idx, maxlen, batch_size)

    P, R, F1 = evaluate(test_y, predictions)

    print('Precision: %.2f%%' % (P * 100))

    print('Recall: %.2f%%' % (R * 100))

    print('F1: %.2f%%' % (F1 * 100))
embed_size = 100

hidden_size = 64

maxlen = 120

epochs = 2

batch_size = 512

run(train_X, train_y, test_X, test_y, embed_size, hidden_size, maxlen, epochs, batch_size, token2idx)