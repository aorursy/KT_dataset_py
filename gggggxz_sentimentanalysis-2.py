import os

import random

import re

import spacy

import time

import numpy as np

import math

from tqdm import tqdm

from collections import Counter

from scipy.sparse import csr_matrix, coo_matrix



random.seed(2)  # 确保每次运行结果一致
def read_imdb(data_path, folder):

    result = []

    for label in ['pos', 'neg']:

        folder_path = os.path.join(data_path, folder, label)

        for file in tqdm(os.listdir(folder_path)):

            with open(os.path.join(folder_path, file), 'r', encoding='utf8') as f:

                text = f.read()

                text = re.sub(r'\s*<br /><br />\s*', ' ', text)

            result.append([text, int(label == 'pos')])

    random.shuffle(result)

    return result
data_path = '../input/aclimdb/aclImdb'

train_data = read_imdb(data_path, 'train')

test_data = read_imdb(data_path, 'test')
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
#############################################################################

#             任务 ：从语料中构建词典                                       #

#############################################################################

def get_vocab(texts):

    """

    @param texts:

    @return:

    """

    tokens = set()

    for sentence in texts:

        tokens.update(sentence)

    idx2token = list(tokens)

    token2idx = {token: idx for idx, token in enumerate(idx2token)}

    return idx2token, token2idx
idx2token, token2idx = get_vocab(train_X)

print('Vocab size: %d' % len(idx2token))



idx = token2idx['hello']

print(idx2token[idx])
#############################################################################

#             任务 ：创建CSR格式的词袋特征                                  #

#############################################################################

def cal_bow(texts, token2idx, use_tf):

    """

    @param texts:

    @param token2idx:

    @param use_tf:

    @return:

    """

    value, col_ind, row_ptr = [], [], [0]

    for text in texts:

        counter = Counter(text)

        value += [v if use_tf else 1 for v in counter.values()]

        col_ind += [token2idx[k] for k in counter.keys()]

        row_ptr.append(len(value))

    return csr_matrix((value, col_ind, row_ptr), dtype=np.float32)
#############################################################################

#             任务 ：计算模型参数                                           #

#############################################################################

def cal_parameter(train_X, train_y, token2idx):

    """

    @param train_X:

    @param train_y:

    @param token2idx:

    @return:

    """

    train_X = cal_bow(train_X, token2idx, use_tf=False)

    train_y = np.array(train_y)

    neg_idx = train_y == 0

    pos_idx = train_y == 1

    pr_0 = (train_X[neg_idx].sum(0) + 1) / (neg_idx.sum() + 2)

    pr_1 = (train_X[pos_idx].sum(0) + 1) / (pos_idx.sum() + 2)

    pr_0 = np.array(pr_0).reshape(-1)

    pr_1 = np.array(pr_1).reshape(-1)

    pr_y = pos_idx.sum() / len(train_y)

    return pr_0, pr_1, pr_y
pr_0, pr_1, pr_y = cal_parameter(train_X, train_y, token2idx)
num = 30

def print_most_common(pr, idx2token, num):

    """

    输出pr中概率值最大的num个位置对应的词

    """

    most_common = np.argsort(pr).tolist()[:: -1]

    for i, idx in enumerate(most_common[: num]):

        print('%-13s%c' % (idx2token[idx], " \n"[(i + 1) % 6 == 0]), end='')  # 每行输出6个

    print()



print_most_common(pr_0, idx2token, num)

print_most_common(pr_1, idx2token, num)
def print_farthest(pr_0, pr_1, idx2token, num):

    """

    输出pr_0和pr_1中排名差距最大的num个词

    """

    V = len(pr_0)

    idx0 = np.argsort(pr_0).tolist()[:: -1]

    idx1 = np.argsort(pr_1).tolist()[:: -1]

    rank0, rank1 = [0] * V, [0] * V

    for i, v in enumerate(idx0):

        rank0[v] = i

    for i, v in enumerate(idx1):

        rank1[v] = i

    diff = sorted(range(V), key=lambda x: rank0[x] - rank1[x])

    for i, idx in enumerate(diff[: num]):

        print('%-13s%c' % (idx2token[idx], ' \n'[(i + 1) % 6 == 0]), end='')

    print()

    for i, idx in enumerate(diff[-num:]):

        print('%-13s%c' % (idx2token[idx], ' \n'[(i + 1) % 6 == 0]), end='')

    

print_farthest(pr_0, pr_1, idx2token, 20)
#############################################################################

#             任务 ：对训练数据进行预测                                     #

#############################################################################

def predict(pr_0, pr_1, pr_y, test_X, token2idx):

    """

    @param pr_0:

    @param pr_1:

    @param pr_y:

    @param test_X:

    @param token2idx:

    @return:

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

        x = cal_vector(text, token2idx)

        p0 = (np.log(pr_0) * x + np.log(1 - pr_0) * (1 - x)).sum() + np.log(1 - pr_y)

        p1 = (np.log(pr_1) * x + np.log(1 - pr_1) * (1 - x)).sum() + np.log(pr_y)

        result.append(int(p1 > p0))

    return result
predictions = predict(pr_0, pr_1, pr_y, test_X, token2idx)
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
P, R, F1 = evaluate(test_y, predictions)

print('Precision: %.2f%%' % (P * 100))

print('Recall: %.2f%%' % (R * 100))

print('F1: %.2f%%' % (F1 * 100))