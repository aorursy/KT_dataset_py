import numpy as np

import pandas as pd

import os

import sys

from joblib import Parallel, delayed





sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')

BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
import tokenization

dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', nrows=10000)
def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    for i in range(example.shape[0]):

        tokens_a = tokenizer.tokenize(example[i])

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    return np.array(all_tokens)
%%time

train_lines, train_labels = train_df['comment_text'].values, train_df.target.values 

token_input = convert_lines(train_lines, 25, tokenizer)
def convert_line(tl, max_seq_length,tokenizer):

    example = str(tl[0])

    y = tl[1]

    max_seq_length -=2

    tokens_a = tokenizer.tokenize(example)

    if len(tokens_a)>max_seq_length:

      tokens_a = tokens_a[:max_seq_length]

    one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

    return one_token, y
train_lines = zip(train_df['comment_text'].values.tolist(), train_df.target.values.tolist())
%time

res = Parallel(n_jobs=4, backend='multiprocessing')(delayed(convert_line)(i, 25, tokenizer) for i in train_lines)