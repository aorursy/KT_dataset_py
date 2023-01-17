# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import shuffle

MAX_SENTENCE_WIDTH = 100
EMBED_DIM = 300
SUPER_BATCH_SIZE = 10000

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    import re
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(df):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from df
    positive_examples = [row.question_text for row in df.itertuples() if row.target == 1]
    positive_examples = [clean_str(s.strip()) for s in positive_examples]
    negative_examples = [row.question_text for row in df.itertuples() if row.target == 0]
    negative_examples = [clean_str(s.strip()) for s in negative_examples]
    return positive_examples, negative_examples

def vocab(x_text):
    return set(word for question in x_text for word in question.split(' '))

def get_embedding(fname, vocab):
    embeddings = {}
    for i, line in enumerate(open(fname, 'r')):
        line_split = line.split(' ')
        word = line_split[0]
        if word in vocab:
            vec = np.array([float(s) for s in line_split[1:]], dtype='float32')
            embeddings[word] = vec
    return embeddings
    
def generate_features (question, embeddings):
    feats = np.zeros((MAX_SENTENCE_WIDTH, EMBED_DIM))
    for i, word in enumerate(question.split(" ")):
        if word in embeddings and i < MAX_SENTENCE_WIDTH:
            feats[i, :] = embeddings[word]
    return feats

df = pd.read_csv("../input/train.csv")
positive_examples, negative_examples = load_data_and_labels(df)
del df
vocab = vocab(positive_examples + negative_examples)
fname = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
embeddings = get_embedding(fname, vocab)
num_pos_all = len(positive_examples)
num_neg_all = len(negative_examples)
train_fraction = 0.99

train_data = {
    "pos": positive_examples[:int(train_fraction * num_pos_all)],
    "neg": negative_examples[:int(train_fraction * num_neg_all)]
}

dev_data = {
    "pos": positive_examples[int(train_fraction * num_pos_all):],
    "neg": negative_examples[int(train_fraction * num_neg_all):]
}
del positive_examples, negative_examples

LABEL_FRACTION = 0.5
num_pos = int(LABEL_FRACTION * SUPER_BATCH_SIZE)
num_neg = SUPER_BATCH_SIZE - num_pos

IDX = [i for i in range(SUPER_BATCH_SIZE)]
shuffle(IDX)

pos_idx = 0
neg_idx = 0

def generate_training_batch(pos_idx, neg_idx):

    pos_idxs = [(pos_idx + i) % len(train_data["pos"]) for i in range(num_pos)]
    neg_idxs = [(neg_idx + i) % len(train_data["neg"]) for i in range(num_neg)]
    x = np.zeros(shape=(SUPER_BATCH_SIZE, MAX_SENTENCE_WIDTH, EMBED_DIM))
    y = np.zeros(shape=(SUPER_BATCH_SIZE, 2))
    for i, question in enumerate([train_data["pos"][i] for i in pos_idxs]):
        x[i, :, :] = generate_features(question, embeddings)
        y[i, 1] = 1
    for i, question in enumerate([train_data["neg"][i] for i in neg_idxs]):
        x[-i - 1, :, :] = generate_features(question, embeddings)
        y[-i - 1, 0] = 1
    return x[IDX, :, :], y[IDX, :]
def generate_val_data():
    n = len(dev_data["pos"]) + len(dev_data["neg"])
    x = np.zeros(shape=(n, MAX_SENTENCE_WIDTH, EMBED_DIM))
    y = np.zeros(shape=(n, 2))
    for i, question in enumerate(dev_data["pos"]):
        x[i, :, :] = generate_features(question, embeddings)
        y[i, 1] = 1
    for i, question in enumerate(dev_data["neg"]):
        x[-i - 1, :, :] = generate_features(question, embeddings)
        y[-i - 1, 0] = 1
    return x, y

x_dev, y_dev = generate_val_data()

def define_model():
    from keras.layers import Dense, Input, Flatten
    from keras.layers import Reshape, Dropout, Concatenate
    from keras.layers import Conv1D, MaxPool1D
    from keras.models import Model
    
    inputs = Input(shape=(MAX_SENTENCE_WIDTH, EMBED_DIM), dtype='float32')

    conv_0 = Conv1D(128, kernel_size=(3), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
    conv_1 = Conv1D(128, kernel_size=(4), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
    conv_2 = Conv1D(128, kernel_size=(5), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
    
    maxpool_0 = MaxPool1D(pool_size=(5), strides=(3), padding='valid')(conv_0)
    maxpool_1 = MaxPool1D(pool_size=(5), strides=(3), padding='valid')(conv_1)
    maxpool_2 = MaxPool1D(pool_size=(5), strides=(3), padding='valid')(conv_2)
    
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.5)(flatten)
    preds = Dense(2, activation='softmax')(dropout)

    # this creates a model that includes inputs and outputs
    model = Model(inputs=inputs, outputs=preds)

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc', 'mae'])

    model.summary()
    
    return model
    

model = define_model()

ival = pos_idx + neg_idx
NUM_TRAIN_SAMPLES = num_pos_all + num_neg_all

while ival * 1. / NUM_TRAIN_SAMPLES < 2.:
    ival = pos_idx + neg_idx
    x_train, y_train = generate_training_batch(pos_idx, neg_idx)
    model.fit(x_train, y_train,
          batch_size=128,
          epochs=1,
          validation_data=(x_dev, y_dev))
    ival += SUPER_BATCH_SIZE
    pos_idx += num_pos
    neg_idx += num_neg
    print(ival * 1. / NUM_TRAIN_SAMPLES)
    
y_val_pred = model.predict(x_dev)[:,1]
y_val_bin = y_dev[:,1]
def precision(pred, true, thresh):
    n_true = sum(1 for p, t in zip(pred, true) if p > thresh and t == 1)
    n_fire = sum(1 for p, t in zip(pred, true) if p > thresh)
    return n_true * 1. / n_fire if n_fire > 0 else 1.

def recall(pred, true, thresh):
    n_true = sum(1 for p, t in zip(pred, true) if p > thresh and t == 1)
    n_all = sum(1 for p, t in zip(pred, true) if t == 1)
    return n_true * 1. / n_all if n_all > 0 else 1.

thresh = [i * 0.01 for i in range(101)]
precision_list = [precision(y_val_pred, y_val_bin, t) for t in thresh]
recall_list = [recall(y_val_pred, y_val_bin, t) for t in thresh]
f_measure = [2. * p * r / (p + r) for p, r in zip(precision_list, recall_list)]
fval, tval = max(zip(f_measure, thresh))
fval, tval
df_test = pd.read_csv("../input/test.csv")
new_vocab = set(w for row in df_test.itertuples() for w in clean_str(row.question_text.strip()).split() if w not in embeddings)

len(new_vocab)
def add_to_embedding(fname, new_vocab, embeddings):
    for i, line in enumerate(open(fname, 'r')):
        line_split = line.split(' ')
        word = line_split[0]
        if word in new_vocab:
            vec = np.array([float(s) for s in line_split[1:]], dtype='float32')
            embeddings[word] = vec
            
add_to_embedding(fname, new_vocab, embeddings)
def predict(row):
    qid = row.qid
    question = clean_str(row.question_text.strip())
    x = np.zeros((1, MAX_SENTENCE_WIDTH, EMBED_DIM))
    x[0, :, :] = generate_features(question, embeddings)
    y = 1 if model.predict(x)[0,1] > tval else 0
    return (qid, y)

tups = [predict(row) for row in df_test.itertuples()]
df_pred = pd.DataFrame(tups, columns=['qid', 'prediction'])
df_pred.to_csv('submission.csv', index=False)
fval
