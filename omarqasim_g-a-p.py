# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import gc

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.t
DATA_ROOT = '../input/'

GAP_DATA_FOLDER = os.path.join(DATA_ROOT, 'gap-coreference')

SUB_DATA_FOLDER = os.path.join(DATA_ROOT, 'gendered-pronoun-resolution')

FAST_TEXT_DATA_FOLDER = os.path.join(DATA_ROOT, 'fasttext-crawl-300d-2m')
test_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-test.tsv')

train_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-development.tsv')

dev_df_path = os.path.join(GAP_DATA_FOLDER, 'gap-validation.tsv')

sub_test_df_path = os.path.join(SUB_DATA_FOLDER, 'test_stage_2.tsv')



train_df = pd.read_csv(train_df_path, sep='\t')

test_df = pd.read_csv(test_df_path, sep='\t')

dev_df = pd.read_csv(dev_df_path, sep='\t')

sub_test_df = pd.read_csv(sub_test_df_path, sep='\t')



#pd.options.display.max_colwidth = 1000
train_df.head()
from spacy.lang.en import English

from spacy.pipeline import DependencyParser

import spacy

from nltk import Tree
nlp = spacy.load('en_core_web_lg')



# build a tree

def to_nltk_tree(node):

    if node.n_lefts + node.n_rights > 0:

        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])

    else:

        return node.orth_

# binary search for a target_

def bs(list_, target_):

    lo, hi = 0, len(list_) -1

    

    while lo < hi:

        mid = lo + int((hi - lo) / 2)

        

        if target_ < list_[mid]:

            hi = mid

        elif target_ > list_[mid]:

            lo = mid + 1

        else:

            return mid + 1

    return lo



# get k preceding words starting from offest

def _get_preceding_words(tokens, offset, k):

    start = offset - k

    

    precedings = [None] * max(0, 0-start)

    start = max(0, start)

    precedings += tokens[start: offset]

    

    return precedings



# get k following words starting from offest

def _get_following_words(tokens, offset, k):

    end = offset + k

    

    followings = [None] * max(0, end - len(tokens))

    end = min(len(tokens), end)

    followings += tokens[offset: end]

    

    return followings

        



def extrac_embed_features_tokens(text, char_offset):

    doc = nlp(text)

    

    # char offset to token offset

    lens = [token.idx for token in doc] # list if indices of the start of each token in the doc 

    mention_offset = bs(lens, char_offset) - 1

    # mention_word

    mention = doc[mention_offset]

    

    # token offset to sentence offset

    lens = [len(sent) for sent in doc.sents]

    acc_lens = [len_ for len_ in lens]

    pre_len = 0

    for i in range(0, len(acc_lens)):

        pre_len += acc_lens[i]

        acc_lens[i] = pre_len

    sent_index = bs(acc_lens, mention_offset)

    # mention sentence

    sent = list(doc.sents)[sent_index]

    

    # dependency parent

    head = mention.head

    

    # last word and first word

    first_word, last_word = sent[0], sent[-2]

    

    assert mention_offset >= 0

    

    # two preceding words and two following words

    tokens = list(doc)

    precedings2 = _get_preceding_words(tokens, mention_offset, 2)

    followings2 = _get_following_words(tokens, mention_offset, 2)

    

    # five preceding words and five following words

    precedings5 = _get_preceding_words(tokens, mention_offset, 5)

    followings5 = _get_following_words(tokens, mention_offset, 5)

    

    # sentence words

    sent_tokens = [token for token in sent]

    

    return mention, head, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens
print("Texts: ")

text = u"Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline."

print(text)



print("\nDependency parsing trees: ")

doc = nlp(text)

[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]



print("\nFeatures:")

mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens = extrac_embed_features_tokens(text, 274)

features = pd.Series([str(feature) for feature in (mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens)], index=['mention', 'parent', 'first_word', 'last_word', 'precedings2', 'followings2', 'precedings5', 'followings5', 'sent_tokens'])

features
num_embed_features = 11

embed_dim = 300
def create_embedding_features(df, text_column, offset_column):

    text_offset_list = df[[text_column, offset_column]].values.tolist()

    num_features = num_embed_features

    

    embed_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features, embed_dim))

    for text_offset_index in range(len(text_offset_list)):

        text_offset = text_offset_list[text_offset_index]

        mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens = extrac_embed_features_tokens(text_offset[0], text_offset[1])

        

        feature_index = 0

        embed_feature_matrix[text_offset_index, feature_index, :] = mention.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = parent.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = first_word.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = last_word.vector

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings2])

        feature_index += len(precedings2)

        embed_feature_matrix[text_offset_index, feature_index:feature_index+2, :] = np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings2])

        feature_index += len(followings2)

        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings5]), axis=0)

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings5]), axis=0)

        feature_index += 1

        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(np.asarray([token.vector for token in sent_tokens]), axis=0) if len(sent_tokens) > 0 else np.zeros(embed_dim)

        feature_index += 1

    

    return embed_feature_matrix
def bs_(list_, target_):

    lo, hi = 0, len(list_) -1

    

    while lo < hi:

        mid = lo + int((hi - lo) / 2)

        

        if target_ < list_[mid]:

            hi = mid

        elif target_ > list_[mid]:

            lo = mid + 1

        else:

            return mid

    return lo



# one hot encoding distance

def ohe_dist(dist, buckets):

    idx = bs_(buckets, dist)

    oh = np.zeros(shape=(len(buckets),), dtype=np.float32)

    oh[idx] = 1

    

    return oh
def extrac_positional_features(text, char_offset1, char_offset2):

    doc = nlp(text)

    max_len = 64

    

    # char offset to token offset

    lens = [token.idx for token in doc]

    mention_offset1 = bs(lens, char_offset1) - 1

    mention_offset2 = bs(lens, char_offset2) - 1

    

    # token offset to sentence offset

    lens = [len(sent) for sent in doc.sents]

    acc_lens = [len_ for len_ in lens]

    pre_len = 0

    for i in range(0, len(acc_lens)):

        pre_len += acc_lens[i]

        acc_lens[i] = pre_len

    sent_index1 = bs(acc_lens, mention_offset1)

    sent_index2 = bs(acc_lens, mention_offset2)

    

    sent1 = list(doc.sents)[sent_index1]

    sent2 = list(doc.sents)[sent_index2]

    

    # buckets

    bucket_dist = [1, 2, 3, 4, 5, 8, 16, 32, 64]

    

    # relative distance

    dist = mention_offset2 - mention_offset1

    dist_oh = ohe_dist(dist, bucket_dist)

    

    # buckets

    bucket_pos = [0, 1, 2, 3, 4, 5, 8, 16, 32]

    

    # absolute position in the sentence

    # position of the first mention from both sides of the sentence

    sent_pos1 = mention_offset1 + 1

    if sent_index1 > 0:

        sent_pos1 = mention_offset1 - acc_lens[sent_index1-1]

    sent_pos_oh1 = ohe_dist(sent_pos1, bucket_pos)

    sent_pos_inv1 = len(sent1) - sent_pos1

    assert sent_pos_inv1 >= 0

    sent_pos_inv_oh1 = ohe_dist(sent_pos_inv1, bucket_pos)

    

    # position of the second mention from both sides of the sentence

    sent_pos2 = mention_offset2 + 1

    if sent_index2 > 0:

        sent_pos2 = mention_offset2 - acc_lens[sent_index2-1]

    sent_pos_oh2 = ohe_dist(sent_pos2, bucket_pos)

    sent_pos_inv2 = len(sent2) - sent_pos2

    if sent_pos_inv2 < 0:

        print(sent_pos_inv2)

        print(len(sent2))

        print(sent_pos2)

        raise ValueError

    sent_pos_inv_oh2 = ohe_dist(sent_pos_inv2, bucket_pos)

    

    sent_pos_ratio1 = sent_pos1 / len(sent1)

    sent_pos_ratio2 = sent_pos2 / len(sent2)

    

    return dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2
text = 'He admitted making four trips to China and playing golf there. He also admitted that ZTE officials, whom he says are his golf buddies, hosted and paid for the trips. Jose de Venecia III, son of House Speaker Jose de Venecia Jr, alleged that Abalos offered him US$10 million to withdraw his proposal on the NBN project.'

dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2 = extrac_positional_features(text, 256, 208)

features = pd.Series([str(feature) for feature in (dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2)], index=['dist_oh', 'sent_pos_oh1', 'sent_pos_oh2', 'sent_pos_inv_oh1', 'sent_pos_inv_oh2'])

features
num_pos_features = 45
def create_dist_features(df, text_column, pronoun_offset_column, name_offset_column):

    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()

    num_features = num_pos_features

    

    pos_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features))

    for text_offset_index in range(len(text_offset_list)):

        text_offset = text_offset_list[text_offset_index]

        dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2 = extrac_positional_features(text_offset[0], text_offset[1], text_offset[2])

        

        feature_index = 0

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(dist_oh)] = np.asarray(dist_oh)

        feature_index += len(dist_oh)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh1)] = np.asarray(sent_pos_oh1)

        feature_index += len(sent_pos_oh1)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh2)] = np.asarray(sent_pos_oh2)

        feature_index += len(sent_pos_oh2)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh1)] = np.asarray(sent_pos_inv_oh1)

        feature_index += len(sent_pos_inv_oh1)

        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh2)] = np.asarray(sent_pos_inv_oh2)

        feature_index += len(sent_pos_inv_oh2)

    

    return pos_feature_matrix
p_emb_tra = create_embedding_features(train_df, 'Text', 'Pronoun-offset')

p_emb_dev = create_embedding_features(dev_df, 'Text', 'Pronoun-offset')

p_emb_test = create_embedding_features(test_df, 'Text', 'Pronoun-offset')

p_emb_sub_test = create_embedding_features(sub_test_df, 'Text', 'Pronoun-offset')



a_emb_tra = create_embedding_features(train_df, 'Text', 'A-offset')

a_emb_dev = create_embedding_features(dev_df, 'Text', 'A-offset')

a_emb_test = create_embedding_features(test_df, 'Text', 'A-offset')

a_emb_sub_test = create_embedding_features(sub_test_df, 'Text', 'A-offset')



b_emb_tra = create_embedding_features(train_df, 'Text', 'B-offset')

b_emb_dev = create_embedding_features(dev_df, 'Text', 'B-offset')

b_emb_test = create_embedding_features(test_df, 'Text', 'B-offset')

b_emb_sub_test = create_embedding_features(sub_test_df, 'Text', 'B-offset')



pa_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'A-offset')

pa_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'A-offset')

pa_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'A-offset')

pa_pos_sub_test = create_dist_features(sub_test_df, 'Text', 'Pronoun-offset', 'A-offset')



pb_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'B-offset')

pb_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'B-offset')

pb_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'B-offset')

pb_pos_sub_test = create_dist_features(sub_test_df, 'Text', 'Pronoun-offset', 'B-offset')
def _row_to_y(row):

    if row.loc['A-coref']:

        return 0

    if row.loc['B-coref']:

        return 1

    return 2



y_tra = train_df.apply(_row_to_y, axis=1)

y_dev = dev_df.apply(_row_to_y, axis=1)

y_test = test_df.apply(_row_to_y, axis=1)
X_train = [p_emb_tra, a_emb_tra, b_emb_tra, pa_pos_tra, pb_pos_tra]

X_dev = [p_emb_dev, a_emb_dev, b_emb_dev, pa_pos_dev, pb_pos_dev]

X_test = [p_emb_test, a_emb_test, b_emb_test, pa_pos_test, pb_pos_test]

X_sub_test = [p_emb_sub_test, a_emb_sub_test, b_emb_sub_test, pa_pos_sub_test, pb_pos_sub_test]
import numpy as np

from keras import backend

from keras import layers

from keras import models
def build_mlp_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    model_dim, mlp_dim, mlp_depth=1, drop_out=0.5, return_customized_layers=False):

    """

    Create A Multi-Layer Perceptron Model.

    

    inputs: 

        embeddings: [batch, num_embed_feature, embed_dims] * 3 ## pronoun, A, B

        positional_features: [batch, num_pos_feature] * 2 ## pronoun-A, pronoun-B

        

    outputs: 

        [batch, num_classes] # in our case there should be 3 output classes: A, B, None

        

    :param output_dim: the output dimension size

    :param model_dim: rrn dimension size

    :param mlp_dim: the dimension size of fully connected layer

    :param mlp_depth: the depth of fully connected layers

    :param drop_out: dropout rate of fully connected layers

    :param return_customized_layers: boolean, default=False

        If True, return model and customized object dictionary, otherwise return model only

    :return: keras model

    """

    

    def _mlp_channel1(feature_dropout_layer, feature_map_layer, flatten_layer, x):

        x = feature_dropout_layer(x)

        x = feature_map_layer(x)

        x = flatten_layer(x)

        return x

    

    def _mlp_channel2(feature_map_layer, x):

        x = feature_map_layer(x)

        return x



    # inputs

    inputs1 = list()

    for fi in range(num_feature_channels1):

        inputs1.append(models.Input(shape=(num_features1, feature_dim1), dtype='float32', name='input1_' + str(fi)))

        

    print('inputs1 ', inputs1)

    inputs2 = list()

    for fi in range(num_feature_channels2):

        inputs2.append(models.Input(shape=(num_features2, ), dtype='float32', name='input2_' + str(fi)))

    

    # define feature map layers

    # MLP Layers

    feature_dropout_layer1 = layers.TimeDistributed(layers.Dropout(rate=drop_out, name="input_dropout_layer"))

    feature_map_layer1 = layers.TimeDistributed(layers.Dense(model_dim, name="feature_map_layer1", activation="relu"))

    flatten_layer1 = layers.Flatten(name="feature_flatten_layer1")

    feature_map_layer2 = layers.Dense(model_dim, name="feature_map_layer2", activation="relu")

    

    print('feature_dropout_layer1 ', feature_dropout_layer1)

    x1 = [_mlp_channel1(feature_dropout_layer1, feature_map_layer1, flatten_layer1, input_) for input_ in inputs1]

    x2 = [_mlp_channel2(feature_map_layer2, input_) for input_ in inputs2]

    

    print('x1+x2 ', x1+x2 )

    x = layers.Concatenate(axis=1, name="concate_layer")(x1+x2)

    print('x ', x)

    # MLP Layers

    x = layers.BatchNormalization(name='batch_norm_layer')(x)

    x = layers.Dropout(rate=drop_out, name="dropout_layer")(x)

        

    for i in range(mlp_depth - 1):

        x = layers.Dense(mlp_dim, activation='selu', kernel_initializer='lecun_normal', name='selu_layer' + str(i))(x)

        x = layers.AlphaDropout(drop_out, name='alpha_layer' + str(i))(x)



    outputs = layers.Dense(output_dim, activation="softmax", name="softmax_layer0")(x)



    model = models.Model(inputs1 + inputs2, outputs)



    if return_customized_layers:

        return model, {}



    return model
num_feature_channels1 = 3

num_feature_channels2 = 2



num_embed_features = 11

num_features1 = num_embed_features

num_features2 = num_pos_features

feature_dim1 = embed_dim

output_dim = 3

model_dim = 10 

mlp_dim = 60

mlp_depth=1

drop_out=0.5

return_customized_layers=True



model, co_mlp = build_mlp_model(

    num_feature_channels1, num_feature_channels2, num_features1, num_features2, feature_dim1, output_dim, 

    model_dim, mlp_dim, mlp_depth, drop_out, return_customized_layers

)
from keras import callbacks as kc

from keras import optimizers as ko

from keras import initializers, regularizers, constraints



import matplotlib.pyplot as plt

from IPython.display import SVG





histories = list()
print(model.summary())

from keras.utils import plot_model

plot_model(model, to_file='model.png')
from PIL import Image

img = Image.open('model.png')

display(img)
adam = ko.Nadam()

model.compile(adam, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])



file_path = "best_mlp_model.hdf5"

check_point = kc.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")

early_stop = kc.EarlyStopping(monitor = "val_loss", mode = "min", patience=3)

history = model.fit(X_train, y_tra, batch_size=20, epochs=20, validation_data=(X_dev, y_dev), callbacks = [check_point, early_stop])



histories.append(np.min(np.asarray(history.history['val_loss'])))



# del model, history

gc.collect()

history.history['val_sparse_categorical_accuracy']
loss, score = model.evaluate(x=X_test, y=y_test)

print('model evalutation loss = {} and score = {}'.format(loss, score))


y_sub_preds = model.predict(X_sub_test, batch_size = 1024, verbose = 1)



sub_df_path = os.path.join(SUB_DATA_FOLDER, 'sample_submission_stage_2.csv')

sub_df = pd.read_csv(sub_df_path)

sub_df.loc[:, 'A'] = pd.Series(y_sub_preds[:, 0])

sub_df.loc[:, 'B'] = pd.Series(y_sub_preds[:, 1])

sub_df.loc[:, 'NEITHER'] = pd.Series(y_sub_preds[:, 2])



sub_df.head()

sub_df
sub_df.to_csv("submission.csv", index=False)