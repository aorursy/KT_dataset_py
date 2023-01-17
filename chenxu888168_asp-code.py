# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

# @Time    : 2019/10/30 14:41

# @Author  : 陈旭

# @Software: PyCharm

# @File    : data_loader.py

import os

import pickle



import numpy as np

import pandas as pd

from gensim.models import KeyedVectors

from tensorflow import keras

from tensorflow.keras.utils import to_categorical



root_path = "/kaggle/input/"
class DataSet():



    def __init__(self, data_path):

        self.data_path = data_path

        self.train_path = os.path.join(self.data_path, "train.csv")

        self.test_path = os.path.join(self.data_path, "test.csv")

        self.train_x = None

        self.train_y = None

        self.test_x = None

        self.test_y = None

        self._genarate_data()



    def _genarate_data(self):

        if self.train_x == None:

            self.train_x, self.train_y = self._read_data(self.train_path)

        if self.test_x == None:

            self.test_x, self.test_y = self._read_data(self.test_path)



    def _read_data(self, data_path):

        """读取文件，保存为x和y"""

        df = pd.read_csv(data_path, delimiter=";", header=None)

        df.columns = ["text", "target", "label"]

        self.x = df[["text", "target"]]

        self.y = df["label"]

        return self.x, self.y



    def get_train_data(self):

        return self.train_x, self.train_y



    def get_test_data(self):

        return self.test_x, self.test_y


class DataLoader():



    def __init__(self, dataset: DataSet, dis_len, max_len=None):

        self.dataset = dataset

        self.data_path = dataset.data_path

        self.word_index = {}

        self.train_x_ids = None

        self.train_y_ids = None

        self.tran_x_dict = None

        self.train_data_dict = None

        self.test_data_dict = None

        self.label_index = {

            "conflict": 3,

            "negative": 2,

            "neutral": 1,

            "positive": 0,

        }

        self.max_len = max_len

        self.dis_len = dis_len

        self.w2v_path = root_path + "data/restaurant/rest_2014_w2v.bin"

        self.w2v = KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)

        self.w2v["[PAD]"] = np.array([0.0] * 300)

        self.w2v["[UNK]"] = np.array([0.9] * 300)  # 随机生成



    def _get_word_set(self, data: pd.DataFrame):

        """给定数据，把text的所有单词加入到一个集合中，作为所有出现的单词"""

        word_set = set()

        for each_text in data["text"]:

            word_list = each_text.split()

            for each_word in word_list:

                word_set.add(each_word.strip())

        return word_set



    def _generate_word_index(self, dataset: DataSet):

        """根据给定的dataset，生成对用的word_index映射字典"""

        train_x_word_set = self._get_word_set(dataset.train_x)

        test_x_word_set = self._get_word_set(dataset.test_x)

        all_word_set = train_x_word_set.union(test_x_word_set)

        word_index = {

            "[PAD]": 0,

            "[UNK]": 1

        }

        for index, each_word in enumerate(all_word_set):

            word_index[each_word] = index + 2

        return word_index



    def get_word_index(self):

        """获取单词映射矩阵，如果不存在对应的pkl文件，则调用方法生成对应的word_index并保存"""

        word_index_path = os.path.join(self.data_path, "word_index.pkl")

        try:

            with open(word_index_path, "rb") as fout:

                word_index = pickle.load(fout)

                return word_index

        except Exception as e:

            print(e)

            word_index = self._generate_word_index(self.dataset)

            with open(word_index_path, "wb") as fout:

                pickle.dump(word_index, fout)

            return word_index



    def genarate_sen_ids(self, data: pd.DataFrame):

        """传入pd格式的数据，生成对应的text的ids返回"""

        if self.word_index == {}:

            self.word_index = self.get_word_index()



        all_sample_ids = []

        all_sample = []

        all_sample_w2v = []

        for each_sample in data.values:

            each_text = each_sample[0]

            each_target = each_sample[1]

            each_text = each_text.replace("$T$", each_target)

            word_list = each_text.split()

            word_list = [c for c in word_list if c.strip() != ""]

            sample_ids = []

            sample = []

            for each_word in word_list:

                sample_ids.append(self.word_index.get(each_word.lower().strip()))

                if each_word.lower().strip() in self.w2v:

                    sample.append(each_word.lower().strip())

                else:

                    sample.append("[UNK]")

            sample = sample + ["[PAD]"] * (self.max_len - len(sample))

            # print(sample)

            all_sample_w2v.append(self.w2v[sample])

            all_sample_ids.append(sample_ids)

            all_sample.append(sample)

        return all_sample, all_sample_ids, all_sample_w2v



    def genarate_dis(self, data: pd.DataFrame):

        """产生距离的表征"""

        all_sample_dis = []

        for each_sample in data.values:

            each_text = each_sample[0]

            each_target = each_sample[1]

            word_list = each_text.split()

            word_list = [c for c in word_list if c.strip() != ""]

            sample_dis = []

            try:

                target_index = word_list.index("$t$")

            except:

                print(each_text)

            for index, each_word in enumerate(word_list):

                if index == target_index:

                    target_word_list = each_target.split()

                    sample_dis.extend([self.dis_len] * len(target_word_list))

                sample_dis.append(index - target_index + self.max_len)

            all_sample_dis.append(sample_dis)

        return all_sample_dis



    def get_y_list(self, data: pd.DataFrame):

        """给定数据，输出对应的y_list"""

        y_list = []

        for each in data.values:

            y_list.append(self.label_index[each.strip()])

        return y_list



    def get_train_data_dict(self):

        """获取训练数据，包括很多特征"""

        if self.train_data_dict == None:

            # todo 在这里加入更多的拓展特征

            all_sample, all_sample_ids, all_sample_w2v = self.genarate_sen_ids(self.dataset.train_x)  # 每一个样本转化为固定的序列表示

            all_sample_dis = self.genarate_dis(self.dataset.train_x)  # 每一个样本转化为固定的距离表示

            y_list = self.get_y_list(self.dataset.train_y)

            if self.max_len != None:

                all_sample_ids = keras.preprocessing.sequence.pad_sequences(all_sample_ids, maxlen=self.max_len,

                                                                            value=self.word_index["[PAD]"],

                                                                            padding="post")

                all_sample_dis = keras.preprocessing.sequence.pad_sequences(all_sample_dis, maxlen=self.max_len,

                                                                            value=90, padding="post")



            # all_sample_bert_w2v = extract_embeddings(model_path, all_sample)

            all_sample_bert_w2v = None

            self.train_data_dict = {

                u"all_sample_ids": all_sample_ids,

                "all_sample_dis": all_sample_dis,

                "all_sample_w2v": np.array(all_sample_w2v, dtype=np.float32),

                "all_sample": all_sample,

                "all_sample_bert_w2v": all_sample_bert_w2v,

                u"y_list": to_categorical(y_list)

            }

        return self.train_data_dict



    def get_test_data_dict(self):

        """获取测试数据，包括很多特征"""

        if self.test_data_dict == None:

            # todo 在这里加入更多的拓展特征

            all_sample, all_sample_ids, all_sample_w2v = self.genarate_sen_ids(self.dataset.test_x)  # 每一个样本转化为固定的序列表示

            all_sample_dis = self.genarate_dis(self.dataset.test_x)  # 每一个样本转化为固定的距离表示

            y_list = self.get_y_list(self.dataset.test_y)

            if self.max_len != None:

                all_sample_ids = keras.preprocessing.sequence.pad_sequences(all_sample_ids, maxlen=self.max_len,

                                                                            value=self.word_index["[PAD]"],

                                                                            padding="post")

                all_sample_dis = keras.preprocessing.sequence.pad_sequences(all_sample_dis, maxlen=self.max_len,

                                                                            value=self.dis_len, padding="post")

            # all_sample_bert_w2v = extract_embeddings(model_path, all_sample)

            all_sample_bert_w2v = None

            self.test_data_dict = {

                u"all_sample_ids": all_sample_ids,

                "all_sample_dis": all_sample_dis,

                "all_sample_w2v": np.array(all_sample_w2v, dtype=np.float32),

                "all_sample": all_sample,

                "all_sample_bert_w2v": all_sample_bert_w2v,

                "y_list": to_categorical(y_list)

            }

        return self.test_data_dict
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras import layers

from tensorflow.keras.layers import Flatten, Dropout, Concatenate

from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, Lambda, Layer, GRU

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam



def dilated_gated_conv1d(seq, mask, dilation_rate=1):

    """膨胀门卷积（残差式）

    """

    dim = K.int_shape(seq)[-1]

    h = Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate)(seq)



    def _gate(x):

        dropout_rate = 0.1

        s, h = x

        g, h = h[:, :, :dim], h[:, :, dim:]

        g = K.in_train_phase(K.dropout(g, dropout_rate), g)

        g = K.sigmoid(g)

        return g * s + (1 - g) * h



    seq = Lambda(_gate)([seq, h])

    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])

    return seq





def sequence_masking(x, mask, mode=0, axis=None, heads=1):

    """为序列条件mask的函数

    mask: 形如(batch_size, seq_len)的0-1矩阵；

    mode: 如果是0，则直接乘以mask；

          如果是1，则在padding部分减去一个大正数。

    axis: 序列所在轴，默认为1；

    heads: 相当于batch这一维要被重复的次数。

    """

    if mask is None or mode not in [0, 1]:

        return x

    else:

        if heads is not 1:

            mask = K.expand_dims(mask, 1)

            mask = K.tile(mask, (1, heads, 1))

            mask = K.reshape(mask, (-1, K.shape(mask)[2]))

        if axis is None:

            axis = 1

        if axis == -1:

            axis = K.ndim(x) - 1

        assert axis > 0, 'axis muse be greater than 0'

        for _ in range(axis - 1):

            mask = K.expand_dims(mask, 1)

        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):

            mask = K.expand_dims(mask, K.ndim(mask))

        if mode == 0:

            return x * mask

        else:

            return x - (1 - mask) * 1e12





class MultiHeadAttention(Layer):

    """多头注意力机制

    """



    def __init__(self, heads, head_size, key_size=None, **kwargs):

        super(MultiHeadAttention, self).__init__(**kwargs)

        self.heads = heads

        self.head_size = head_size

        self.out_dim = heads * head_size

        self.key_size = key_size if key_size else head_size



    def build(self, input_shape):

        super(MultiHeadAttention, self).build(input_shape)

        self.q_dense = Dense(self.key_size * self.heads)

        self.k_dense = Dense(self.key_size * self.heads)

        self.v_dense = Dense(self.out_dim)

        self.o_dense = Dense(self.out_dim)



    def call(self, inputs, q_mask=False, v_mask=False, a_mask=False):

        """实现多头注意力

        q_mask: 对输入的query序列的mask。

                主要是将输出结果的padding部分置0。

        v_mask: 对输入的value序列的mask。

                主要是防止attention读取到padding信息。

        a_mask: 对attention矩阵的mask。

                不同的attention mask对应不同的应用。

        """

        q, k, v = inputs[:3]

        # 处理mask

        idx = 3

        if q_mask:

            q_mask = inputs[idx]

            idx += 1

        else:

            q_mask = None

        if v_mask:

            v_mask = inputs[idx]

            idx += 1

        else:

            v_mask = None

        if a_mask:

            if len(inputs) > idx:

                a_mask = inputs[idx]

            else:

                a_mask = 'history_only'

        else:

            a_mask = None

        # 线性变换

        qw = self.q_dense(q)

        kw = self.k_dense(k)

        vw = self.v_dense(v)

        # 形状变换

        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))

        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))

        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))

        # 维度置换

        qw = K.permute_dimensions(qw, (0, 2, 1, 3))

        kw = K.permute_dimensions(kw, (0, 2, 1, 3))

        vw = K.permute_dimensions(vw, (0, 2, 1, 3))

        # 转为三阶张量

        qw = K.reshape(qw, (-1, K.shape(q)[1], self.key_size))

        kw = K.reshape(kw, (-1, K.shape(k)[1], self.key_size))

        vw = K.reshape(vw, (-1, K.shape(v)[1], self.head_size))

        # Attention

        a = K.batch_dot(qw, kw, [2, 2]) / self.key_size ** 0.5

        a = sequence_masking(a, v_mask, 1, -1, self.heads)

        if a_mask is not None:

            if a_mask == 'history_only':

                ones = K.ones_like(a[:1])

                a_mask = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e12

                a = a - a_mask

            else:

                a = a - (1 - a_mask) * 1e12

        a = K.softmax(a)

        # 完成输出

        o = K.batch_dot(a, vw, [2, 1])

        o = K.reshape(o, (-1, self.heads, K.shape(q)[1], self.head_size))

        o = K.permute_dimensions(o, (0, 2, 1, 3))

        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))

        o = self.o_dense(o)

        o = sequence_masking(o, q_mask, 0)

        return o



    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], input_shape[0][1], self.out_dim)



    def get_config(self):

        config = {

            'heads': self.heads,

            'head_size': self.head_size,

            'key_size': self.key_size

        }

        base_config = super(MultiHeadAttention, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
def DGCNN_DIS(inputlen, vocabulary, num_class, dis_len,

              w2v_len=300,

              embedding_dim=128,

              num_filters=512,

              filter_sizes=[3, 4, 5],

              drop_rate=0.5,

              lr=1e-4):

    inputs_dis = Input(shape=(inputlen,))

    inputs_w2v = Input(shape=(inputlen, w2v_len))

    inputs_ids = Input(shape=(inputlen,))



    mask = Lambda(lambda x: K.cast(

        K.greater(K.expand_dims(x, 2), 0), 'float32'))(inputs_ids)

    embedding_dis = Embedding(input_dim=dis_len,

                              output_dim=embedding_dim,

                              input_length=inputlen)(inputs_dis)



    # reshape_w2v = Reshape((inputlen, w2v_len, 1))(inputs_w2v)

    # reshape_dis = Reshape((inputlen, embedding_dim, 1))(embedding_dis)



    t = Concatenate(axis=2)([inputs_w2v, embedding_dis])

    t = Lambda(lambda x: x[0] * x[1])([t, mask])

    t = dilated_gated_conv1d(t, mask, 1)

    t = dilated_gated_conv1d(t, mask, 2)

    t = dilated_gated_conv1d(t, mask, 5)

    t = dilated_gated_conv1d(t, mask, 1)

    t = dilated_gated_conv1d(t, mask, 2)

    t = dilated_gated_conv1d(t, mask, 5)

    t = dilated_gated_conv1d(t, mask, 1)

    t = dilated_gated_conv1d(t, mask, 2)

    t = dilated_gated_conv1d(t, mask, 5)

    t = dilated_gated_conv1d(t, mask, 1)

    t = dilated_gated_conv1d(t, mask, 1)

    t = dilated_gated_conv1d(t, mask, 1)

    h = MultiHeadAttention(8, 16)([t, t, t, mask])

    h_dim = K.int_shape(h)[-1]

    h = layers.Reshape(target_shape=(inputlen, h_dim))(h)

    lstm_ids = GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(inputs_w2v)

    h = Concatenate()([t, h, lstm_ids])

    conv_0 = Conv1D(embedding_dim,

                    filter_sizes[0], activation='relu', padding='same')(h)

    conv_1 = Conv1D(embedding_dim,

                    filter_sizes[0], activation='relu', padding='same')(h)

    conv_2 = Conv1D(embedding_dim,

                    filter_sizes[0], activation='relu', padding='same')(h)



    # maxpool_0 = MaxPool1D(pool_size=inputlen - filter_sizes[0] + 1,

    #                       padding='valid')(conv_0)

    # maxpool_1 = MaxPool1D(pool_size=inputlen - filter_sizes[1] + 1,

    #                       padding='valid')(conv_1)

    # maxpool_2 = MaxPool1D(pool_size=inputlen - filter_sizes[2] + 1,

    #                       padding='valid')(conv_2)



    concatenated_tensor = Concatenate(axis=-1)(

        [conv_0, conv_1, conv_2])



    flatten = Flatten()(concatenated_tensor)

    dropout = Dropout(drop_rate)(flatten)



    output = Dense(units=num_class, activation='softmax')(dropout)



    model = Model(inputs=[inputs_w2v, inputs_dis, inputs_ids], outputs=output)

    model.compile(optimizer=Adam(lr=lr),

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model





def get_train_batch(X_train, y_train, batch_size):

    """



    :param X_train:

    :param y_train:

    :param batch_size:

    :return:

    """

    train_sample_w2v, train_sample_dis = X_train

    num = len(train_sample_w2v)

    while 1:

        for i in range(0, len(X_train), batch_size):

            start = i

            end = min(num, (i + 1) * batch_size)

            sample_w2v = train_sample_w2v[start:end]

            sample_dis = train_sample_dis[start:end]

            y = y_train[start:end]

            yield [sample_w2v, sample_dis], y
def run_main():

    # 第一步, 定义参数

    max_len = 100

    batch_size = 16

    epochs = 10

    vocabulary = 4550

    num_class = 4

    dis_len = 100

    # 第二步，读取数据

    dataset = DataSet(data_path= root_path + "/data/restaurant/")

    train_x, train_y = dataset.get_train_data()

    test_x, test_y = dataset.get_test_data()

    data_loader = DataLoader(dataset=dataset, dis_len=dis_len, max_len=max_len)



    # train_data

    train_data_dict = data_loader.get_train_data_dict()

    train_sample_ids = train_data_dict["all_sample_ids"]

    train_sample_w2v = train_data_dict["all_sample_w2v"]

    train_sample_dis = train_data_dict["all_sample_dis"]

    train_y_list = train_data_dict["y_list"]

    # print(train_sample_w2v.shape)

    # test_data

    test_data_dict = data_loader.get_test_data_dict()

    test_sample_ids = test_data_dict["all_sample_ids"]

    test_sample_w2v = test_data_dict["all_sample_w2v"]

    test_sample_dis = test_data_dict["all_sample_dis"]

    test_y_list = test_data_dict["y_list"]



    # 第三步，构建模型

    model = DGCNN_DIS(max_len, vocabulary, num_class, dis_len=2 * dis_len)

    model.summary()

    tf.keras.utils.plot_model(model, show_shapes=True, to_file="DGCNN-dis.png")

    model.fit([train_sample_w2v, train_sample_dis, train_sample_ids], train_y_list,

              batch_size=batch_size,

              epochs=epochs,

              validation_data=([test_sample_w2v, test_sample_dis, test_sample_ids], test_y_list))

#     model.evaluate([test_sample_w2v, test_sample_dis, test_sample_ids], test_y_list, batch_size=batch_size)
run_main()