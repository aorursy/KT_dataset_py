import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import re

import warnings

warnings.simplefilter('ignore')
%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, add, BatchNormalization

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import glorot_normal, orthogonal

from keras.layers import concatenate

from keras.callbacks import *

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score
train_df = pd.read_csv('../input/ccf-zgyd/train/Train_DataSet.csv')

label_df = pd.read_csv('../input/ccf-zgyd/train/Train_DataSet_Label.csv')

test_df  = pd.read_csv('../input/ccf-zgyd/test/Test_DataSet.csv')
train_df.head()
label_df.head()
test_df.head()
print(f'train shape: {train_df.shape}')

print(f'test  shape: {test_df.shape}')
label_df.label.value_counts().plot.bar()
train_df = pd.merge(train_df, label_df, on="id")

train_df.head()
# label == 0 正面评价

train_df[train_df.label == 0].head()
print(f"{train_df[train_df.label == 0].title.shape}")

print(f"{train_df[train_df.label == 0].title.unique().shape}")  # 有重复项
# label == 1 中立评价

train_df[train_df.label == 1].head()
print(f"{train_df[train_df.label == 1].title.shape}")

print(f"{train_df[train_df.label == 1].title.unique().shape}")  # 也有重复项
# label == 2 反面评价

train_df[train_df.label == 2].head()
print(f"{train_df[train_df.label == 2].title.shape}")

print(f"{train_df[train_df.label == 2].title.unique().shape}")  # 也有重复项
print(f"before drop duplicates: {train_df.shape}")

train_df.drop_duplicates("title", inplace=True)

print(f"after  drop duplicates: {train_df.shape}")
# 有文本内容甚至包含了 html



train_df.content.iloc[0]
train_df[train_df.content.isnull()].shape  # 有缺失数据
test_df[test_df.content.isnull()].shape  # 有缺失数据
train_df[train_df.title.isnull()].shape  # 也有缺失数据,但只有 1 个
train_df[train_df.title.isnull()]
train_df[train_df.title.isnull()].content.iloc[0]
train_df.loc[train_df.title.isnull(), "title"] = "William ZARIT：希望中国能更加开放投资环境"
test_df[test_df.title.isnull()].shape  # 没有缺失数据
train_df["title_length"] = train_df.title.apply(lambda x: len(x))

test_df["title_length"] = test_df.title.apply(lambda x: len(x))
train_df.title_length.describe()
train_df[train_df.title_length <= 4].sort_values(by="title_length")
train_df[train_df.title_length <= 8].label.value_counts().plot.pie()
test_df.title_length.describe()
test_df[test_df.title_length <= 4].sort_values(by="title_length")
train_df.content.fillna('')

test_df.content.fillna('')



train_df['content_length'] = train_df.content.apply(lambda i: len(str(i)))

test_df['content_length'] = test_df.content.apply(lambda i: len(str(i)))



train_df.head()
train_df.content_length.describe()
test_df.content_length.describe()
train_df["text"] = train_df.title + "：" + train_df.content

test_df["text"] = test_df.title + "：" + test_df.content
train_df.head()
test_df.head()
print(train_df.text.iloc[0])
train_df["text"] = train_df.text.apply(lambda i: " ".join(re.findall(r'[\u4e00-\u9fa5，。：“”【】《》？；、（）‘’『』「」﹃﹄〔〕—·]', str(i))))

test_df["text"] = test_df.text.apply(lambda i: " ".join(re.findall(r'[\u4e00-\u9fa5，。：“”【】《》？；、（）‘’『』「」﹃﹄〔〕—·]', str(i))))
train_df.text.iloc[0:5]
test_df.text.iloc[0:5]
max_features = 50000

max_len = 80



tokenizer = Tokenizer(num_words=max_features)



train_X = train_df["text"].values

test_X = test_df["text"].values



tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

test_X = tokenizer.texts_to_sequences(test_X)



train_X = pad_sequences(train_X, maxlen=max_len)

test_X = pad_sequences(test_X, maxlen=max_len)



train_X.shape, test_X.shape
y = train_df.label.values

y = to_categorical(y)

y.shape
X_tr, X_val, y_tr, y_val = train_test_split(train_X, y, test_size=0.15, random_state=11)
class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]

        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None

        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:

            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:

            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
def squash(x, axis=-1):

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)



    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer='glorot_uniform',

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer='glorot_uniform',

                                     trainable=True)



    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]



        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)
from keras.layers import Wrapper



class DropConnect(Wrapper):

    def __init__(self, layer, prob=1., **kwargs):

        self.prob = prob

        self.layer = layer

        super(DropConnect, self).__init__(layer, **kwargs)

        if 0. < self.prob < 1.:

            self.uses_learning_phase = True



    def build(self, input_shape):

        if not self.layer.built:

            self.layer.build(input_shape)

            self.layer.built = True

        super(DropConnect, self).build()



    def compute_output_shape(self, input_shape):

        return self.layer.compute_output_shape(input_shape)



    def call(self, x):

        if 0. < self.prob < 1.:

            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)

            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)

        return self.layer.call(x)
def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def build_my_model():

    inp = Input(shape=(max_len,))

    x = Embedding(max_features, 100, trainable=True)(inp)

    x = SpatialDropout1D(rate=0.2)(x)

    x = Bidirectional(CuDNNLSTM(128, 

                                return_sequences=True, 

                                kernel_initializer=glorot_normal(seed=1029), 

                                recurrent_initializer=orthogonal(gain=1.0, seed=1029)))(x)



    x_1 = Attention(max_len)(x)

    x_1 = DropConnect(Dense(32, activation="relu"), prob=0.1)(x_1)

    

    x_2 = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)

    x_2 = Flatten()(x_2)

    x_2 = DropConnect(Dense(32, activation="relu"), prob=0.1)(x_2)



    conc = concatenate([x_1, x_2])

    outp = Dense(3, activation="softmax")(conc)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[f1])

    return model
model = build_my_model()

model.summary()
file_path = "best_model.h5"

check_point = ModelCheckpoint(file_path, monitor="val_f1", verbose=1, save_weights_only=True,

                              save_best_only=True, mode="max")

early_stop = EarlyStopping(monitor="val_f1", mode="max", patience=10)



model.fit(X_tr, 

          y_tr, 

          validation_data=(X_val, y_val), 

          epochs=40, 

          batch_size=512, 

          callbacks=[check_point, early_stop])
model.load_weights('best_model.h5')
preds = model.predict(test_X, batch_size=1024)

preds = np.argmax(preds, axis=1)

preds.shape
submission = pd.read_csv("../input/ccf-zgyd/submit_example.csv")

submission.head()
submission.label = preds

submission.to_csv("submission.csv", index=False)

submission.head()