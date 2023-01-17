import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import librosa

import matplotlib.pyplot as plt

import gc



from tqdm import tqdm, tqdm_notebook

from sklearn.metrics import label_ranking_average_precision_score

from sklearn.model_selection import train_test_split



tqdm.pandas()
def calculate_overall_lwlrap_sklearn(truth, scores):

    """Calculate the overall lwlrap using sklearn.metrics.lrap."""

    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.

    sample_weight = np.sum(truth > 0, axis=1)

    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

    overall_lwlrap = label_ranking_average_precision_score(

        truth[nonzero_weight_sample_indices, :] > 0, 

        scores[nonzero_weight_sample_indices, :], 

        sample_weight=sample_weight[nonzero_weight_sample_indices])

    return overall_lwlrap
def split_and_label(rows_labels):

    

    row_labels_list = []

    for row in rows_labels:

        row_labels = row.split(',')

        labels_array = np.zeros((80))

        

        for label in row_labels:

            index = label_mapping[label]

            labels_array[index] = 1

        

        row_labels_list.append(labels_array)

    

    return row_labels_list
train_curated = pd.read_csv('../input/train_curated.csv')

train_noisy = pd.read_csv('../input/train_noisy.csv')

test = pd.read_csv('../input/sample_submission.csv')
print(train_curated.shape, train_noisy.shape, test.shape)
label_columns = test.columns[1:]
label_mapping = dict((label, index) for index, label in enumerate(label_columns))
label_mapping
for col in tqdm(label_columns):

    train_curated[col] = 0

    train_noisy[col] = 0

    

print(train_curated.shape, train_noisy.shape)
train_curated_labels = split_and_label(train_curated['labels'])

train_noisy_labels = split_and_label(train_noisy['labels'])
train_curated[label_columns] = train_curated_labels

train_noisy[label_columns] = train_noisy_labels
train_curated['num_labels'] = train_curated[label_columns].sum(axis=1)

train_noisy['num_labels'] = train_noisy[label_columns].sum(axis=1)
plt.figure(figsize=(18,6))



plt.subplot(121)

ax1 = train_curated['num_labels'].value_counts().plot(kind='bar')

plt.xlabel('Number of labels')

plt.ylabel('Counts')

plt.xticks(rotation=0)

plt.title('Curated Training Set')



for p in ax1.patches:

    ax1.annotate(str(p.get_height()), 

                (p.get_x() + p.get_width()/2., p.get_height() * 1.005), 

                ha='center',

                va='center',

                xytext=(0,5), 

                textcoords='offset points')



plt.subplot(122)

ax2 = train_noisy['num_labels'].value_counts().sort_index().plot(kind='bar', )

plt.xlabel('Number of labels')

plt.ylabel('Counts')

plt.xticks(rotation=0)

plt.title('Noisy Training Set')



for p in ax2.patches:

    ax2.annotate(str(p.get_height()), 

                (p.get_x() + p.get_width()/2., p.get_height() * 1.005), 

                ha='center',

                va='center',

                xytext=(0,5), 

                textcoords='offset points')



    

plt.show()
# Special thanks to https://github.com/makinacorpus/easydict/blob/master/easydict/__init__.py



class EasyDict(dict):



    def __init__(self, d=None, **kwargs):

        if d is None:

            d = {}

        if kwargs:

            d.update(**kwargs)

        for k, v in d.items():

            setattr(self, k, v)

        # Class attributes

        for k in self.__class__.__dict__.keys():

            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):

                setattr(self, k, getattr(self, k))



    def __setattr__(self, name, value):

        if isinstance(value, (list, tuple)):

            value = [self.__class__(x)

                     if isinstance(x, dict) else x for x in value]

        elif isinstance(value, dict) and not isinstance(value, self.__class__):

            value = self.__class__(value)

        super(EasyDict, self).__setattr__(name, value)

        super(EasyDict, self).__setitem__(name, value)



    __setitem__ = __setattr__



    def update(self, e=None, **f):

        d = e or dict()

        d.update(f)

        for k in d:

            setattr(self, k, d[k])



    def pop(self, k, d=None):

        delattr(self, k)

        return super(EasyDict, self).pop(k, d)
conf = EasyDict()

conf.sampling_rate = 44100

conf.duration = 5

conf.hop_length = 347 # to make time steps 128

conf.fmin = 20

conf.fmax = conf.sampling_rate // 2

conf.n_mels = 128

conf.n_fft = conf.n_mels * 20



conf.samples = conf.sampling_rate * conf.duration



train_curated_path = '../input/train_curated/'

train_noisy_path = '../input/train_noisy/'

test_path = '../input/test/'
def read_audio(conf, pathname, trim_long_data):

    y, sr = librosa.load(pathname, sr=conf.sampling_rate)

    # trim silence

    if 0 < len(y): # workaround: 0 length causes error

        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    # make it unified length to conf.samples

    if len(y) > conf.samples: # long enough

        if trim_long_data:

            y = y[0:0+conf.samples]

    else: # pad blank

        padding = conf.samples - len(y)    # add padding at both ends

        offset = padding // 2

        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

    return y



def audio_to_melspectrogram(conf, audio):

    spectrogram = librosa.feature.melspectrogram(audio, 

                                                 sr=conf.sampling_rate,

                                                 n_mels=conf.n_mels,

                                                 hop_length=conf.hop_length,

                                                 n_fft=conf.n_fft,

                                                 fmin=conf.fmin,

                                                 fmax=conf.fmax)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    return spectrogram



def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):

    x = read_audio(conf, pathname, trim_long_data)

    mels = audio_to_melspectrogram(conf, x)

    if debug_display:

        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))

        show_melspectrogram(conf, mels)

    return mels



def convert_wav_to_image(df, source):

    X = []

    for i, row in tqdm_notebook(df.iterrows()):

        try:

            x = read_as_melspectrogram(conf, f'{source[0]}/{str(row.fname)}', trim_long_data=True)

        except:

            x = read_as_melspectrogram(conf, f'{source[1]}/{str(row.fname)}', trim_long_data=True)



        #x_color = mono_to_color(x)

        X.append(x.transpose())

        #df.loc[i, 'length'] = x.shape[1]

    return X
#For baseline, noisy set is not used.

#train = pd.concat([train_curated, train_noisy],axis=0)
#del train_curated, train_noisy



#gc.collect()
%%time



#X = np.array(convert_wav_to_image(train, source=[train_curated_path, train_noisy_path]))

X = np.array(convert_wav_to_image(train_curated, source=[train_curated_path]))
Y = train_curated[label_columns].values
#del train

#gc.collect()
from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import Embedding, Input, Dense, CuDNNGRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
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
sequence_input = Input(shape=(636,128), dtype='float32')

x = CuDNNGRU(128, return_sequences=True)(sequence_input) 

att = Attention(636)(x)

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x) 



x = concatenate([att, avg_pool, max_pool])



preds = Dense(80, activation='softmax')(x)



model = Model(sequence_input, preds)

model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(0.005),metrics=['acc'])
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=123)
del X, Y

gc.collect()
assert len(x_train) == len(y_train)

assert len(x_val) == len(y_val)
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=3)
model.fit(np.array(x_train),

          y_train,

          batch_size=1024,

          epochs=100,

          validation_data=(np.array(x_val), y_val),

          callbacks = [es])
y_train_pred = model.predict(np.array(x_train))

y_val_pred = model.predict(np.array(x_val))
train_lwlrap = calculate_overall_lwlrap_sklearn(y_train, y_train_pred)

val_lwlrap = calculate_overall_lwlrap_sklearn(y_val, y_val_pred)



print(f'Training LWLRAP : {train_lwlrap:.4f}')

print(f'Validation LWLRAP : {val_lwlrap:.4f}')
%%time

X_test = np.array(convert_wav_to_image(test, source=[test_path]))
predictions = model.predict(np.array(X_test))
test[label_columns] = predictions
test.to_csv('submission.csv', index=False)