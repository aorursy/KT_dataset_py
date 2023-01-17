import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
df = pd.read_csv('../input/train.csv',index_col=0)
target = 'target'
X = df.drop(target,axis=1)
y = df[target]
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder

from progressbar import ProgressBar
pbar = ProgressBar()

# len of lstm, chosen to be 200 based on the length of the questions derived from the eda in my main kernel
max_len = 200
# number of features
max_features = 50000
# size of embedding vectors
embed_size = 300
## preprocess
def preprocess(X):
    # first we tokenize using the tokenizer
    print("tokenizing")
    tk = Tokenizer(lower = True, filters='', num_words=max_features)
    full_text = list(X['question_text'].values)
    tk.fit_on_texts(full_text)
    
    # then we load up the embeddings
    embedding_path = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
    
    # and now we'll retrieve the map from the embeddings
    print("getting embeddings")
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore'))
    
    # build up the embedding matrix that maps words to vectors
    print("building embedding matrix")
    word_index = tk.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return tk, embedding_matrix

## featurize X
def featurize(X):
    # tokenize the text
    X_tokenized = tk.texts_to_sequences(X['question_text'].fillna('missing'))
    
    # remove sequences longer than max_len and pad the rest
    X_pad = pad_sequences(X_tokenized, maxlen = max_len)
    
    return X_pad
# first preprocess on all the available data
# we need to be careful with what we do here - want to avoid any data
# leakage when doing the testing later one
tk, embedding_matrix = preprocess(X)
# attention layer from https://www.kaggle.com/shujian/different-embeddings-with-attention-fork-fork
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, CuDNNLSTM, Dense, Flatten, Conv2D, Conv1D
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from keras.layers import AveragePooling1D, MaxPooling1D, TimeDistributed
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K

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
# this model uses my main kernel as inspiration
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, CuDNNLSTM, Dense, Flatten, Conv1D
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from keras.layers import AveragePooling1D, MaxPooling1D
def createModel1():
    # input layer
    inp = Input(shape = (max_len,))
    # embedding layer
    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    # spatial dropout to add robustness with respect to feature maps
    x1 = SpatialDropout1D(0.3)(x)
    
    # here I use my original kernel (link above) as inspiration to develop an architecture
    # in my original kernel, I tried to use feature engineering in terms of topic modeling to encapsulate the interaction terms between features before doing classification
    # here, I'll follow the same basic approach
    # first create a convolutional layer that attempts to find the "interactive" relationships between words
    # I'll use a kernel of 3 to encapsulate a "3-gram"
    x_conv1 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x1)
    x_conv1 = MaxPooling1D(pool_size=3)(x_conv1)
    # I'll also create an lstm that processes the words directly, followed by a convolutional layer to accumulate the results
    x_lstm1 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x1)
    x_lstm1 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm1)
    x_lstm1 = MaxPooling1D(pool_size=3)(x_lstm1)
    
    # create an x2 layer that combines the information from first layers
    x2 = concatenate([x_conv1, x_lstm1])
    
    # create a second layer similar to the first layer, except we also introduce a skip connection to the original layer
    # this is to emulate the second level stacked model
    # implement the convolutional layer
    x_conv2 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x2)
    x_conv2 = MaxPooling1D(pool_size=3)(x_conv2)
    # lstm layer
    x_lstm2 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x2)
    x_lstm2 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm2)
    x_lstm2 = MaxPooling1D(pool_size=3)(x_lstm2)
    
    # create the final layer that will be used as an input to the classifier unit, including the skip connection to the first layer
    x3 = concatenate([x_conv2, x_lstm2])
    x3 = Dense(64, activation='relu')(x2)
    
    # create the output classifier unit
    out = Flatten()(x3)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(16,activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=inp, output=out)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
# this is an ensemble network
# https://www.kaggle.com/yekenot/2dcnn-textclassifier
# https://www.kaggle.com/ashishsinhaiitr/different-embeddings-with-attention-fork-fork
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, CuDNNLSTM, Dense, Flatten, Conv1D
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from keras.layers import AveragePooling1D, MaxPooling1D, TimeDistributed, Reshape
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers import CuDNNGRU
def createModel2():
    # input layer
    inp = Input(shape = (max_len,))
    # embedding layer
    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    # spatial dropout to add robustness with respect to feature maps
    x1 = SpatialDropout1D(0.3)(x)
    
    x_lstm1 = Bidirectional(CuDNNLSTM(64, return_sequences = True))(x1)
    x_lstm2 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_lstm1)
    x_avg1 = GlobalAveragePooling1D()(x_lstm1)
    x_max1 = GlobalMaxPooling1D()(x_lstm1)
    x_avg2 = GlobalAveragePooling1D()(x_lstm2)
    x_max2 = GlobalMaxPooling1D()(x_lstm2)
    
    x_gru1 = Bidirectional(CuDNNGRU(64, return_sequences = True))(x1)
    x_gru2 = Bidirectional(CuDNNGRU(32, return_sequences = True))(x_gru1)
    x_avg3 = GlobalAveragePooling1D()(x_gru1)
    x_max3 = GlobalMaxPooling1D()(x_gru1)
    x_avg4 = GlobalAveragePooling1D()(x_gru2)
    x_max4 = GlobalMaxPooling1D()(x_gru2)
    
    x_lstm3 = Bidirectional(CuDNNLSTM(64, return_sequences = True))(x1)
    x_lstm4 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_lstm3)
    x_att_1 = Attention(max_len)(x_lstm3)
    x_att_2 = Attention(max_len)(x_lstm4)
    
    x_gru3 = Bidirectional(CuDNNGRU(64, return_sequences = True))(x1)
    x_gru4 = Bidirectional(CuDNNGRU(32, return_sequences = True))(x_gru3)
    x_att_3 = Attention(max_len)(x_gru3)
    x_att_4 = Attention(max_len)(x_gru4)
    
    filters = [1, 3, 5, 7]

    x_conv = Reshape((max_len, embed_size, 1))(x1)
    conv_maxpool = []
    conv_avgpool = []
    for filt_size in filters:
        conv = Conv2D(36, kernel_size=(filt_size, embed_size), kernel_initializer='glorot_uniform', activation='elu')(x_conv)
        conv_maxpool.append(MaxPooling2D(pool_size=(max_len - filt_size + 1, 1))(conv))
        conv_avgpool.append(AveragePooling2D(pool_size=(max_len - filt_size + 1, 1))(conv))
    
    x_avg5 = Flatten()(concatenate(conv_avgpool))
    x_max5 = Flatten()(concatenate(conv_maxpool))
    
    out = concatenate([x_avg1,x_max1, x_avg2, x_max2,
                       x_avg3,x_max3, x_avg4, x_max4,
                       x_avg5, x_max5,
                       x_att_1, x_att_2, x_att_3, x_att_4])
    
    # create the output classifier unit
#     out = Flatten()(x_lstm1)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(256,activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=inp, output=out)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
# this model uses skip connections to create a deeper network
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, CuDNNLSTM, Dense, Flatten, Conv1D
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from keras.layers import AveragePooling1D, MaxPooling1D
def createModel3():
    # input layer
    inp = Input(shape = (max_len,))
    # embedding layer
    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    # spatial dropout to add robustness with respect to feature maps
    x1 = SpatialDropout1D(0.3)(x)
    
    # I'm going to several layers of lstm-cnn blocks, while introducing skip connections every other block
    x_lstm1 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x1)
    x_conv1 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm1)
    x_max1 = MaxPooling1D(pool_size=3)(x_conv1)
    
    x_lstm2 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_conv1)
    x_conv2 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm2)
    x_max2 = MaxPooling1D(pool_size=3)(x_conv2)
    
    x_max12 = concatenate([x_max1,x_max2])
    
    x_lstm3 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_max12)
    x_conv3 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm3)
    x_max3 = MaxPooling1D(pool_size=3)(x_conv3)
    
    x_lstm4 = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_conv3)
    x_conv4 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm4)
    x_max4 = MaxPooling1D(pool_size=3)(x_conv4)
    
    x_max34 = concatenate([x_max3, x_max4])
    
    x_lstm5= Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_max34)
    x_conv5= Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm5)
    x_max5 = MaxPooling1D(pool_size=3)(x_conv5)
                                                                                       
    x_lstm6= Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_conv5)
    x_conv6 = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm6)
    x_max6 = MaxPooling1D(pool_size=3)(x_conv6)
    
    out = concatenate([x_max5, x_max6])
    out = Flatten()(out)
    
    # create the output classifier unit
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(16,activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=inp, output=out)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
# simple 2 layer model
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, CuDNNLSTM, Dense, Flatten, Conv1D
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from keras.layers import AveragePooling1D, MaxPooling1D, TimeDistributed
def createModel4():
    # input layer
    inp = Input(shape = (max_len,))
    # embedding layer
    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    # spatial dropout to add robustness with respect to feature maps
    x1 = SpatialDropout1D(0.3)(x)
    
    x_lstm = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x1)
    x_lstm = TimeDistributed(Dense(32))(x_lstm)
    x_lstm = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm)
    x_lstm = Bidirectional(CuDNNLSTM(32, return_sequences = True))(x_lstm)
    x_lstm = TimeDistributed(Dense(32))(x_lstm)
    x_lstm = Conv1D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x_lstm)
    x_lstm = BatchNormalization()(x_lstm)
    x_lstm = Dropout(0.3)(x_lstm)
    
    out = Flatten()(x_lstm)
    
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=inp, output=out)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
# simple model with attention layer from https://www.kaggle.com/shujian/different-embeddings-with-attention-fork-fork
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, CuDNNLSTM, Dense, Flatten, Conv2D, Conv1D
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from keras.layers import AveragePooling1D, MaxPooling1D, TimeDistributed
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
    
def createModel5():
    # input layer
    inp = Input(shape = (max_len,))
    # embedding layer
    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    # spatial dropout to add robustness with respect to feature maps
    x = SpatialDropout1D(0.1)(x)
    
    x_lstm = Bidirectional(CuDNNLSTM(128, return_sequences = True))(x)
    x_lstm = Bidirectional(CuDNNLSTM(64, return_sequences = True))(x_lstm)
    x_lstm = Attention(max_len)(x_lstm)
    
    out = Dense(64, activation='relu')(x_lstm)
    out = BatchNormalization()(out)
    out = Dropout(0.1)(out)
    out = Dense(1, activation='sigmoid')(out)
    
    model = Model(inputs=inp, output=out)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
model = createModel2()
model.summary()
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
history = model.fit(featurize(X_train),y_train,
          epochs=3,
          batch_size=512,
          validation_split=0.1,
          verbose=1,
          callbacks=[early_stop],
          class_weight=class_weights
         )
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
pred = model.predict(featurize(X_test), batch_size=1024, verbose=1)
y_pred = pd.DataFrame(pred)
y_pred.index=X_test.index
y_pred.iloc[:,0].head()
from sklearn import metrics
best_thresh = None
best_score = None
for thresh in np.arange(0.001, 0.51, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(y_test, (y_pred.iloc[:,0]>thresh).astype(int))
    print("F1 score at threshold {0:.5f} is {1:.5f}".format(thresh, score))
    if best_score == None:
        best_thresh = thresh
        best_score = score
    else:
        if score > best_score:
            best_thresh = thresh
            best_score = score
print("Best F1 score at {0:0.3f} with threshold {1:0.3f}".format(best_score,best_thresh))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred.iloc[:,0]>best_thresh))
df_test = pd.read_csv('../input/test.csv', index_col=0)
test_pred = model.predict(featurize(df_test), batch_size=1024, verbose=1)
test_pred = pd.DataFrame(test_pred)
test_pred.index=df_test.index
y_test_pred = (test_pred.iloc[:,0]>best_thresh).astype(int)
submit_df = pd.DataFrame({"qid": df_test.index.to_series(), "prediction": y_test_pred})
submit_df.to_csv("submission.csv", index=False)