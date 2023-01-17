# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation,Dropout, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing import sequence
import keras as keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
print(os.listdir("../input"))
TRAIN = os.path.join("../input", "train.json")
SAMPLE_SUBMISSION = os.path.join("../input", "sample_submission.csv")
# SUBMISSION = os.path.join("submission.csv")


# Any results you write to the current directory are saved as output.
# !cat ../input/train.json

df = pd.read_json(TRAIN)
print(df['cuisine'].describe())
print(df.groupby('cuisine').count())
import re
def flat_ingredients(x): 
    x = ' '.join([i.replace(' ', '_').replace('-', '_').lower() for i in x])
    # x = re.sub(re.compile('[^a-zA-Z0-9]+'), " ", x)
    # print(x)
    return x
df['flat'] = df['ingredients'].apply(lambda x: flat_ingredients(x))
"""print(df.head(15))"""
max_len = 0
for i in df['flat'].tolist():
    l = len(str(i).split())
    if str(i) != str(i).lower(): print(i)
    if max_len < l:
        max_len = l
print(max_len)
cuisines = list(set([str(i) for i in df['cuisine'].tolist()]))
ingredients = list(set([j  for i in df['flat'].tolist() for j in str(i).split()]))
ingredients.append("blank")
cuisines_to_idx = {w: i for i, w in enumerate(cuisines)}
ingredients_to_idx = {w:i for i, w in enumerate(ingredients)}
from keras.utils import to_categorical
from random import shuffle

X = []
for i in df['flat'].tolist():
    temp = []
    for j in str(i).split():
        temp.append(ingredients_to_idx[j])
    X.append(temp)    
y = [cuisines_to_idx[str(i)] for i in df['cuisine'].tolist()]
print(str(len(X)) + " original length! ")
for i in range(0, len(X)):
    for j in range(0, 3):
        temp = X[i]
        shuffle(temp)
        X.append(temp)
        y.append(y[i])
"""print(X[0])
print(shuffle(X[0]))
print(X[0])

print(shuffle(X[0]))
print(X[0])"""
"""print(X[:10], y[:10])
print(len(X))"""
# print(type(X[0]))
# print(y[0])

X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=ingredients_to_idx["blank"])
y = to_categorical(y, num_classes=len(cuisines))
"""from sklearn.model_selection import train_test_split
X_tr, X_vd, y_tr, y_vd = train_test_split(X, y, test_size=0.3, random_state=2)"""
"""
i = 0
# print(len(X_tr))
print(len(y_tr))
# print(type(X_tr[0]))
# print(X_tr.shape)
# a = np.random.permutation(X_tr[i])
# print(a.shape)
# print(len(np.insert(X_tr, len(X_tr), a, axis=0)))
y_tr = np.insert(y_tr, len(y_tr), y_tr[i], axis=0)

# print(X_tr.shape)
print(len(y_tr))"""
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K, activations
class AttentiveConv(Layer):
    def __init__(self, kernel_activation='tanh', filters=3, **kwargs):
        super(AttentiveConv, self).__init__(**kwargs)
        self.kernel_activation = activations.get(kernel_activation)
        if filters%2 == 0:
            self.filters = filters - 1
        else:
            self.filters = filters
        self.filters = filters
        K.set_floatx('float32')

    def build(self, input_shape):
        self.num_words = input_shape[0][1]
        self.em_dim = input_shape[0][2]
        self.W2 = self.add_weight(shape=(self.em_dim, self.filters*self.em_dim), dtype=K.floatx(), name='att_cont_weight', trainable=True, initializer='glorot_normal')
        self.We = self.add_weight(shape=(self.em_dim, self.em_dim), dtype=K.floatx(), name='window_weight', trainable=True, initializer='glorot_normal')
        super(AttentiveConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        #the input is a list of two tensors. As this layers computes a score for every element of the first input I just
        #return the shape of this tensor.
        return input_shape[0]

    def get_config(self):
        config = {'kernel_activation': activations.serialize(self.kernel_activation),
                  'filters': self.filters}
        base_config = super(AttentiveConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        #x is a list of two tensors
        #casting makes no sense so I deleted it
        text = x[0]
        context = x[1]

        #applies bilinear energy funtion (text * We * context)
        #and weights the computed feature map like in equation 6 (W2 * ci)
        
        #shape of text/context is (batch_size, num_words, em_dim), num_words for text is 200 and em_dim is also 200. 
        #I want to do the computation for every sample of the batch. I found batch_matmul but thats not available in 
        #tensorflow 1.5
        #shape of weighted_attentive_context should be the same shape as text.
        weighted_attentive_context = self._compute_attentive_context(text, context)
        return weighted_attentive_context

    def _compute_attentive_context(self, text, context):
        #computes the context-score for every vector like equation 2
        temp = K.dot(text, self.We)
        scores = K.batch_dot(temp, K.permute_dimensions(context, (0,2,1)))

        #softmax along every vector-element
        #scores = text
        scores_softmax = activations.softmax(scores, axis=1)

        #computes the context featur_map like equation 4
        res = tf.matmul(scores_softmax, context)

        #weights the output like equation 6
        res = K.permute_dimensions(K.dot(self.W2,K.permute_dimensions(res, (0,2,1))),(1,2,0))
        #res = scores
        return res
print(str(len(X)) + ' exploded length!')
from sklearn.model_selection import train_test_split
X_tr, X_vd, y_tr, y_vd = train_test_split(X, y, test_size=0.0001, random_state=2)

"""for i in range(0, len(X_tr)):
    for j in range(3):
        X_tr = np.insert(X_tr, len(X_tr), np.random.permutation(X_tr[i]), axis=0)
        y_tr = np.insert(y_tr, len(y_tr), y_tr[i], axis=0)"""

def get_model():
    model_conv = Sequential()
    model_conv.add(Embedding(len(ingredients), 100, input_length=max_len))
    model_conv.add(Dropout(0.2))
    model_conv.add(AttentiveConv(filters=64, kernel_activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(Flatten())
    # model_conv.add(LSTM(100))
    model_conv.add(Dense(64))
    model_conv.add(Activation('relu'))
    model_conv.add(Dropout(0.5))
    model_conv.add(Dense(len(cuisines), activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model_conv.summary())
    return model_conv

"""def get_model2():
    model = Sequential()
    model.add(Dense(2048, input_shape=(max_len,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(cuisines)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print(model.summary())
    return model"""
model = get_model()
model.fit(X_tr, np.array(y_tr), batch_size=128, epochs=11, verbose=1, validation_data=(X_vd, np.array(y_vd)))
model.evaluate(X_vd, np.array(y_vd), batch_size=32, verbose=1)
model.predict_classes(X_vd[:1])
targets = model.predict_classes(X_vd[:20])
print(targets)
for i in range(0, len(targets)):
    if 1 != int(y_vd[i][targets[i]]):
        print("sentence : " + str([ingredients[k] for k in X_vd[i]]))
        print("correct: " + str(y_vd[i]))
        print("incorrect: " + str(cuisines[targets[i]] ))
        print("incorrect: " + str(targets[i] ))
test_df = pd.read_json('../input/test.json')
test_df['flat'] = test_df['ingredients'].apply(lambda x: flat_ingredients(x))
test_df.head()
test_X = []
for i in test_df['flat'].tolist():
    temp = []
    for j in str(i).split():
        try:
            temp.append(ingredients_to_idx[j])
        except:
            temp.append(ingredients_to_idx["blank"])
    test_X.append(temp)
test_X = pad_sequences(maxlen=max_len, sequences=test_X, padding="post", value=ingredients_to_idx["blank"])

targets = model.predict_classes(test_X)
def insert_column(x):
    return(int (x.index[0]))
    return cuisines[targets[x.index]]
# print(test_df.apply(lambda x: insert_column(x)))

test_df['cuisine'] = pd.Series([cuisines[i] for i in targets])
test_df = test_df[['id', 'cuisine']]
test_df.to_csv('submission.csv', index=False)
