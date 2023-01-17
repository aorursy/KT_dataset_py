import time

import gc

import numpy as np

np.random.seed(420)

import pandas as pd

import sentencepiece as spm

from gensim.models import KeyedVectors

import tensorflow

from tensorflow.keras.preprocessing.sequence import pad_sequences

from spacy.lang.en import English

from tqdm import tqdm
MAXLEN_TITLE = 200

MAXLEN_QA = 30

GLOVE_DIMS = 100

nlp = English()

tokenizer = nlp.Defaults.create_tokenizer(nlp)
def get_glove():

    glove = {}

    with open(f'../input/glove-global-vectors-for-word-representation/glove.6B.{GLOVE_DIMS}d.txt','r') as f:

        for line in f:

            values = line.split()

            vectors = np.asarray(values[1:],'float32')

            glove[values[0]]=vectors

    f.close()

    return glove



def urls_to_label(urls):

    return [x.split('//')[1].split('.')[0] for x in urls]



def desuffix(token, glove):

    while token:

        token = token[:-1]

        if token in glove:

            return token

    return token



def embed(glove, tokenizer, texts, maxlen=0, padding='post'):

    seqs = []

    for text in tqdm(tokenizer.pipe(texts), total=len(texts)):

        seq = []

        for t in text:

            t = t.text.lower()

            if t not in glove:

                t = desuffix(t, glove)

                if t:

                    seq.append(glove[t])

                else:

                    seq.append(np.zeros(GLOVE_DIMS))

            else:

                seq.append(glove[t])

        if maxlen:

            seqs.append(seq)

        else:

            try:

                seqs.append(np.mean(seqs, axis=0))

            except:

                seqs.append(np.zeros(GLOVE_DIMS))

    if maxlen:

        return pad_sequences(seqs, maxlen=maxlen, padding=padding, dtype='float32')

    else:

        return seqs
glove = get_glove()
train = pd.read_csv("../input/google-quest-challenge/train.csv")

# train = train.sample(20)

val = train.sample(int((len(train) / 100) * 10))

train = train.drop(val.index)

test = pd.read_csv("../input/google-quest-challenge/test.csv")

# test = test.sample(5)

target_columns = train.columns[len(test.columns):]

target = train[train.columns[-len(train.columns[len(test.columns):]):]]

target_val = val[val.columns[-len(val.columns[len(test.columns):]):]]
target_q = train[[x for x in train.columns[-len(train.columns[len(test.columns):]):] if x.startswith('q')]]

target_a = train[[x for x in train.columns[-len(train.columns[len(test.columns):]):] if x.startswith('a')]]

target_q_val = val[[x for x in val.columns[-len(val.columns[len(test.columns):]):] if x.startswith('q')]]

target_a_val = val[[x for x in val.columns[-len(val.columns[len(test.columns):]):] if x.startswith('a')]]
class SelfAttention(tensorflow.keras.layers.Layer):

    """



    Lifted from https://github.com/uzaymacar/attention-mechanisms <3



    Layer for implementing self-attention mechanism. Weight variables were preferred over Dense()

    layers in implementation because they allow easier identification of shapes. Softmax activation

    ensures that all weights sum up to 1.

    @param (int) size: a.k.a attention length, number of hidden units to decode the attention before

           the softmax activation and becoming annotation weights

    @param (int) num_hops: number of hops of attention, or number of distinct components to be

           extracted from each sentence.

    @param (bool) use_penalization: set True to use penalization, otherwise set False

    @param (int) penalty_coefficient: the weight of the extra loss

    @param (str) model_api: specify to use TF's Sequential OR Functional API, note that attention

           weights are not outputted with the former as it only accepts single-output layers

    """



    def __init__(

        self,

        size,

        num_hops=8,

        use_penalization=False,

        penalty_coefficient=0.1,

        model_api='functional',

        W1=None,

        W2=None,

        **kwargs,

    ):

        if model_api not in ['sequential', 'functional']:

            raise ValueError("Argument for param @model_api is not recognized")

        self.size = size

        self.num_hops = num_hops

        self.use_penalization = use_penalization

        self.penalty_coefficient = penalty_coefficient

        self.model_api = model_api

        super(SelfAttention, self).__init__(**kwargs)



    def get_config(self):

        base_config = super(SelfAttention, self).get_config()

        base_config['size'] = self.size

        base_config['num_hops'] = self.num_hops

        base_config['use_penalization'] = self.use_penalization

        base_config['penalty_coefficient'] = self.penalty_coefficient

        base_config['model_api'] = self.model_api

        return base_config



    def build(self, input_shape):

        self.W1 = self.add_weight(

            name='W1',

            shape=(self.size, input_shape[2]),

            initializer='glorot_uniform',

            trainable=True,

        )

        self.W2 = self.add_weight(

            name='W2',

            shape=(self.num_hops, self.size),

            initializer='glorot_uniform',

            trainable=True,

        )

        super(SelfAttention, self).build(input_shape)



    def call(self, inputs):

        # Expand weights to include batch size through implicit broadcasting

        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]

        hidden_states_transposed = tensorflow.keras.layers.Permute(dims=(2, 1))(inputs)

        attention_score = tensorflow.matmul(W1, hidden_states_transposed)

        attention_score = tensorflow.keras.layers.Activation('tanh')(attention_score)

        attention_weights = tensorflow.matmul(W2, attention_score)

        attention_weights = tensorflow.keras.layers.Activation('softmax')(attention_weights)

        embedding_matrix = tensorflow.matmul(attention_weights, inputs)

        embedding_matrix_flattened = tensorflow.keras.layers.Flatten()(embedding_matrix)



        if self.use_penalization:

            attention_weights_transposed = tensorflow.keras.layers.Permute(dims=(2, 1))(attention_weights)

            product = tensorflow.matmul(attention_weights, attention_weights_transposed)

            identity = tensorflow.eye(self.num_hops, batch_shape=(inputs.shape[0],))

            frobenius_norm = tensorflow.sqrt(tf.reduce_sum(tf.square(product - identity)))

            self.add_loss(self.penalty_coefficient * frobenius_norm)



        if self.model_api == 'functional':

            return embedding_matrix_flattened, attention_weights

        elif self.model_api == 'sequential':

            return embedding_matrix_flattened

start_time = time.time()

title = embed(glove, tokenizer, train["question_title"].values, maxlen=MAXLEN_TITLE)

title_val = embed(glove, tokenizer, val["question_title"].values, maxlen=MAXLEN_TITLE)

title_test = embed(glove, tokenizer, test["question_title"].values, maxlen=MAXLEN_TITLE)

question = embed(glove, tokenizer, train["question_body"].values, maxlen=MAXLEN_QA)

question_val = embed(glove, tokenizer, val["question_body"].values, maxlen=MAXLEN_QA)

question_test = embed(glove, tokenizer, test["question_body"].values, maxlen=MAXLEN_QA)

answer = embed(glove, tokenizer, train["answer"].values, maxlen=MAXLEN_QA)

answer_val = embed(glove, tokenizer, val["answer"].values, maxlen=MAXLEN_QA)

answer_test = embed(glove, tokenizer, test["answer"].values, maxlen=MAXLEN_QA)

elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

print('Done embedding in:', elapsed)
# Question model



input_title = tensorflow.keras.layers.Input(

    shape=(MAXLEN_TITLE, GLOVE_DIMS), dtype='float32'

)

input_question = tensorflow.keras.layers.Input(

    shape=(MAXLEN_QA, GLOVE_DIMS), dtype='float32'

)

input_answer = tensorflow.keras.layers.Input(

    shape=(MAXLEN_QA, GLOVE_DIMS), dtype='float32'

)



dropped_title = tensorflow.keras.layers.Dropout(0.5)(input_title)

dropped_question = tensorflow.keras.layers.Dropout(0.5)(input_question)

dropped_answer = tensorflow.keras.layers.Dropout(0.5)(input_answer)



attended_title, _ = SelfAttention(size=128, num_hops=MAXLEN_TITLE)(dropped_title)

attended_question, _ = SelfAttention(size=128, num_hops=int(MAXLEN_QA))(dropped_question)

attended_answer, _ = SelfAttention(size=128, num_hops=int(MAXLEN_QA))(dropped_answer)



lstm_title, _, _ = tensorflow.keras.layers.LSTM(

    128,

    dropout=0.5,

    recurrent_dropout=0.5,

    return_sequences=True,

    return_state=True,

)(input_title)

lstm_question, _, _ = tensorflow.keras.layers.LSTM(

    256,

    dropout=0.5,

    recurrent_dropout=0.5,

    return_sequences=True,

    return_state=True,

)(input_question)

lstm_answer, _, _ = tensorflow.keras.layers.LSTM(

    128,

    dropout=0.5,

    recurrent_dropout=0.5,

    return_sequences=True,

    return_state=True,

)(input_answer)



attended_lstm_title, _ = SelfAttention(size=128, num_hops=MAXLEN_TITLE)(lstm_title)

attended_lstm_question, _ = SelfAttention(size=256, num_hops=int(MAXLEN_QA))(lstm_question)

attended_lstm_answer, _ = SelfAttention(size=128, num_hops=int(MAXLEN_QA))(lstm_answer)



conc = tensorflow.keras.layers.concatenate(

    [attended_title, attended_question, attended_answer, attended_lstm_title, attended_lstm_question, attended_lstm_answer]

)

output_layer = tensorflow.keras.layers.Dense(len(target_q.columns), activation='sigmoid')(conc)

model = tensorflow.keras.models.Model(

    inputs=[input_title, input_question, input_answer],

    outputs=output_layer

)



optimizer = tensorflow.keras.optimizers.get('adam')

optimizer.learning_rate = 1e-4



model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
start_time = time.time()

model.fit(

    [title, question, answer], [target_q],

    validation_data=([title_val, question_val, answer_val], [target_q_val]),

    batch_size=32,

    epochs=50,

    callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],

    verbose=1,

)

elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

print('Done training in:', elapsed)
preds_q = model.predict(

    [title_test, question_test, answer_test]

)
tensorflow.keras.backend.clear_session()

del model

gc.collect()
# Answer model



input_title = tensorflow.keras.layers.Input(

    shape=(MAXLEN_TITLE, GLOVE_DIMS), dtype='float32'

)

input_question = tensorflow.keras.layers.Input(

    shape=(MAXLEN_QA, GLOVE_DIMS), dtype='float32'

)

input_answer = tensorflow.keras.layers.Input(

    shape=(MAXLEN_QA, GLOVE_DIMS), dtype='float32'

)



dropped_title = tensorflow.keras.layers.Dropout(0.5)(input_title)

dropped_question = tensorflow.keras.layers.Dropout(0.5)(input_question)

dropped_answer = tensorflow.keras.layers.Dropout(0.5)(input_answer)



attended_title, _ = SelfAttention(size=128, num_hops=MAXLEN_TITLE)(dropped_title)

attended_question, _ = SelfAttention(size=128, num_hops=int(MAXLEN_QA))(dropped_question)

attended_answer, _ = SelfAttention(size=128, num_hops=int(MAXLEN_QA))(dropped_answer)



lstm_title, _, _ = tensorflow.keras.layers.LSTM(

    128,

    dropout=0.5,

    recurrent_dropout=0.5,

    return_sequences=True,

    return_state=True,

)(input_title)

lstm_question, _, _ = tensorflow.keras.layers.LSTM(

    128,

    dropout=0.5,

    recurrent_dropout=0.5,

    return_sequences=True,

    return_state=True,

)(input_question)

lstm_answer, _, _ = tensorflow.keras.layers.LSTM(

    256,

    dropout=0.5,

    recurrent_dropout=0.5,

    return_sequences=True,

    return_state=True,

)(input_answer)



attended_lstm_title, _ = SelfAttention(size=128, num_hops=MAXLEN_TITLE)(lstm_title)

attended_lstm_question, _ = SelfAttention(size=128, num_hops=int(MAXLEN_QA))(lstm_question)

attended_lstm_answer, _ = SelfAttention(size=256, num_hops=int(MAXLEN_QA))(lstm_answer)



conc = tensorflow.keras.layers.concatenate(

    [attended_title, attended_question, attended_answer, attended_lstm_title, attended_lstm_question, attended_lstm_answer]

)

output_layer = tensorflow.keras.layers.Dense(len(target_a.columns), activation='sigmoid')(conc)

model = tensorflow.keras.models.Model(

    inputs=[input_title, input_question, input_answer],

    outputs=output_layer

)



optimizer = tensorflow.keras.optimizers.get('adam')

optimizer.learning_rate = 1e-4



model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
start_time = time.time()

model.fit(

    [title, question, answer], [target_a],

    validation_data=([title_val, question_val, answer_val], [target_a_val]),

    batch_size=32,

    epochs=50,

    callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],

    verbose=1,

)

elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

print('Done training in:', elapsed)
preds_a = model.predict(

    [title_test, question_test, answer_test]

)
preds = np.concatenate([preds_q, preds_a], axis=1)
# preds = model.predict(

#     [title_test, question_test, answer_test]

# )

submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")

for i, column in enumerate(target_columns):

    submission[column] = preds[:, i]

submission.to_csv("submission.csv", index=False)