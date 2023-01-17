import numpy as np

import pandas as pd

import os



os.environ['WANDB_API_KEY'] = '0' # to silence warning



from transformers import BertTokenizer, TFBertModel

import matplotlib.pyplot as plt

import tensorflow as tf
MAX_LEN = 50

INPUT_DIR = '/kaggle/input/contradictory-my-dear-watson/'
for dirname, _, filenames in os.walk(INPUT_DIR):

    for filename in filenames:

        print(os.path.join(dirname, filename))
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)
train = pd.read_csv(INPUT_DIR + '/train.csv')
train.head()
train.premise.values[1]
train.hypothesis.values[1]
train.label.values[1]
labels, frequencies = np.unique(train.language.values, return_counts = True)



plt.figure(figsize = (10,10))

plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')

plt.show()
model_name = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_sentence(s):

    tokens = list(tokenizer.tokenize(s))

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)
encode_sentence('I love machine learning')
def bert_encode(hypotheses, premises, tokenizer):

    

    num_examples = len(hypotheses)



    sentence1 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])

    sentence2 = tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])



    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]

    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)



    input_mask = tf.ones_like(input_word_ids).to_tensor()



    type_cls = tf.zeros_like(cls)

    type_s1 = tf.zeros_like(sentence1)

    type_s2 = tf.ones_like(sentence2)

    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()



    inputs = {

        'input_word_ids': input_word_ids.to_tensor(),

        'input_mask': input_mask,

        'input_type_ids': input_type_ids}



    return inputs
train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)
def build_model():

    bert_encoder = TFBertModel.from_pretrained(model_name)

    input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')

    input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')

    input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

    

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]



    x = embedding

    

    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(32, 3)(x)

    

    x = x[:,0,:]

    

    output = tf.keras.layers.Dense(3, activation='softmax')(x)

    

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(lr=1e-5),

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy'])

    

    return model
with strategy.scope():

    model = build_model()

    model.summary()
EPOCHS = 4



history = model.fit(train_input,

                    train.label.values,

                    epochs=EPOCHS,

                    verbose=1,

                    batch_size=64,

                    validation_split=0.2)
losses = pd.DataFrame(model.history.history)

losses.plot()
test = pd.read_csv(INPUT_DIR + '/test.csv')

test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)
test.head()
predictions = [np.argmax(i) for i in model.predict(test_input)]
submission = test.id.copy().to_frame()

submission['prediction'] = predictions
submission.head()
submission.to_csv('submission.csv', index=False)