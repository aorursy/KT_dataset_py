import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModel

from tqdm.notebook import tqdm
!pip install nlp

import nlp
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
MODEL_NAME = 'jplu/tf-xlm-roberta-large'

EPOCHS = 10

MAX_LEN = 80

RATE = 1e-5



BATCH_SIZE = 64 * strategy.num_replicas_in_sync
train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')

test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')

submission = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')
train.info()
train.head()
train = train[['premise', 'hypothesis', 'label']]
multigenre_data = nlp.load_dataset(path='glue', name='mnli')
index = []

premise = []

hypothesis = []

label = []



for example in multigenre_data['train']:

    premise.append(example['premise'])

    hypothesis.append(example['hypothesis'])

    label.append(example['label'])
multigenre_df = pd.DataFrame(data={

    'premise': premise,

    'hypothesis': hypothesis,

    'label': label

})
multigenre_df.head()
adversarial_data = nlp.load_dataset(path='anli')
index = []

premise = []

hypothesis = []

label = []



for example in adversarial_data['train_r1']:

    premise.append(example['premise'])

    hypothesis.append(example['hypothesis'])

    label.append(example['label'])

    

for example in adversarial_data['train_r2']:

    premise.append(example['premise'])

    hypothesis.append(example['hypothesis'])

    label.append(example['label'])

    

for example in adversarial_data['train_r3']:

    premise.append(example['premise'])

    hypothesis.append(example['hypothesis'])

    label.append(example['label'])
adversarial_df = pd.DataFrame(data={

    'premise': premise,

    'hypothesis': hypothesis,

    'label': label

})
train = pd.concat([train, multigenre_df, adversarial_df])
train.info()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_text = train[['premise', 'hypothesis']].values.tolist()

test_text = test[['premise', 'hypothesis']].values.tolist()
train_encoded = tokenizer.batch_encode_plus(

    train_text,

    pad_to_max_length=True,

    max_length=MAX_LEN

)
test_encoded = tokenizer.batch_encode_plus(

    test_text,

    pad_to_max_length=True,

    max_length=MAX_LEN

)
train.premise.values[0]
print(train_encoded.input_ids[0][0:14])
vocab = tokenizer.get_vocab()



print(vocab['<s>'])

print(vocab['▁and'])

print(vocab['▁these'])

print(vocab['▁comments'])

print(vocab['▁were'])

print(vocab['▁considered'])

print(vocab['▁in'])

print(vocab['▁formula'])

print(vocab['ting'])

print(vocab['▁the'])

print(vocab['▁inter'])

print(vocab['im'])

print(vocab['▁rules'])

print(vocab['.'])
train.hypothesis.values[0]
print(train_encoded.input_ids[0][14:32])
print(vocab['</s>'])

print(vocab['▁The'])

print(vocab['▁rules'])

print(vocab['▁developed'])

print(vocab['▁in'])

print(vocab['▁the'])

print(vocab['▁inter'])

print(vocab['im'])
train_encoded.keys()
print(train_encoded.attention_mask[0][0:35])
x_train, x_valid, y_train, y_valid = train_test_split(

    train_encoded['input_ids'], 

    train.label.values, 

    test_size=0.2, 

    random_state=2020

)
x_test = test_encoded['input_ids']
auto = tf.data.experimental.AUTOTUNE



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(auto)

)
valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(auto)

)
test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
with strategy.scope():

    backbone = TFAutoModel.from_pretrained(MODEL_NAME)
with strategy.scope():

    x_input = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")



    x = backbone(x_input)[0]



    x = x[:, 0, :]



    x = tf.keras.layers.Dense(3, activation='softmax')(x)



    model = tf.keras.models.Model(inputs=x_input, outputs=x)
model.compile(

    tf.keras.optimizers.Adam(lr=RATE), 

    loss='sparse_categorical_crossentropy', 

    metrics=['accuracy']

)
model.summary()
steps = len(x_train) // BATCH_SIZE



history = model.fit(

    train_dataset,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    steps_per_epoch=steps,

)
fig, ax = plt.subplots(2, 2, figsize=(15, 5))



ax[0,0].set_title('Train Loss')

ax[0,0].plot(history.history['loss'])



ax[0,1].set_title('Train Accuracy')

ax[0,1].plot(history.history['accuracy'])



ax[1,0].set_title('Val Loss')

ax[1,0].plot(history.history['val_loss'])



ax[1,1].set_title('Val Accuracy')

ax[1,1].plot(history.history['val_accuracy'])
test_preds = model.predict(test_dataset, verbose=1)

submission['prediction'] = test_preds.argmax(axis=1)
submission.head()
submission.to_csv('submission.csv', index=False)