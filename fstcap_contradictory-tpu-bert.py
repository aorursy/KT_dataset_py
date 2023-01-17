!pip install -q tf-models-official==2.2.0
!pip install -q nlp
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from official.nlp import bert
from official import nlp

import official.nlp.bert.bert_models
import official.nlp.bert.tokenization
import official.nlp.bert.configs
import official.nlp.optimization

import nlp as an_nlp

from kaggle_datasets import KaggleDatasets

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from collections import Counter
import json
import time
import unicodedata
import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
GCS_PATH_TO_SAVEDMODEL = KaggleDatasets().get_gcs_path("multi-cased-l12-h768-a12")
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)

start_time = time.time()
data = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')
data.head(10)
data.info()
fig, ax = plt.subplots(figsize=(6, 15))
sns.countplot(y='language', hue='label', data=data)
dataset = data[['premise', 'hypothesis', 'label']]
multigenre_data = an_nlp.load_dataset(path='glue', name='mnli')
multigenre_data
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
adversarial_data = an_nlp.load_dataset(path='anli')
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
train = dataset.sample(frac=0.99)
val = dataset[~dataset.index.isin(train.index)]
train = pd.concat([train, multigenre_df, adversarial_df])
train.info()
tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(GCS_PATH_TO_SAVEDMODEL, "vocab.txt"), do_lower_case=True, split_on_punc=True)
print('Vocab size', len(tokenizer.vocab))
tokenizer.tokenize('and these comments were considered in formulat')
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocessing_text(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'[" "0-9]+', " ", s)
    s = s.rstrip().strip()
    return s

def encode_sentence(s, tokenizer):
    s = preprocessing_text(s)
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(glue_dict, tokenizer):
    sentence1 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(glue_dict['premise'])])
    sentence2 = tf.ragged.constant([encode_sentence(s, tokenizer) for s in np.array(glue_dict['hypothesis'])])
    
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
    
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    
    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()
    
    return {
        "input_word_ids": input_word_ids.to_tensor(),
        "input_mask": input_mask,
        "input_type_ids": input_type_ids
    }
glue_train = bert_encode(train, tokenizer)
glue_train_labels = train['label']

glue_val = bert_encode(val, tokenizer)
glue_val_labels = val['label']

for key, value in glue_train.items():
    print(f"{key:15s} shape: {value.shape}")
print(f"glue_train_labels shape: {glue_train_labels.shape}")
bert_config_file = os.path.join(GCS_PATH_TO_SAVEDMODEL, 'bert_config.json')
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
config_dict['attention_probs_dropout_prob'] = 0.1
config_dict['hidden_dropout_prob'] = 0.1

bert_config = bert.configs.BertConfig.from_dict(config_dict)
print(config_dict)
with strategy.scope():
    multi_bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=3, max_seq_length=None)

    checkpoint = tf.train.Checkpoint(model=bert_encoder)
    checkpoint.restore(os.path.join(GCS_PATH_TO_SAVEDMODEL, 'bert_model.ckpt')).assert_consumed()
epochs = 3
batch_size = 256

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)
with strategy.scope():
    optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    multi_bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)
checkpoint_path = os.path.join("/kaggle/working","tmp/cp.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor='val_accuracy',
    mode='max',
    save_weights_only=True,
    save_best_only=True)

multi_bert_classifier.fit(glue_train, glue_train_labels, validation_data=(glue_val, glue_val_labels), batch_size=batch_size, epochs=epochs)
test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
test.info()
data_language = dict(Counter(data['language']).most_common())
test_language = dict(Counter(test['language']).most_common())

language_name = np.concatenate([list(data_language.keys()), list(test_language.keys())])
language_name = list(set(language_name))

data_num = []
test_num = []
for index, value in enumerate(language_name):
    if value in data_language.keys():
        data_num.append(data_language[value])
    else:
        data_num.append(0)
    
    if value in test_language.keys():
        test_num.append(test_language[value])
    else:
        test_num.append(0)

language_num = pd.DataFrame({'language': language_name, 'data': data_num, 'test': test_num})

fig, ax = plt.subplots(figsize=(6, 8))
sns.set_color_codes("pastel")
sns.barplot(x='data' , y='language', data=language_num, label="data", color="b")
sns.set_color_codes("muted")
sns.barplot(x='test' , y='language', data=language_num, label="test", color="b")
ax.legend(ncol=2, loc="lower right", frameon=True)
sns.despine(left=True, bottom=True)
plt.show()
glue_test = bert_encode(test, tokenizer)
#multi_bert_classifier.load_weights(checkpoint_path)
predictions = multi_bert_classifier.predict(glue_test)
predictions = tf.math.argmax(predictions, axis=-1)
results = pd.DataFrame({'id': test['id'], 'prediction': predictions.numpy()})
results.to_csv('/kaggle/working/submission.csv', index=False)