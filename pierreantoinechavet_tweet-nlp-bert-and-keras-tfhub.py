# We use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

import tokenization

import matplotlib.pyplot as plt

import seaborn as sns
def modelConstruct(network_bert, maxlength=512):

    id_word = Input(shape=(maxlength,), dtype=tf.int32, name="input_word_ids")

    masked = Input(shape=(maxlength,), dtype=tf.int32, name="input_mask")

    id_segmented = Input(shape=(maxlength,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = network_bert([id_word , masked,id_segmented])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)    

    model = Model(inputs=[id_word , masked,id_segmented], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])    

    return model
def encoderbert(data, tokenizer, maxlength=512):     

    masked = []

    segmented = []

    token = []   

    for text in data:

        text = tokenizer.tokenize(text)  

        text = text[:maxlength-2]  

        segmented_id = [0] * maxlength

        sequence = ["[CLS]"] + text + ["[SEP]"]

        tokens = tokenizer.convert_tokens_to_ids(sequence)

        padlength = maxlength - len(sequence)

        masked_pad = [1] * len(sequence) + [0] * padlength

        tokens += [0] * padlength      

        token.append(tokens)

        masked.append(masked_pad)

        segmented.append(segmented_id)

    return np.array(token), np.array(masked), np.array(segmented)
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

network_bert = hub.KerasLayer(module_url, trainable=True)
df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})

df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})



print('Training Set Shape = {}'.format(df_train.shape))

print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))

print('Test Set Shape = {}'.format(df_test.shape))

print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))
df_train_fake = df_train[df_train['target'] == 1]

keyword_cnt_fake = df_train_fake.keyword.value_counts()

keyword_cnt_fake
ax = sns.countplot(x='target',  data=df_train)

plt.show()
keyword_cnt = df_train.keyword.value_counts()

keyword_cnt
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=df_train[df_train['target']==1]['text'].str.len()

ax1.hist(tweet_len,color='red')

ax1.set_title('disaster tweets')

tweet_len=df_train[df_train['target']==0]['text'].str.len()

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
missing_cols = ['keyword', 'location']



df_fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)



sns.barplot(x=df_train[missing_cols].isnull().sum().index, y=df_train[missing_cols].isnull().sum().values, ax=axes[0])

sns.barplot(x=df_test[missing_cols].isnull().sum().index, y=df_test[missing_cols].isnull().sum().values, ax=axes[1])



axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)

axes[0].tick_params(axis='x', labelsize=15)

axes[0].tick_params(axis='y', labelsize=15)

axes[1].tick_params(axis='x', labelsize=15)

axes[1].tick_params(axis='y', labelsize=15)



axes[0].set_title('Training Set', fontsize=13)

axes[1].set_title('Test Set', fontsize=13)



plt.show()



for df in [df_train, df_test]:

    for col in ['keyword', 'location']:

        df[col] = df[col].fillna(f'no_{col}')
df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')



df_fig = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=df_train.sort_values(by='target_mean', ascending=False)['keyword'],

              hue=df_train.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()
Data_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

Data_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
vocab_file = network_bert.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = network_bert.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = encoderbert(Data_train.text.values, tokenizer, maxlength=160)

test_input = encoderbert(Data_test.text.values, tokenizer, maxlength=160)

train_labels = Data_train.target.values
model = modelConstruct(network_bert, maxlength=160)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    batch_size=16

)



model.save('model.h5')
test_pred = model.predict(test_input)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)