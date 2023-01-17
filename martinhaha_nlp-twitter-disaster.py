import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import collections
%%capture
!pip install -U keras-tuner
from kerastuner.tuners import RandomSearch
tf.random.set_seed(10)
np.random.seed(10)
df = pd.read_csv("../input/nlp-getting-started/train.csv")
df.head()
df.shape
df['target'].value_counts()
df.duplicated().sum()
df.isnull().sum()
plt.figure(figsize=(9,6))
sns.countplot(y=df.keyword, order = df.keyword.value_counts().iloc[:15].index)
plt.title('Top 15 keywords')
plt.show()
df_disaster = df[df['target'] == 1]
df_non_disaster = df[df['target'] == 0]
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.countplot(y=df_disaster.keyword, order = df_disaster.keyword.value_counts().iloc[:15].index)
plt.title('Top keywords for disaster tweets')
plt.subplot(122)
sns.countplot(y=df_non_disaster.keyword, order = df_non_disaster.keyword.value_counts().iloc[:15].index)
plt.title('Top keywords for non-disaster tweets')
plt.show()
df['keyword'] = df['keyword'].fillna('')
df['keyword'] = df['keyword'].str.lower()
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    text = text.lower()
    return text
df['messy_text'] = df['text']
df['original_text'] =  df['text'].apply(clean_text)
df['text'] = 'keyword: ' + df['keyword'] + '. ' + df['original_text']
df[df['id'] == 10832]['text']
df['text'].apply(len).max()
df['text'].apply(len).mean()
(df['text'].apply(len) < 150).sum()
train, validation = train_test_split(df, test_size=0.2)
train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
train.shape
validation.shape
vocab_size = 2500
embedding_dim = 6
max_length = 150
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train['text'])
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(train['text'])
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(validation['text'])
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12,kernel_regularizer = tf.keras.regularizers.l2(0.1))),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(12, activation='elu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
model.summary()
training_padded = np.array(training_padded)
training_labels = train['target'].values
validation_padded = np.array(validation_padded)
validation_labels = validation['target'].values
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 8)
history = model.fit(training_padded, training_labels, epochs=30, 
                    validation_data=(validation_padded, validation_labels),
                   callbacks=[early_stopping_cb],verbose=0)
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
plot_graphs(history, 'accuracy')
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, hp.Int('embedding dim',min_value=6, max_value=12,step=4), input_length=max_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('unit1',min_value=12,max_value=24,step=6), 
                                                   kernel_regularizer = tf.keras.regularizers.l2(
                                                       hp.Choice('regularizer',values=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]))))),
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout_rate1',values=[0.01, 0.05, 0.1, 0.15, 0.2]))),
    model.add(tf.keras.layers.Dense(units=hp.Int('unit3',min_value=12,max_value=36,step=12),
                           activation='elu'))
    model.add(tf.keras.layers.Dropout(hp.Choice('dropout_rate2',values=[0.01, 0.05, 0.1, 0.15, 0.2])))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(hp.Choice('learning_rate',values=[0.01, 0.005, 0.001])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    return model
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=6,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')
tuner.search(training_padded, training_labels,
             validation_data=(validation_padded, validation_labels), verbose = 0,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
models = tuner.get_best_models(num_models=1)
parameters = tuner.get_best_hyperparameters(1)[0]
model_final = models[0]
model_final.summary()
tf.keras.utils.plot_model(model_final, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
parameters.values
model_final_use = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, parameters.values['embedding dim'], input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(parameters.values['unit1'],
                                                       kernel_regularizer = tf.keras.regularizers.l2(parameters.values['regularizer']))),
    tf.keras.layers.Dropout(parameters.values['dropout_rate1']),
    tf.keras.layers.Dense(parameters.values['unit3'], activation='elu'),
    tf.keras.layers.Dropout(parameters.values['dropout_rate2']),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_final_use.compile(loss='binary_crossentropy',optimizer= tf.keras.optimizers.RMSprop(learning_rate=parameters.values['learning_rate']),metrics=['accuracy'])
model_final_use.summary()
model_final_use.save_weights('model.h5')
history2 = model_final_use.fit(training_padded, training_labels, epochs=30, 
                    validation_data=(validation_padded, validation_labels),
                   callbacks=[early_stopping_cb],verbose=0)
plot_graphs(history2, 'accuracy')
def transfer_text(text_array,vocab_size = 2500, max_length = 150, trunc_type='post', padding_type='post', oov_tok = "<OOV>"):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(text_array)
    word_index = tokenizer.word_index
    training_sequences = tokenizer.texts_to_sequences(text_array)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return tokenizer,np.array(training_padded)
tokenizer,final_text = transfer_text(df['text'])
model_final_use.load_weights('model.h5')
history_test = model_final_use.fit(final_text, df['target'], epochs=8, verbose = 0)
history_test.history
training_result = model_final_use.predict_classes(final_text)
training_result[7580]
model_final_use.evaluate(final_text, df['target'])
word_index = tokenizer.word_index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_sentence(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
import io 
out_v = io.open('vecs.tsv', 'w', encoding = 'utf-8')
out_m = io.open('meta.tsv', 'w', encoding = 'utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
test = pd.read_csv("../input/nlp-getting-started/test.csv")
test.shape
test.head()
test['keyword'] = test['keyword'].fillna('')
test['keyword'] = test['keyword'].str.lower()
test['messy_text'] = test['text']
test['original_text'] =  test['text'].apply(clean_text)
test['text'] = 'keyword: ' + test['keyword'] + '. ' + test['original_text']
test['text'][0]
test_sequences = tokenizer.texts_to_sequences(test['text'])
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission.head()
sample_submission['target'] = model_final_use.predict_classes(test_padded)
sample_submission['target'].head()
sample_submission.to_csv('submission_final.csv',index=False)
