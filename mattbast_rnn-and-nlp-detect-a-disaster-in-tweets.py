import numpy as np

import pandas as pd

import tensorflow as tf



import re

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer



import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', -1)
train_data = pd.read_csv(

    '/kaggle/input/nlp-getting-started/train.csv', 

    usecols=['text', 'target'], 

    dtype={'text': str, 'target': np.int64}

)



len(train_data)
train_data['text'].head().values
test_data = pd.read_csv(

    '/kaggle/input/nlp-getting-started/test.csv', 

    usecols=['text', 'id'], 

    dtype={'text': str, 'id': str}

)
indices = [4415, 4400, 4399,4403,4397,4396, 4394,4414, 4393,4392,4404,4407,4420,4412,4408,4391,4405]

train_data.loc[indices]
train_data.loc[indices, 'target'] = 0
indices = [6840,6834,6837,6841,6816,6828,6831]

train_data.loc[indices]
train_data.loc[indices, 'target'] = 0
indices = [601,576,584,608,606,603,592,604,591, 587]

train_data.loc[indices]
train_data.loc[indices, 'target'] = 1
indices = [3913,3914,3936,3921,3941,3937,3938,3136,3133,3930,3933,3924,3917]

train_data.loc[indices]
train_data.loc[indices, 'target'] = 0
indices = [246,270,266,259,253,251,250,271]

train_data.loc[indices]
train_data.loc[indices, 'target'] = 0
indices = [6119,6122,6123,6131,6160,6166,6167,6172,6212,6221,6230,6091,6108]

train_data.loc[indices]
train_data.loc[indices, 'target'] = 0
indices = [7435,7460,7464,7466,7469,7475,7489,7495,7500,7525,7552,7572,7591,7599]

train_data.loc[indices]
train_data.loc[indices, 'target'] = 0
val_data = train_data.tail(1500)

train_data = train_data.head(6113)
def remove_url(sentence):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'', sentence)
def remove_at(sentence):

    url = re.compile(r'@\S+')

    return url.sub(r'', sentence)
def remove_html(sentence):

    html = re.compile(r'<.*?>')

    return html.sub(r'', sentence)
def remove_emoji(sentence):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    return emoji_pattern.sub(r'', sentence)
def remove_stopwords(sentence):

    words = sentence.split()

    words = [word for word in words if word not in stopwords.words('english')]

    

    return ' '.join(words)
stemmer = SnowballStemmer('english')



def stem_words(sentence):

    words = sentence.split()

    words = [stemmer.stem(word) for word in words ]

    

    return ' '.join(words)
def clean_text(data):

    data['text'] = data['text'].apply(lambda x : remove_url(x))

    data['text'] = data['text'].apply(lambda x : remove_at(x))

    data['text'] = data['text'].apply(lambda x : remove_html(x))

    data['text'] = data['text'].apply(lambda x : remove_emoji(x))

    data['text'] = data['text'].apply(lambda x : remove_stopwords(x))

    data['text'] = data['text'].apply(lambda x : stem_words(x))

    

    return data
train_data = clean_text(train_data)

val_data = clean_text(val_data)

test_data = clean_text(test_data)



train_data['text'].head().values
def define_tokenizer(train_sentences, val_sentences, test_sentences):

    sentences = pd.concat([train_sentences, val_sentences, test_sentences])

    

    tokenizer = tf.keras.preprocessing.text.Tokenizer()

    tokenizer.fit_on_texts(sentences)

    

    return tokenizer

    

def encode(sentences, tokenizer):

    encoded_sentences = tokenizer.texts_to_sequences(sentences)

    encoded_sentences = tf.keras.preprocessing.sequence.pad_sequences(encoded_sentences, padding='post')

    

    return encoded_sentences
tokenizer = define_tokenizer(train_data['text'], val_data['text'], test_data['text'])



encoded_sentences = encode(train_data['text'], tokenizer)

val_encoded_sentences = encode(val_data['text'], tokenizer)

encoded_test_sentences = encode(test_data['text'], tokenizer)
tokenizer.word_index['disaster']
len(tokenizer.word_index)
print('Lower: ', tokenizer.get_config()['lower'])

print('Split: ', tokenizer.get_config()['split'])

print('Filters: ', tokenizer.get_config()['filters'])
embedding_dict = {}



with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:],'float32')

        embedding_dict[word] = vectors

        

f.close()
num_words = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((num_words, 100))



for word, i in tokenizer.word_index.items():

    if i > num_words:

        continue

    

    emb_vec = embedding_dict.get(word)

    

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec
tf_data = tf.data.Dataset.from_tensor_slices((encoded_sentences, train_data['target'].values))
def pipeline(tf_data, buffer_size=100, batch_size=32):

    tf_data = tf_data.shuffle(buffer_size)    

    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)

    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None],[]))

    

    return tf_data



tf_data = pipeline(tf_data, buffer_size=1000, batch_size=32)
print(tf_data)
tf_val_data = tf.data.Dataset.from_tensor_slices((val_encoded_sentences, val_data['target'].values))
def val_pipeline(tf_data, batch_size=1):        

    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)

    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None],[]))

    

    return tf_data



tf_val_data = val_pipeline(tf_val_data, batch_size=len(val_data))
print(tf_val_data)
embedding = tf.keras.layers.Embedding(

    len(tokenizer.word_index) + 1,

    100,

    embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),

    trainable = True

)
model = tf.keras.Sequential([

    embedding,

    tf.keras.layers.SpatialDropout1D(0.2),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(

    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

    optimizer=tf.keras.optimizers.Adam(0.0001),

    metrics=['accuracy', 'Precision', 'Recall']

)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),

]
history = model.fit(

    tf_data, 

    validation_data = tf_val_data,

    epochs = 50,

    callbacks = callbacks

)
metrics = model.evaluate(tf_val_data)



precision = metrics[2]

recall = metrics[3]

f1 = 2 * (precision * recall) / (precision + recall)



print('F1 score: ' + str(f1)) 
fig, axs = plt.subplots(1, 4, figsize=(20, 5))



axs[0].set_title('Loss')

axs[0].plot(history.history['loss'], label='train')

axs[0].plot(history.history['val_loss'], label='val')

axs[0].legend()



axs[1].set_title('Accuracy')

axs[1].plot(history.history['accuracy'], label='train')

axs[1].plot(history.history['val_accuracy'], label='val')

axs[1].legend()



axs[2].set_title('Precision')

axs[2].plot(history.history['Precision'], label='train')

axs[2].plot(history.history['val_Precision'], label='val')

axs[2].legend()



axs[3].set_title('Recall')

axs[3].plot(history.history['Recall'], label='train')

axs[3].plot(history.history['val_Recall'], label='val')

axs[3].legend()
predictions = model.predict(tf_val_data)

predictions = np.concatenate(predictions).round().astype(int)



val_data['predictions'] = predictions
false_positives = val_data[(val_data['predictions'] == 1) & (val_data['target'] == 0)]



print('Count of false positives: ' + str(len(false_positives)))
false_positives.head(10)
false_negatives = val_data[(val_data['predictions'] == 0) & (val_data['target'] == 1)]



print('Count of false negatives: ' + str(len(false_negatives)))
false_positives.tail(10)
tf_test_data = tf.data.Dataset.from_tensor_slices((encoded_test_sentences))
def test_pipeline(tf_data, batch_size=1):        

    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)

    tf_data = tf_data.padded_batch(batch_size, padded_shapes=([None]))

    

    return tf_data



tf_test_data = test_pipeline(tf_test_data)
predictions = model.predict(tf_test_data)
predictions = np.concatenate(predictions).round().astype(int)
submission = pd.DataFrame(data={'target': predictions}, index=test_data['id'])

submission.index = submission.index.rename('id')

submission.to_csv('submission.csv')
submission.head()
def compare_words(train_words, test_words):

    unique_words = len(np.union1d(train_words, test_words))

    matching = len(np.intersect1d(train_words, test_words))

    not_in_train = len(np.setdiff1d(test_words, train_words))

    not_in_test = len(np.setdiff1d(train_words, test_words))

    

    print('Count of unique words in both arrays: ' + str(unique_words))

    print('Count of matching words: ' + str(matching))

    print('Count of words in first array but not in second: ' + str(not_in_test))

    print('Count of words in second array but not first: ' + str(not_in_train))
compare_words(encoded_sentences, val_encoded_sentences)
compare_words(encoded_sentences, encoded_test_sentences)
# model = tf.keras.Sequential([

#     tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 200),

#     tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),

#     tf.keras.layers.Dense(1, activation='sigmoid')

# ])