!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

import tokenization



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/steam-reviews-dataset/steam_reviews.csv')

data.head()
data.describe()
data.recommendation.value_counts()
sizes = [data.recommendation.value_counts()[0], data.recommendation.value_counts()[1]]

labels = ['Recommended', 'Not Recommended']



explode = (0, 0.1)

fig1, ax1 = plt.subplots()

ax1.set_title('Games recommendation')

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)



ax1.axis('equal')  

plt.tight_layout()

plt.show()
data['hour_played_reviews'] = data.groupby('hour_played')['hour_played'].transform('count')

x = data.hour_played

y = data['hour_played_reviews']

fig = plt.figure(figsize = (13,8))

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.scatter(x,y)

ax.set_title('Dependence of the number of ratings on the duration of the game')

ax.set_xlabel('Hours played')

ax.set_ylabel('Number of reviews')
top_reviewed_games = data.title.value_counts()

print('Top 10 reviewed games:\n\n{}'.format(data.title.value_counts()[:10]))
data = data.assign(y = (data.recommendation == 'Recommended').astype(int))

data.head(3)
print(len(data)/2)

data_cut = data[0:25000] # We will use just a small portion of data 

data_cut.tail(1)         # because BERT with a full data size will work for a very long time
data_cut.review = [str(x) for x in data_cut.review.values] # So that there are no problems in the tokenizer
from sklearn.model_selection import train_test_split

X = data_cut.review

y = data_cut.y

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 42, test_size=0.50)

for each in [y_train, y_test]:

    print(f"y fraction = {each.mean():.4f}")
print('Train : {}, Test: {}'.format(len(X_train),len(X_test)))
#X_test = X_test[:-2] # if it's not equal

#y_test = y_test[:-2]

#X_train = X_train[:-1]

#y_train = y_train[:-1]

print('\n train X: {} \n train y: {} \n Val X: {} \n val y: {}'.format(len(X_train),len(y_train),len(X_test),len(y_test)))
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
def bert_encode(input_text, tokenizer, max_len = 512):

    token_input = [] 

    mask_input = []

    seg_input = []

    

    for text in input_text:

        text = tokenizer.tokenize(text)

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)      

        token_input.append(tokens + [0]*pad_len)

        mask_input.append([1]*len(input_sequence) + [0]*pad_len)

        seg_input.append([0] * max_len)

        

    return np.array(token_input), np.array(mask_input), np.array(seg_input)
def build_model(bert_layer, max_len = 512):

    input_word_ids = Input(shape=(max_len, ),dtype = tf.int32,name = 'input_words_ids')

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
%%time

train_input = bert_encode(X_train.values, tokenizer, max_len=160)

test_input = bert_encode(X_test.values, tokenizer, max_len=160)

train_labels = y_train.values
model = build_model(bert_layer, max_len=160)

model.summary()
%%time

train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    batch_size=16

)



model.save('model.h5')
prediction = model.predict(test_input)

preds = []

for x in prediction:

    preds.append(int(x.round()))



from sklearn.metrics import accuracy_score

print("Accuracy: ", accuracy_score(preds, y_test.values))