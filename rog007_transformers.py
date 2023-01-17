# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install transformers clean-text
train_path = '../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'



data = pd.read_csv(train_path)
print('Number of Records: {}, Number of features/columns: {}'.format(data.shape[0], data.shape[1]))
print('Null values: {}'.format(data.isnull().values.sum()))
target_columns = list(data.columns)[2:]

y_labels = data[target_columns].values
from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertModel

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tqdm import tqdm

from cleantext import clean
distil_bert = 'distilbert-base-uncased'



tokenizer = DistilBertTokenizer.from_pretrained(distil_bert, do_lower_case=True, add_special_tokens=True,

                                                max_length=128, pad_to_max_length=True)
def cleaning(text):

    return clean(text, no_line_breaks=True, no_urls=True, no_punct=True)



def tokenize(sentences, tokenizer):

    

    input_ids = []

    input_masks = []

    #input_segments = []

    

    for sentence in tqdm(sentences):

        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, 

                                       max_length=128, pad_to_max_length=True, 

                                       return_attention_mask=True, return_token_type_ids=True)

        

        input_ids.append(inputs['input_ids'])

        input_masks.append(inputs['attention_mask'])

        #input_segments.append(inputs['token_type_ids'])        

        

    return np.asarray(input_ids, dtype='int32'),np.asarray(input_masks, dtype='int32')
data['comment_text'] = data['comment_text'].apply(cleaning)

input_ids, input_masks = tokenize(data['comment_text'], tokenizer)
config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)



config.output_hidden_states = False



transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config=config)



input_ids_in = tf.keras.layers.Input(shape=(128,), name='input_token', dtype='int32')

input_masks_in = tf.keras.layers.Input(shape=(128,), name='masked_token', dtype='int32')



embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]

X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, 

                                                       return_sequences=True, 

                                                       dropout=0.1, 

                                                       recurrent_dropout=0.1))(embedding_layer)

X = tf.keras.layers.GlobalMaxPool1D()(X)

X = tf.keras.layers.Dense(50, activation='relu')(X)

X = tf.keras.layers.Dropout(0.2)(X)

X = tf.keras.layers.Dense(6, activation='sigmoid')(X)



model = tf.keras.models.Model(inputs=[input_ids_in, input_masks_in], outputs=X)



for layer in model.layers[:3]:

    layer.trainable = False
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
X_train_id, X_test_id, X_train_mask, X_test_mask, y_train, y_test = train_test_split(input_ids, 

                                                                                     input_masks, 

                                                                                     y_labels,

                                                                                     test_size=0.2, 

                                                                                     random_state=42)
hist = model.fit([X_train_id, X_train_mask], 

                 y_train, 

                 validation_data=([X_test_id, X_test_mask], y_test),

                 epochs=1,

                 batch_size=64)
# model.save_weights('toxix.h5')
sample_text = 'I hate you, you idiot!'

clean_txt = cleaning(sample_text)

input_ids_test, input_masks_test = tokenize(clean_txt, tokenizer)
preds = model.predict([input_ids_test, input_masks_test])[0]

prediction = target_columns[np.argmax(preds, axis=0)]

print(prediction)
sample_submission_path = '../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip'

sample_submission = pd.read_csv(sample_submission_path)

sample_submission.head()
test_path = '../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip'

df_test = pd.read_csv(test_path)
df_test.head()
ids = df_test['id']

X_t = df_test['comment_text'].apply(cleaning)

sub_input_ids, sub_input_masks = tokenize(X_t, tokenizer)
predictions = model.predict([sub_input_ids, sub_input_masks])
ids = pd.Series(ids)

y_preds = pd.DataFrame(predictions, columns=target_columns)
final_submission = pd.concat([ids, y_preds], axis=1)
final_submission.head()
final_submission.to_csv('submission.csv', index=False)