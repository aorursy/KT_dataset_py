# libraries
import re
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import transformers as hf
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sarcasm_df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)
sarcasm_df.drop("article_link", axis=1, inplace=True)
sarcasm_df['headline'] = sarcasm_df['headline'].apply(lambda x: x.lower())
sarcasm_df['headline'] = sarcasm_df['headline'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
sarcasm_df['headline'] = sarcasm_df['headline'].apply((lambda x: re.sub('\s+',' ',x)))
sarcasm_df = sarcasm_df.sample(4000) # memory problems
sarcasm_df.head()
# train/test split
x_train, x_test, y_train, y_test = train_test_split(sarcasm_df['headline'].values, sarcasm_df['is_sarcastic'].values, train_size=0.5)
word_limit = 100
tokenizer = tf.keras.preprocessing.text.Tokenizer(word_limit, split=' ')
tokenizer.fit_on_texts(x_train)

train_sequence = tokenizer.texts_to_sequences(x_train)
train_sequence = tf.keras.preprocessing.sequence.pad_sequences(train_sequence)

test_sequence = tokenizer.texts_to_sequences(x_test)
test_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_sequence)
baseline_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(word_limit, 128, input_length=train_sequence.shape[1]),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.LSTM(32, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
baseline_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
baseline_model.fit(train_sequence, tf.cast(y_train, tf.int32), epochs=50, batch_size=100)
baseline_predictions = tf.squeeze(baseline_model.predict(test_sequence));
dbert_tokenizer = hf.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbert_model = hf.DistilBertModel.from_pretrained('distilbert-base-uncased')

def dbert_tokenization(word_list):
    tokens = list(map(lambda x: dbert_tokenizer.encode(x, add_special_tokens=True), word_list))
    padded = np.array([i + [0]*(word_limit-len(i)) for i in tokens])
    attention_mask = np.where(padded != 0, 1, 0)
    return padded, attention_mask

dbert_train_tokenized, train_mask = dbert_tokenization(x_train)
dbert_test_tokenized, test_mask = dbert_tokenization(x_test)
# train data
input_ids = torch.tensor(dbert_train_tokenized)  
attention_mask = torch.tensor(train_mask)
with torch.no_grad():
    train_last_hidden_states = dbert_model(input_ids, attention_mask=attention_mask)

train_dbert_features = train_last_hidden_states[0][:,0,:].numpy()

input_ids = torch.tensor(dbert_test_tokenized)  
attention_mask = torch.tensor(test_mask)
with torch.no_grad():
    test_last_hidden_states = dbert_model(input_ids, attention_mask=attention_mask)
    
test_dbert_features = test_last_hidden_states[0][:,0,:].numpy()
lr = LogisticRegression()
lr.fit(train_dbert_features, y_train)
bert_predictions = np.squeeze(lr.predict_proba(test_dbert_features)[:, 1])
{
    "bert": roc_auc_score(y_test, bert_predictions),
    "baseline": roc_auc_score(y_test, baseline_predictions),
}
