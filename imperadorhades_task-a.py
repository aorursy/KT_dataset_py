import numpy as np 

import pandas as pd 

import os

import tensorflow as tf

import numpy as np

import pandas as pd

from transformers import BertTokenizer

from transformers import TFBertForSequenceClassification

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, f1_score
os.listdir("../input/nlp-unifor")
dataset = pd.read_csv("../input/nlp-unifor/olid-training-v1.0.tsv", sep="\t")

dataset.subtask_a = dataset.subtask_a.apply(lambda x: 1 if x=="OFF" else 0)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):

    return {

      "input_ids": input_ids,

      "token_type_ids": token_type_ids,

      "attention_mask": attention_masks,

           }, label



def get_bert_input(sentence, max_length = 20, tokenizer = tokenizer):

    return tokenizer.encode_plus(sentence,                      

                                 add_special_tokens = True, 

                                 max_length = max_length,

                                 truncation = True,

                                 pad_to_max_length = True, 

                                 return_attention_mask = True)



def get_dataset(dataset, max_length):

    input_ids_list = []

    token_type_ids_list = []

    attention_mask_list = []

    label_list = []



    for idx, row in dataset.iterrows():

        bert_input = get_bert_input(row.tweet, max_length = max_length)

        input_ids_list.append(bert_input['input_ids'])

        token_type_ids_list.append(bert_input['token_type_ids'])

        attention_mask_list.append(bert_input['attention_mask'])

        label_list.append([row.subtask_a])



    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)
train_size = 0.8

max_len = 100

batch_size = 64

train_cutoff = int(dataset.shape[0]*train_size)

train_dataset = get_dataset(dataset.iloc[:train_cutoff], max_len).shuffle(10000).batch(batch_size)

test_dataset = get_dataset(dataset.iloc[train_cutoff:], max_len).batch(batch_size)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath= "model.h5" ,save_best_only=True)





learning_rate = 2e-5

number_of_epochs = 2

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
bert_history = model.fit(train_dataset, 

                         epochs=number_of_epochs, 

                         validation_data=test_dataset,

                         callbacks=[model_checkpoint_callback])
y_pred = np.argmax(model.predict(test_dataset)[0], axis=1)

y_true = np.array(dataset.iloc[train_cutoff:].subtask_a)
print("classification report")

print(classification_report(y_true, y_pred, labels=[0,1], target_names=['NOT', 'OFF']))

print("confusion matrix")

print(confusion_matrix(y_true, y_pred))

print("f1_score")

print(f1_score(y_true, y_pred))
learning_rate = 2e-5

number_of_epochs = 2

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])



full_dataset = get_dataset(dataset, max_len).shuffle(10000).batch(batch_size)

bert_history = model.fit(full_dataset, 

                         epochs=number_of_epochs)
test_set = pd.read_csv("../input/nlp-unifor/testset-levela.tsv", sep="\t")

test_set["subtask_a"] = np.nan

test_ds = get_dataset(test_set, max_len).batch(batch_size)

y_pred = np.argmax(model.predict(test_ds)[0], axis=1)

test_set["subtask_a"] = y_pred
test_set.to_csv("subtask_a.csv", index=False)