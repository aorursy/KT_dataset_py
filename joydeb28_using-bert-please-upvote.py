!pip install bert-for-tf2
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Activation, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import bert
from tqdm import tqdm
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)
# Params for bert model and tokenization
class LoadingData():
            
    def __init__(self):
        train_file_path = os.path.join("..","input","nlp-benchmarking-data-for-intent-and-entity","benchmarking_data","Train")
        validation_file_path = os.path.join("..","input","nlp-benchmarking-data-for-intent-and-entity","benchmarking_data","Validate")
        
        self.train_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
        self.validation_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
        self.test_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
        self.train_df = self.train_df.sample(frac = 1)

load_data_obj = LoadingData()
load_data_obj.train_df.head()
class BertModel(object):
    
    def __init__(self):
        
        self.max_len = 128
        #bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
        bert_path = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2"
        FullTokenizer=bert.bert_tokenization.FullTokenizer
        
        self.bert_module = hub.KerasLayer(bert_path,trainable=True)

        self.vocab_file = self.bert_module.resolved_object.vocab_file.asset_path.numpy()

        self.do_lower_case = self.bert_module.resolved_object.do_lower_case.numpy()

        self.tokenizer = FullTokenizer(self.vocab_file,self.do_lower_case)
        
    def get_masks(self,tokens, max_seq_length):
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

    def get_segments(self,tokens, max_seq_length):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))
    def get_ids(self,tokens, tokenizer, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens,)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids
    def create_single_input(self,sentence,maxlen):

        stokens = self.tokenizer.tokenize(sentence)

        stokens = stokens[:maxlen]

        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        ids = self.get_ids(stokens, self.tokenizer, self.max_len)
        masks = self.get_masks(stokens, self.max_len)
        segments = self.get_segments(stokens, self.max_len)

        return ids,masks,segments

    def create_input_array(self,sentences):
        
        input_ids, input_masks, input_segments = [], [], []

        for sentence in tqdm(sentences,position=0, leave=True):
            ids,masks,segments=self.create_single_input(sentence,self.max_len-2)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
            
        tensor = [np.asarray(input_ids, dtype=np.int32), 
                np.asarray(input_masks, dtype=np.int32), 
                np.asarray(input_segments, dtype=np.int32)]
        return tensor

class PreprocessingBertData():
    
    def prepare_data_x(self,train_sentences):
        x = bert_model_obj.create_input_array(train_sentences)
        return x
    
    def prepare_data_y(self,train_labels):
        y = list()
        for item in train_labels:
            label = item
            y.append(label)
        y = np.array(y)
        return y
        
bert_model_obj = BertModel()
list_classes = ["toxic"]

train_sentences = load_data_obj.train_df["comment_text"].values
train_labels = load_data_obj.train_df["toxic"].values

preprocess_bert_data_obj = PreprocessingBertData()
x = preprocess_bert_data_obj.prepare_data_x(train_sentences)
y = preprocess_bert_data_obj.prepare_data_y(train_labels)

train_input_ids, train_input_masks, train_segment_ids = x
train_labels = y
class DesignModel():
    def __init__(self):
        self.model = None        
        self.train_data = [train_input_ids, train_input_masks, train_segment_ids]
        self.train_labels = train_labels
        
    def bert_model(self,max_seq_length): 
        in_id = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
        in_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_masks")
        in_segment = Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
        
        bert_inputs = [in_id, in_mask, in_segment]
        pooled_output, sequence_output = bert_model_obj.bert_module(bert_inputs)
        
        x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=out)
        
        self.model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

        
        self.model.summary()
    
    def model_train(self,batch_size,num_epoch):
        n_steps = len(self.train_data) // batch_size
        print("Fitting to model")
        
        self.model.fit(self.train_data,self.train_labels,steps_per_epoch=n_steps,epochs=num_epoch,batch_size=batch_size,validation_split=0.2,shuffle=True)
        
        print("Model Training complete.")
model_obj = DesignModel()
model_obj.bert_model(bert_model_obj.max_len)
model_obj.model_train(32,1)

test_sentences = load_data_obj.test_df["content"].values
inputs = preprocess_bert_data_obj.prepare_data_x(test_sentences)
result = model_obj.model.predict(inputs,verbose=1)
print(result[:5])
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub['toxic'] = result
sub.to_csv('submission.csv', index=False)
sub.head()
