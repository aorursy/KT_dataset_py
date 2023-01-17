
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Activation, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

import transformers
from tokenizers import BertWordPieceTokenizer

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
class LoadingData():
            
    def __init__(self):
        train_file_path = os.path.join("..","input","nlp-benchmarking-data-for-intent-and-entity","benchmarking_data","Train")
        validation_file_path = os.path.join("..","input","nlp-benchmarking-data-for-intent-and-entity","benchmarking_data","Validate")
        
        self.train_df1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
        self.train_df2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')
        self.train_df2.toxic = self.train_df2.toxic.round().astype(int)

        self.train_df = pd.concat([self.train_df1[['comment_text', 'toxic']],self.train_df2[['comment_text', 'toxic']].query('toxic==1'),
                           self.train_df2[['comment_text', 'toxic']].query('toxic==0').sample(n=150000, random_state=3982)])
        
        self.validation_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
        
        self.test_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

load_data_obj = LoadingData()
load_data_obj.train_df

class BertModel(object):
    def __init__(self):
        self.tokenizer = None
        self.model_name = 'distilbert-base-multilingual-cased'
    def set_tokenizer(self):
        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(self.model_name)
        # Save it locally
        self.tokenizer.save_pretrained('.')
        # Load it with huggingface transformer library
        self.fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
        
bert_model_obj = BertModel()
bert_model_obj.set_tokenizer()
class PreprocessingBertData():
    def __init__(self):
        self.nb_epochs = 1
        self.batch_size = 16 * strategy.num_replicas_in_sync
        self.maxLen = 192
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        
    def encode_text(self,text, tokenizer, chunk_size=256, maxlen=512):
        tokenizer.enable_truncation(max_length=maxlen)
        tokenizer.enable_padding(max_length=maxlen)
        ids = list()
        for i in tqdm(range(0, len(text), chunk_size)):
            current_chunk = text[i:i+chunk_size].tolist()
            encs = tokenizer.encode_batch(current_chunk)
            ids.extend([enc.ids for enc in encs])
        return np.array(ids)
    
    def tokenize_data(self):
        self.x_train = self.encode_text(load_data_obj.train_df.comment_text.astype(str), tokenizer=bert_model_obj.fast_tokenizer, maxlen=self.maxLen)
        self.x_test = self.encode_text(load_data_obj.test_df.content.astype(str), tokenizer=bert_model_obj.fast_tokenizer, maxlen=self.maxLen)
        self.x_valid = self.encode_text(load_data_obj.validation_df.comment_text.astype(str), tokenizer=bert_model_obj.fast_tokenizer, maxlen=self.maxLen)

        self.y_train = load_data_obj.train_df.toxic.values
        self.y_valid = load_data_obj.validation_df.toxic.values
        
        
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
    def get_dataset(self):
        AUTO = tf.data.experimental.AUTOTUNE
        train_dataset = (tf.data.Dataset
                         .from_tensor_slices((self.x_train, self.y_train))
                         .repeat()
                         .shuffle(2048)
                         .batch(self.batch_size)
                         .prefetch(AUTO)
                        )
        valid_dataset = (tf.data.Dataset
                         .from_tensor_slices((self.x_valid, self.y_valid))
                         .batch(self.batch_size)
                         .cache()
                         .prefetch(AUTO)
                        )
        test_dataset = (tf.data.Dataset
                        .from_tensor_slices(self.x_test)
                        .batch(self.batch_size)
                       )
        return train_dataset,valid_dataset,test_dataset
    
preprocess_bert_data_obj = PreprocessingBertData()
preprocess_bert_data_obj.tokenize_data()
train_data,valid_data,test_data = preprocess_bert_data_obj.get_dataset()
train_len = preprocess_bert_data_obj.x_train.shape[0]
valid_len = preprocess_bert_data_obj.x_valid.shape[0]
class DesignModel():
    def __init__(self):
        self.model = None 
        self.train_history = None
        self.valid_history = None
        
    def bert_model(self,transformer,max_seq_length=512): 
        
        bert_inputs = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
        sequence_output = transformer(bert_inputs)[0]
        class_token = seq_output[:, 0, :]
        bert_out = tf.keras.layers.GlobalAveragePooling1D()(class_token)
        bert_out = tf.keras.layers.Dropout(0.2)(bert_out)
        bert_outputs = tf.keras.layers.Dense(1, activation="sigmoid")(bert_out)
        self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=bert_outputs)
        
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        self.model.summary()
    
    def model_train(self,batch_size,num_epoch):
        n_steps = train_len // batch_size
        print("Fitting to model")
        self.train_history = self.model.fit(train_data,steps_per_epoch=n_steps,validation_data=valid_data,epochs=num_epoch)        
        print("Model Training complete.")
        n_steps = valid_len // batch_size
        self.valid_history = self.model.fit(valid_data.repeat(),steps_per_epoch=n_steps,epochs=num_epoch)
max_len = 192
num_epoch = 5
batch_size = 16 * strategy.num_replicas_in_sync
model_obj = DesignModel()
with strategy.scope():
    transformer = (transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased'))
    model_obj.bert_model(transformer,max_seq_length=max_len)
    model_obj.model_train(batch_size,num_epoch)

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub['toxic'] = model_obj.model.predict(test_data, verbose=1)
sub.to_csv("submission.csv", index=False)
sub.head()