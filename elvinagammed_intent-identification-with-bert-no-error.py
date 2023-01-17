!pip install bert-for-tf2
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import os

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

        category_id = 0

        self.cat_to_intent = {}

        self.intent_to_cat = {}

        

        for dirname, _, filenames in os.walk(train_file_path):

            for filename in filenames:

                file_path = os.path.join(dirname, filename)

                intent_id = filename.replace(".json","")

                self.cat_to_intent[category_id] = intent_id

                self.intent_to_cat[intent_id] = category_id

                category_id+=1

        print(self.cat_to_intent)

        print(self.intent_to_cat)

        '''Training data'''

        training_data = list() 

        for dirname, _, filenames in os.walk(train_file_path):

            for filename in filenames:

                file_path = os.path.join(dirname, filename)

                intent_id = filename.replace(".json","")

                training_data+=self.make_data_for_intent_from_json(file_path,intent_id,self.intent_to_cat[intent_id])

        self.train_data_frame = pd.DataFrame(training_data, columns =['query', 'intent','category'])   

        

        self.train_data_frame = self.train_data_frame.sample(frac = 1)





        

        '''Validation data'''

        validation_data = list()    

        for dirname, _, filenames in os.walk(validation_file_path):

            for filename in filenames:

                file_path = os.path.join(dirname, filename)

                intent_id = filename.replace(".json","")

                validation_data +=self.make_data_for_intent_from_json(file_path,intent_id,self.intent_to_cat[intent_id])                

        self.validation_data_frame = pd.DataFrame(validation_data, columns =['query', 'intent','category'])



        self.validation_data_frame = self.validation_data_frame.sample(frac = 1)

        

        

    def make_data_for_intent_from_json(self,json_file,intent_id,cat):

        json_d = json.load(open(json_file))         

        

        json_dict = json_d[intent_id]



        sent_list = list()

        for i in json_dict:

            each_list = i['data']

            sent =""

            for i in each_list:

                sent = sent + i['text']+ " "

            sent =sent[:-1]

            for i in range(3):

                sent = sent.replace("  "," ")

            sent_list.append((sent,intent_id,cat))

        return sent_list

            
load_data_obj = LoadingData()
load_data_obj.train_data_frame.head()
load_data_obj.validation_data_frame.head().values
class BertModel(object):

    

    def __init__(self):

        

        self.max_len = 128

        bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

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
train_sentences = load_data_obj.train_data_frame["query"].tolist()

train_labels = load_data_obj.train_data_frame["category"].tolist()



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

        out = tf.keras.layers.Dense(len(load_data_obj.cat_to_intent), activation="softmax", name="dense_output")(x)

        self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=out)

        

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),

                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

                           metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

        

        self.model.summary()

    

    def model_train(self,batch_size,num_epoch):

        print("Fitting to model")

        

        self.model.fit(self.train_data,self.train_labels,epochs=num_epoch,batch_size=batch_size,validation_split=0.2,shuffle=True)

        

        print("Model Training complete.")



    def save_model(self,model,model_name):    

        self.model.save(model_name+".h5")

        print("Model saved to Model folder.")
model_obj = DesignModel()

model_obj.bert_model(bert_model_obj.max_len)

model_obj.model_train(32,1)
model_obj.save_model(model_obj.model,"bert")
class Evaluation():

    def get_accuracy(self,actuals, predictions):

        acc = accuracy_score(actuals, predictions)

        return acc
class Prediction():

    def __init__(self):

        self.model = model_obj.model

        

    def predict_validation(self):

        valid_sentences = load_data_obj.validation_data_frame["query"].tolist()

        valid_labels = load_data_obj.validation_data_frame["category"].tolist()



        preprocess_bert_data_obj = PreprocessingBertData()

        val_x = preprocess_bert_data_obj.prepare_data_x(valid_sentences)

        prediction_labels = list(self.model.predict(val_x).argmax(axis=-1))

        return valid_labels,prediction_labels

        

    

    def predict(self,query):

        query_seq = bert_model_obj.create_input_array([query])

        pred = self.model.predict(query_seq)

        pred = np.argmax(pred)

        result = load_data_obj.cat_to_intent[pred]

        return result
pred_obj = Prediction()

#pred_obj.predict_validation()
querylist = [['I want to see Medal for the General', 'SearchScreeningEvent', 1],

       ['Book a reservation for 5 people at the top-rated brasserie restaurant',

        'BookRestaurant', 5],

       ['Can I put this tune onto my sin estres playlist?',

        'AddToPlaylist', 6],

       ['add the artist Pete Murray to my relaxing playlist',

        'AddToPlaylist', 6],

       ['Book me a reservation for a party of 3 at a pub in Northern Mariana Islands',

        'BookRestaurant', 5]]

for query in querylist:

    result = pred_obj.predict(query[0])

    print("Predicted Intent: "+str(result)+"\tActual Intent: "+(load_data_obj.cat_to_intent[query[2]])+"\tQuery: "+str(query[0]))

eval_obj = Evaluation()

ytest,ypred = pred_obj.predict_validation()

#print(ytest,ypred)

acc = eval_obj.get_accuracy(ytest,ypred)

print("Auc: {:.2%}".format(acc))
