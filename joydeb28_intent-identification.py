# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import en_core_web_sm
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, Bidirectional, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
load_data_obj.train_data_frame
load_data_obj.validation_data_frame.head()
class Preprocessing():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.spacy_model = en_core_web_sm.load()
        self.tokenizer = None

    def createData(self):
        self.tokenizer = Tokenizer(num_words=None)
        self.max_len = 50
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(load_data_obj.train_data_frame['query'].tolist(),load_data_obj.train_data_frame['category'].tolist(),test_size=0.1)
        self.tokenizer.fit_on_texts(list(self.x_train) + list(self.x_valid))
        self.x_train = self.tokenizer.texts_to_sequences(self.x_train)
        self.x_valid = self.tokenizer.texts_to_sequences(self.x_valid)

        #zero pad the sequences
        self.x_train = pad_sequences(self.x_train, maxlen=self.max_len)
        self.x_valid = pad_sequences(self.x_valid, maxlen=self.max_len)
        self.y_train = to_categorical(self.y_train)
        self.y_valid = to_categorical(self.y_valid)
        self.word_index = self.tokenizer.word_index
        
    def getSpacyEmbeddings(self,sentneces):
        sentences_vectors = list()
        for item in sentneces:
            query_vec = self.spacy_model(item) 
            sentences_vectors.append(query_vec.vector)
        return sentences_vectors
    
    
    
    
        
preprocess_obj = Preprocessing()
preprocess_obj.createData()
preprocess_obj.y_train.shape
preprocess_obj.y_valid.shape
class DesignModel():
    def __init__(self):
        self.model = None
        self.x_train = preprocess_obj.x_train
        self.y_train = preprocess_obj.y_train
        self.x_valid = preprocess_obj.x_valid
        self.y_valid = preprocess_obj.y_valid
        
    def simple_rnn(self):
        self.model = Sequential()
        self.model.add(Embedding(len(preprocess_obj.word_index) + 1,100,input_length=preprocess_obj.max_len))
        self.model.add(SimpleRNN(100))
        self.model.add(Dense(len(load_data_obj.cat_to_intent), activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
    def model_train(self,batch_size,num_epoch):
        print("Fitting to model")
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=num_epoch, validation_data=[self.x_valid, self.y_valid])
        print("Model Training complete.")

    def save_model(self,model,model_name):    
        self.model.save("intent_models/"+model_name+".h5")
        print("Model saved to Model folder.")
model_obj = DesignModel()
model_obj.simple_rnn()
model_obj.model_train(64,5)
class Evaluation():
    def get_accuracy(self,actuals, predictions):
        acc = accuracy_score(actuals, predictions)
        return acc
class Prediction():
    def __init__(self):
        self.model = model_obj.model
        self.tokenizer = preprocess_obj.tokenizer
        self.max_len = preprocess_obj.max_len
        
    def predict_validation(self):
        self.xtest = load_data_obj.validation_data_frame['query'].tolist()
        self.ytest = load_data_obj.validation_data_frame['category'].tolist()
        self.xtest = self.tokenizer.texts_to_sequences(self.xtest)
        self.xtest = pad_sequences(self.xtest, maxlen=self.max_len)
        self.ypred = self.model.predict(self.xtest)
        self.ypred = [np.argmax(item) for item in self.ypred]
    
    def predict(self,query):
        query_seq = self.tokenizer.texts_to_sequences([query])
        query_pad = pad_sequences(query_seq, maxlen=self.max_len)
        pred = self.model.predict(query_pad)
        pred = np.argmax(pred)
        result = load_data_obj.cat_to_intent[pred]
        return result
pred_obj = Prediction()
pred_obj.predict_validation()
querylist = [
    'rate The Gift: Imagination and the Erotic Life of Property five stars',
     'table for Breadline Cafe in Minnesota next friday',
     'Will it be hot at 13:19 in De Funiak Springs Serbia and Montenegro ?',
     'Play some sixties songs on Google Music',
     'rate this textbook four out of 6']
for query in querylist:
    result = pred_obj.predict(query)
    print("Intent: "+str(result)+"\tQuery: "+str(query))
eval_obj = Evaluation()
acc = eval_obj.get_accuracy(pred_obj.ytest,pred_obj.ypred)
print("Auc: {:.2%}".format(acc))
