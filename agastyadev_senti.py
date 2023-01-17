# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import datetime





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")        

tokenizer.save_pretrained('.')
#############################################################################



import pandas as pd

import re

import string

from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from nltk.tokenize import word_tokenize

import numpy as np



#############################################################################



class PreProc:

    '''

    Objective: This class is for cleaning the data set.

    '''

    

    def __init__(self):

        self.df1=pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv",usecols=['comment_text','toxic'])

        print('Data loaded...')

        self.df1['toxic']=self.df1['toxic'].apply(lambda x: 1 if x > 0.5 else 0)

###    

    def rem_links(self,text):

        text=re.sub(pattern=r"http\S+",repl="",string=str(text))

        return text

    

    

    def rem_punct(self,text):

        text=text.translate(str.maketrans(dict.fromkeys(string.punctuation)))

        return text

    

    def rem_white(self,text):

        text=text.strip()

        return text

    

    def rem_line(self,text):

        text=text.replace('\n',' ')

        return text

    

    def number(self,text):

        text=re.sub(pattern=r"1234567890",repl="",string=str(text))

        return text

    

    def pipe_text(self,text):

        cleaner=[self.rem_links,self.rem_punct,self.rem_white,self.rem_line,self.number]

        for func in cleaner:

            text=func(text)



        return text

####     

    def token_it(self,data,name):

        

        data[name]=data[name].apply(lambda x: word_tokenize(str(x)))

        return data

    

    

    def dont_stop_me_now(self,data,name):

        data[name]=data[name].apply(lambda x: [word for word in x if word not in ENGLISH_STOP_WORDS])

        return data

    

    def lemmatize(self,data,name):

        lem=WordNetLemmatizer()

        data[name]=data[name].apply(lambda x: [lem.lemmatize(word) for word in x])

        return data

    

    def pipe_token(self,data,name):

        pre_process=[self.token_it, self.dont_stop_me_now, self.lemmatize]

        for func in pre_process:

            data=func(data,name)

        return data

        

#####

    def preprocess(self):

        print("Cleaning initiated...")

        self.df1['comment_text']=self.df1['comment_text'].apply(lambda x: self.pipe_text(x))



        self.df1=self.pipe_token(self.df1,'comment_text') 



        return self.df1

    

    
from transformers import AutoTokenizer

from tokenizers import BertWordPieceTokenizer

from sklearn.model_selection import train_test_split

import numpy as np





class Pre_Model:

    def __init__(self):

        cleaner = PreProc()

        self.data = cleaner.preprocess()

        print('Data Cleaned.')

        

    def encoding(self,texts,tokenize,chunk_size= 279, max_len=128):

        tokenize.enable_truncation(max_length=max_len)

        tokenize.enable_padding(max_length=max_len)

        all_ids = []

        for i in range(0,len(texts),chunk_size):

            text = texts[i:i+chunk_size:].tolist()

            encs = tokenize.encode_batch(text)



            all_ids.extend([enc.ids for enc in encs])



        return (np.array(all_ids))

    def preModel(self):

        bert_tokenizer = BertWordPieceTokenizer('vocab.txt',lowercase = True)

        x_train = self.encoding(self.data['comment_text'].astype(str), bert_tokenizer)

        y_train = self.data.toxic.values

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 666)

        return x_train, x_test, y_train, y_test

        
berted = Pre_Model()
x_train, x_test, y_train, y_test = berted.preModel()
import transformers

import keras 

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE





class Senti:

    def __init__(self):

        print("Initiated...")

        

        self.x_train,self.x_test,self.y_train, self.y_test = x_train, x_test, y_train, y_test

        print('Samples Loaded.')



        

    def modeling(self,max_len = 128):

        print('Constructing model...')

        transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-uncased')

        )

        inp = Input(shape = (max_len,),dtype = tf.int32)

        sequence_op = transformer_layer(inp)[0]

        cls_token = sequence_op[:,0,:]

        out = Dense(1, activation = 'sigmoid')(cls_token)

        model = Model(inputs = inp, outputs = out)

        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        print('Model constructed.')

        return model

    def training(self):

        

        model = self.modeling()

        call = [keras.callbacks.EarlyStopping(patience = 2)]

        class_weights=compute_class_weight('balanced',np.unique(self.y_train),self.y_train)

        print('\n Fitting Model...')

        history = model.fit(self.x_train, self.y_train, batch_size = 1923, epochs = 20,callbacks = call,shuffle = True, validation_data=(self.x_test,self.y_test))

        return history,model

        

    def predic(self,model):

        y_pred_train=[0 if o < 0.5 else 1 for o in model.predict(self.x_train)]

        y_pred_test=[0 if o < 0.5 else 1 for o in model.predict(self.x_test)]

        acc_tra,f1_tra,pre_tra,rec_tra=accuracy_score(self.y_train,y_pred_train),f1_score(self.y_train,y_pred_train),precision_score(self.y_train,y_pred_train),recall_score(self.y_train,y_pred_train)

        acc_test,f1_test,pre_test,rec_test=accuracy_score(self.y_test,y_pred_test),f1_score(self.y_test,y_pred_test),precision_score(self.y_test,y_pred_test),recall_score(self.y_test,y_pred_test)

        conf_train=confusion_matrix(self.y_train,y_pred_train)

        conf_test=confusion_matrix(self.y_test,y_pred_test)

        

        print('\n\n\nSTATSSSSSS BABYYYYY:\n\n')

        print('TRAINING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_train)

        print('Accuracy: ',acc_tra)

        print('F1_Score: ',f1_tra)

        print('Preision: ',pre_tra)

        print('Recall: ',rec_tra)

        print('ROC AND AUC', roc_auc_score(self.y_train, y_pred_train))



        

        print('\n\n\nTESTING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_test)

        print('Accuracy: ',acc_test)

        print('F1_Score: ',f1_test)

        print('Preision: ',pre_test)

        print('Recall: ',rec_test)

        print('ROC AND AUC', roc_auc_score(self.y_test, y_pred_test))



import datetime

start=datetime.datetime.now()

obj = Senti()
with strategy.scope():

    history, model = obj.training() 

end=datetime.datetime.now()
history.history.keys()
obj.graph(history)
obj.predic(model)
end-start
model.save_weights('Model_bench.h5')
import transformers

import keras 

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE





class Senti_v1:

    def __init__(self):

        print("Initiated...")

        berted = Pre_Model()

        self.x_train,self.x_test,self.y_train, self.y_test = x_train, x_test, y_train, y_test

        print('Samples Loaded.')



        

    def modeling(self,max_len = 128):

        print('Constructing model...')

        transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-uncased')

        )

        inp = Input(shape = (max_len,),dtype = tf.int32)

        sequence_op = transformer_layer(inp)[0]

        cls_token = sequence_op[:,0,:]

        out = Dense(1, activation = 'sigmoid')(cls_token)

        model = Model(inputs = inp, outputs = out)

        model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy'])

        print('Model constructed.')

        return model

    def training(self):

        

        model = self.modeling()

        call = [keras.callbacks.EarlyStopping(patience = 2)]

        class_weights=compute_class_weight('balanced',np.unique(self.y_train),self.y_train)

        print('\n Fitting Model...')

        history = model.fit(self.x_train, self.y_train, batch_size = 1923, epochs = 20,callbacks = call,shuffle = True, validation_data=(self.x_test,self.y_test))

        return history,model

    def graph(self, history):

        plt.plot(history.history['auc'])

        plt.plot(history.history['val_accuracy'])

        plt.title('model accuracy')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        

        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('model loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        

    def predic(self,model):

        y_pred_train=[0 if o < 0.5 else 1 for o in model.predict(self.x_train)]

        y_pred_test=[0 if o < 0.5 else 1 for o in model.predict(self.x_test)]

        acc_tra,f1_tra,pre_tra,rec_tra=accuracy_score(self.y_train,y_pred_train),f1_score(self.y_train,y_pred_train),precision_score(self.y_train,y_pred_train),recall_score(self.y_train,y_pred_train)

        acc_test,f1_test,pre_test,rec_test=accuracy_score(self.y_test,y_pred_test),f1_score(self.y_test,y_pred_test),precision_score(self.y_test,y_pred_test),recall_score(self.y_test,y_pred_test)

        conf_train=confusion_matrix(self.y_train,y_pred_train)

        conf_test=confusion_matrix(self.y_test,y_pred_test)

        

        print('\n\n\nSTATSSSSSS BABYYYYY:\n\n')

        print('TRAINING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_train)

        print('Accuracy: ',acc_tra)

        print('F1_Score: ',f1_tra)

        print('Preision: ',pre_tra)

        print('Recall: ',rec_tra)

        print('ROC AND AUC', roc_auc_score(self.y_train, y_pred_train))



        

        print('\n\n\nTESTING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_test)

        print('Accuracy: ',acc_test)

        print('F1_Score: ',f1_test)

        print('Preision: ',pre_test)

        print('Recall: ',rec_test)

        print('ROC AND AUC', roc_auc_score(self.y_test, y_pred_test))





import datetime

start=datetime.datetime.now()

obj_v1 = Senti_v1()
with strategy.scope():

    history_v1, model_v1 = obj_v1.training() 

end=datetime.datetime.now()
history_v1.history.keys()
graph(history_v1)
obj_v1.predic(model_v1)
end-start
model_v1.save_weights('Model_v1.h5')
import transformers

import keras 

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE





class Senti_v2:

    def __init__(self):

        print("Initiated...")

        berted = Pre_Model()

        self.x_train,self.x_test,self.y_train, self.y_test = x_train, x_test, y_train, y_test

        print('Samples Loaded.')

        

    def modeling(self,max_len = 128):

        print('Constructing model...')

        transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-uncased')

        )

        inp = Input(shape = (max_len,),dtype = tf.int32)

        sequence_op = transformer_layer(inp)[0]

        cls_token = sequence_op[:,0,:]

        out = Dense(1, activation = 'sigmoid')(cls_token)

        model = Model(inputs = inp, outputs = out)

        model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

        print('Model constructed.')

        return model

    def training(self):

        

        model = self.modeling()

        call = [keras.callbacks.EarlyStopping(patience = 2)]

        class_weights=compute_class_weight('balanced',np.unique(self.y_train),self.y_train)

        print('\n Fitting Model...')

        history = model.fit(self.x_train, self.y_train, batch_size = 1923, epochs = 20,callbacks = call,shuffle = True, validation_data=(self.x_test,self.y_test))

        return history,model

    def graph(self, history):

        plt.plot(history.history['accuracy'])

        plt.plot(history.history['val_accuracy'])

        plt.title('model accuracy')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        

        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('model loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        

    def predic(self,model):

        y_pred_train=[0 if o < 0.5 else 1 for o in model.predict(self.x_train)]

        y_pred_test=[0 if o < 0.5 else 1 for o in model.predict(self.x_test)]

        acc_tra,f1_tra,pre_tra,rec_tra=accuracy_score(self.y_train,y_pred_train),f1_score(self.y_train,y_pred_train),precision_score(self.y_train,y_pred_train),recall_score(self.y_train,y_pred_train)

        acc_test,f1_test,pre_test,rec_test=accuracy_score(self.y_test,y_pred_test),f1_score(self.y_test,y_pred_test),precision_score(self.y_test,y_pred_test),recall_score(self.y_test,y_pred_test)

        conf_train=confusion_matrix(self.y_train,y_pred_train)

        conf_test=confusion_matrix(self.y_test,y_pred_test)

        

        print('\n\n\nSTATSSSSSS BABYYYYY:\n\n')

        print('TRAINING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_train)

        print('Accuracy: ',acc_tra)

        print('F1_Score: ',f1_tra)

        print('Preision: ',pre_tra)

        print('Recall: ',rec_tra)

        print('ROC AND AUC', roc_auc_score(self.y_train, y_pred_train))



        

        print('\n\n\nTESTING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_test)

        print('Accuracy: ',acc_test)

        print('F1_Score: ',f1_test)

        print('Preision: ',pre_test)

        print('Recall: ',rec_test)

        print('ROC AND AUC', roc_auc_score(self.y_test, y_pred_test))





start=datetime.datetime.now()

obj_v2 = Senti_v2()
with strategy.scope():

    history_v2, model_v2 = obj_v2.training() 

end=datetime.datetime.now()
history_v2.history.keys()
graph(history_v2)
obj_v2.predic(model)
end-start
model_v2.save_weights('Model_v2.h5')
import transformers

import keras 

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE

model_path = 'final_model.h5'



class Senti_v3:

    def __init__(self):

        print("Initiated...")

        berted = Pre_Model()

        self.x_train,self.x_test,self.y_train, self.y_test = x_train, x_test, y_train, y_test

        print('Samples Loaded.')



        

    def modeling(self,max_len = 128):

        print('Constructing model...')

        transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-uncased')

        )

        inp = Input(shape = (max_len,),dtype = tf.int32)

        sequence_op = transformer_layer(inp)[0]

        cls_token = sequence_op[:,0,:]

        out = Dense(1, activation = 'sigmoid')(cls_token)

        model = Model(inputs = inp, outputs = out)

        model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = [tf.keras.metrics.AUC()])

        print('Model constructed.')

        return model

    def training(self):

        

        model = self.modeling()

        

        early_stop = tf.keras.callbacks.EarlyStopping(patience = 1)

        call = [early_stop]

        class_weights=compute_class_weight('balanced',np.unique(self.y_train),self.y_train)

        print('\n Fitting Model...')

        history = model.fit(self.x_train, self.y_train, batch_size = 1923, epochs = 20,callbacks = call,shuffle = True, validation_data=(self.x_test,self.y_test))

        return history,model

    def graph(history):

        plt.plot(history.history['auc'])

        plt.plot(history.history['val_auc'])

        plt.title('model accuracy')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()



        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('model loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

    def predic(self,model):

        y_pred_train=[0 if o < 0.5 else 1 for o in model.predict(self.x_train)]

        y_pred_test=[0 if o < 0.5 else 1 for o in model.predict(self.x_test)]

        acc_tra,f1_tra,pre_tra,rec_tra=accuracy_score(self.y_train,y_pred_train),f1_score(self.y_train,y_pred_train),precision_score(self.y_train,y_pred_train),recall_score(self.y_train,y_pred_train)

        acc_test,f1_test,pre_test,rec_test=accuracy_score(self.y_test,y_pred_test),f1_score(self.y_test,y_pred_test),precision_score(self.y_test,y_pred_test),recall_score(self.y_test,y_pred_test)

        conf_train=confusion_matrix(self.y_train,y_pred_train)

        conf_test=confusion_matrix(self.y_test,y_pred_test)

        

        print('\n\n\nSTATSSSSSS BABYYYYY:\n\n')

        print('TRAINING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_train)

        print('Accuracy: ',acc_tra)

        print('F1_Score: ',f1_tra)

        print('Preision: ',pre_tra)

        print('Recall: ',rec_tra)

        print('ROC AND AUC', roc_auc_score(self.y_train, y_pred_train))



        

        print('\n\n\nTESTING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_test)

        print('Accuracy: ',acc_test)

        print('F1_Score: ',f1_test)

        print('Preision: ',pre_test)

        print('Recall: ',rec_test)

        print('ROC AND AUC', roc_auc_score(self.y_test, y_pred_test))



import datetime

start=datetime.datetime.now()

obj_v3 = Senti_v3()
with strategy.scope():

    history_v3, model_v3 = obj_v3.training() 

end=datetime.datetime.now()
history_v3.history.keys()
obj_v3.graph(history_v3)
obj_v3.predic(model_v3)
model_v3.save_weights('Model_v3.h5')
end-start
import transformers

import keras 

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE

model_path = 'final_model.h5'



class Senti_v4:

    def __init__(self):

        print("Initiated...")

        berted = Pre_Model()

        self.x_train,self.x_test,self.y_train, self.y_test = x_train, x_test, y_train, y_test

        print('Samples Loaded.')

        print('ADASYN initialising...')

        ada = ADASYN()

        print('ADASYN initiated.')

        self.x_train, self.y_train = ada.fit_resample(self.x_train, self.y_train)

        self.x_test, self.y_test = ada.fit_resample(self.x_test, self.y_test)

        print('Resampled')

        

    def modeling(self,max_len = 128):

        print('Constructing model...')

        transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-uncased')

        )

        inp = Input(shape = (max_len,),dtype = tf.int32)

        sequence_op = transformer_layer(inp)[0]

        cls_token = sequence_op[:,0,:]

        out = Dense(1, activation = 'sigmoid')(cls_token)

        model = Model(inputs = inp, outputs = out)

        model.compile(loss = 'binary_crossentropy', optimizer = 'Nadam', metrics = [tf.keras.metrics.AUC()])

        print('Model constructed.')

        return model

    def training(self):

        

        model = self.modeling()

        

        early_stop = tf.keras.callbacks.EarlyStopping(patience = 1)

        call = [early_stop]

        class_weights=compute_class_weight('balanced',np.unique(self.y_train),self.y_train)

        print('\n Fitting Model...')

        history = model.fit(self.x_train, self.y_train, batch_size = 1123, epochs = 20,callbacks = call,shuffle = True, validation_data=(self.x_test,self.y_test))

        return history,model

    def graph(history):

        plt.plot(history.history['auc'])

        plt.plot(history.history['val_auc'])

        plt.title('model accuracy')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()



        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('model loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

    def predic(self,model):

        y_pred_train=[0 if o < 0.5 else 1 for o in model.predict(self.x_train)]

        y_pred_test=[0 if o < 0.5 else 1 for o in model.predict(self.x_test)]

        acc_tra,f1_tra,pre_tra,rec_tra=accuracy_score(self.y_train,y_pred_train),f1_score(self.y_train,y_pred_train),precision_score(self.y_train,y_pred_train),recall_score(self.y_train,y_pred_train)

        acc_test,f1_test,pre_test,rec_test=accuracy_score(self.y_test,y_pred_test),f1_score(self.y_test,y_pred_test),precision_score(self.y_test,y_pred_test),recall_score(self.y_test,y_pred_test)

        conf_train=confusion_matrix(self.y_train,y_pred_train)

        conf_test=confusion_matrix(self.y_test,y_pred_test)

        

        print('\n\n\nSTATSSSSSS BABYYYYY:\n\n')

        print('TRAINING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_train)

        print('Accuracy: ',acc_tra)

        print('F1_Score: ',f1_tra)

        print('Preision: ',pre_tra)

        print('Recall: ',rec_tra)

        print('ROC AND AUC', roc_auc_score(self.y_train, y_pred_train))



        

        print('\n\n\nTESTING DATA:\n\n\n')

        print('CONFUCIAN MATRIX: \n',conf_test)

        print('Accuracy: ',acc_test)

        print('F1_Score: ',f1_test)

        print('Preision: ',pre_test)

        print('Recall: ',rec_test)

        print('ROC AND AUC', roc_auc_score(self.y_test, y_pred_test))
import datetime

start=datetime.datetime.now()

obj_v4 = Senti_v4()
with strategy.scope():

    history_v4, model_v4 = obj_v4.training() 

end=datetime.datetime.now()
obj_v4.predic(model_v4)
end-start
model_v4.save_weights('Model_v4.h5')