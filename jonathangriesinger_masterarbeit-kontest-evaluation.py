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
!pip install tensorflow==1.15.0
!pip install keras==2.2.4
!pip install git+https://www.github.com/keras-team/keras-contrib.git
#Iport Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras.utils import np_utils
from platform import python_version
import json 
from tqdm import tqdm 
from keras_contrib.layers import CRF
import keras 
from keras import backend as K
import math
#Hier Pfad zu den Testdaten angeben
filename = '../input/master/train_data_ctxt_preliminary.json'
with open(filename,'r', encoding='utf8') as infile:
    example_data = json.load(infile)
    
for i,(k,v) in enumerate(example_data.items()):
    tokens = v.get('tokens')
    tokens = [token.lower() for token in tokens]
    example_data[k]['tokens'] = tokens
    
train_tokens = [v['tokens'] for k,v in example_data.items()]
train_labels = [v['labels'] for k,v in example_data.items()]  
train_text =   [v['text'] for k,v in example_data.items()]  
#Label to IDS
labelclass_to_id={'O':0,
'B_C':1,'I_C':2, # 'companionship'
'B_T':3,'I_T':4, # 'time'
'B_W':5,'I_W':6, # 'weather'
'B_P':7,'I_P':8, # 'position'
'B_M':9,'I_M':10, # 'motivation'
'B_E':11,'I_E':12, # 'emotional_state'
'B_A':13,'I_A':14, # 'activity'
'B_V':15,'I_V':16# 'visited_before'
}

n_tags = len(labelclass_to_id)
#Padding 
max_len = 50
batch_size = 32

new_X = []
for seq in train_tokens:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_X.append(new_seq)
    

#Labels to ids
label_ids_list = []
for index in tqdm(range(len(train_labels)), desc="Converting tokens & labels to ids "):
    labels = train_labels[index]
    label_ids = [labelclass_to_id[label] for label in labels]
    while len(label_ids) < max_len:
            label_ids.append(0) 
    while len(label_ids) > max_len:
            label_ids.pop()
            
    label_ids_list.append(label_ids)

label_ids = np.array(label_ids_list)  




sess = tf.Session()
K.set_session(sess)
#Split set
X_test = new_X
y_test = label_ids
#Die Anzahl der Eingabe Sätze muss bei ELMo ein vielfache der Batchsize sein
x = math.floor(len(X_test)/32)
X_test = X_test[:x*batch_size]
y_test = y_test[:x*batch_size]
#ELMo Modell laden
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
# Erstellen des Modells 

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, 'string')),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]

input_text = Input(shape=(max_len,), dtype='string',name='Eingabe')
# batch_size = tf.shape(max_len)[0]

embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024),name='ELMo_Embedding')(input_text)
x = Bidirectional(LSTM(units=300, return_sequences=True,
                       recurrent_dropout=0.5, dropout=0.5),name='1.BiLSTM-Layer')(embedding)
x_rnn = Bidirectional(LSTM(units=300, return_sequences=True,
                           recurrent_dropout=0.5, dropout=0.5),name='2.BiLSTM-Layer')(x)
x = add([x, x_rnn],name='Residuale-Verbindung') 
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
model = Model(input_text, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")  
model.summary()

model.load_weights('../input/model-weights/Kontest_Weight.h5')
#Evaluation Testset
print('Evaluation gestaret...')
y_test_pred_prob = model.predict(np.array(X_test))
y_test_pred_sparse = y_test_pred_prob.argmax(axis=-1)
y_test_pred = np_utils.to_categorical(np.array(y_test_pred_sparse), num_classes=n_tags)

# compute confusion matrix
conf_matrix = np.zeros((n_tags, n_tags))
# for i,tokens in enumerate(test_tokens):
for i,tokens in enumerate(X_test):
    for j,_ in enumerate(tokens):
        class_true = y_test[i,j]
        class_pred = y_test_pred[i,j].argmax()
        conf_matrix[class_true,class_pred] += 1
names_rows = list(s+'_true' for s in labelclass_to_id.keys())
names_columns = list(s+'_pred' for s in labelclass_to_id.keys())
conf_matrix = pd.DataFrame(data=conf_matrix,index=names_rows,columns=names_columns)

# compute final evaluation measures
list_labels = list(s for s in labelclass_to_id.keys())
precision_per_class = np.zeros((n_tags,))
recall_per_class = np.zeros((n_tags,))
for i in range(n_tags):
    if conf_matrix.values[i,i] > 0:
        precision_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[:,i])
        recall_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[i,:])
    print(list_labels[i]+': \t Precision ' + str(precision_per_class[i]) + '\n \t Recall:' + str(recall_per_class[i]))
precision = np.mean(precision_per_class)
recall = np.mean(recall_per_class)
f1 = 2*(precision*recall)/(precision+recall)

print('Precision: '+str(precision))
print('Recall: '+str(recall))
print('F1-measure: '+str(f1))
