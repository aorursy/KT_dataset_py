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
## Für den Code verwendete Quellen:
## https://medium.com/@utkarsh.kumar2407/named-entity-recognition-using-bidirectional-lstm-crf-9f4942746b3c
## https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede
## https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/
!pip install tensorflow==1.14
!pip install -q pyyaml h5py  # Required to save models in HDF5 format
!pip install keras==2.2.4
import pandas as pd
import numpy as np
import keras
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
!pip install git+https://www.github.com/keras-team/keras-contrib.git
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
import math
keras.__version__
tf.__version__
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

class_list = [
    'O',
'B_C','I_C', # 'companionship'
'B_T','I_T', # 'time'
'B_W','I_W', # 'weather'
'B_P','I_P', # 'position'
'B_M','I_M', # 'motivation'
'B_E','I_E', # 'emotional_state'
'B_A','I_A', # 'activity'
'B_V','I_V'# 'visited_before'
]

n_tags = len(labelclass_to_id)
#Padding 
max_len = 50
batch_size = 32
    
from keras.preprocessing.sequence import pad_sequences


elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, 'string')),
#                             "sequence_len": tf.constant(batch_size*[max_len])
                                    "sequence_len": tf.constant(32*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]
def create_model():
    input_text = Input(shape=(max_len,), dtype='string')
    embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
    x = Bidirectional(LSTM(units=300, return_sequences=True,
                           recurrent_dropout=0.6, dropout=0.6))(embedding)
#     x_rnn = Bidirectional(LSTM(units=150, return_sequences=True,
#                                recurrent_dropout=0.2, dropout=0.2))(x)
#     x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
#     crf = CRF(n_tags,sparse_target=True)
#     out = crf(x)  # output
    model = Model(input_text, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
#     model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])    
    return model
model_filepath = '../input/elmoweightscontest/weights_elmo'
model = create_model()
model.load_weights(model_filepath)
def test_model(test_tokens, test_labels):
    print('Training started. Depending on the test size this could take a couple of minutes.')
    y_test_pred_prob = model.predict(np.array(test_tokens))
    y_test_pred_sparse = y_test_pred_prob.argmax(axis=-1)
    y_test_pred = np_utils.to_categorical(np.array(y_test_pred_sparse), num_classes=n_tags)
    class_pred_previous = -1

    # compute confusion matrix
    conf_matrix = np.zeros((n_tags, n_tags))
    for i,tokens in enumerate(tqdm(test_tokens)):
        for tokens in test_tokens[i]:
            for j,_ in enumerate(tokens):
                class_true = test_labels[i,j]
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
                    f1_score = 2 * (precision_per_class[i]*recall_per_class[i])/(precision_per_class[i]+recall_per_class[i])
                    f1_list_per_class[i] = f1_score
        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)


    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = 2*(precision*recall)/(precision+recall)

    print('Precision: '+str(precision))
    print('Recall: '+str(recall))
    print('F1-measure: '+str(f1))
    #     return conf_matrix
    if conf_matrix.values[i,i] > 0:
                    precision_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[:,i])
                    recall_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[i,:])
                    f1_score = 2 * (precision_per_class[i]*recall_per_class[i])/(precision_per_class[i]+recall_per_class[i])
                    f1_list_per_class[i] = f1_score

def load_test_data(path_to_file):
    #Import Data
    filename = path_to_file

    with open(filename,'r', encoding='utf8') as infile:
        example_data = json.load(infile)
            
    for i,(k,v) in enumerate(example_data.items()):
        tokens = v.get('tokens')
        tokens = [token.lower() for token in tokens]
        example_data[k]['tokens'] = tokens

    tokens = [v['tokens'] for k,v in example_data.items()]
    test_labels = [v['labels'] for k,v in example_data.items()]  
    text =   [v['text'] for k,v in example_data.items()]  
    
    n_tags = len(labelclass_to_id)
    #Padding 
    max_len = 50
    batch_size = 32
    test_tokens = []
    for seq in tokens:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("PADword")
        test_tokens.append(new_seq)
    
    #Labels to ids
    label_ids_list = []
    for index in tqdm(range(len(test_labels)), desc="Converting tokens & labels to ids "):
        labels = test_labels[index]
        label_ids = [labelclass_to_id[label] for label in labels]
        while len(label_ids) < max_len:
                label_ids.append(0) 
        while len(label_ids) > max_len:
                label_ids.pop()

        label_ids_list.append(label_ids)

    test_labels = np.array(label_ids_list)
    ## the test size has to be a multiple of 32
    length_of_dataset_in_batches = math.floor(len(test_tokens) / 32)
    test_tokens = test_tokens[:32*length_of_dataset_in_batches]
    test_labels = test_labels[:32*length_of_dataset_in_batches]
    test_model(test_tokens, test_labels)
## Put path to your test file here
# test_file = "path/to/your/file"
test_file = "../input/traindata/train_data_ctxt_preliminary.json"
load_test_data(test_file)