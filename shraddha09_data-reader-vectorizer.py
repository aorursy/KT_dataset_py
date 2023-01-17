# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import time
import gc

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def read_SRA_data(file_path1, file_path2):
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    df1.rename(columns= {'student-ans':'premise', 'ref-ans':'hypothesis', 'student-ans-score':'target'}, inplace=True)
    df2.rename(columns= {'student-ans':'premise', 'ref-ans':'hypothesis', 'student-ans-score':'target'}, inplace=True)
    df1 = df1[['premise', 'hypothesis', 'target']]
    df2 = df2[['premise', 'hypothesis', 'target']]
    print('\nMerging df1 and df2...\n')
    main_df = pd.concat([df1,df2], axis=0, ignore_index=True)
    main_df['target'] = main_df['target'].astype('category')
    main_df = main_df.sample(axis=0, frac=1).reset_index(drop=True)
    print('Resulting data shape:',main_df.shape)
    print('Total number of responses: ',len(main_df))
    print('Categories to predict: ',main_df['target'].unique())
    return main_df

def read_SNLI_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['sentence1','sentence2','gold_label']]
    data.rename(columns= {'sentence1':'premise', 'sentence2':'hypothesis', 'gold_label':'target'}, inplace=True)
    data = data.loc[data['target'] != '-' ]
    data.dropna(axis=0, inplace=True)
    data['target'].replace({'entailment':'correct', 'neutral':'incorrect', 'contradiction':'contradictory'}, inplace=True)
    data['target'] = data['target'].astype('category')
    data = data.sample(axis=0, frac=1).reset_index(drop=True)
    print('Resulting data shape:',data.shape)
    print('Total number of responses: ',len(data))
    print('Categories to predict: ',data['target'].unique())
    return data

import pickle

def save_pickle(df,file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(df,f)
        
def load_pickle(file_path):
    with open(file_path,'rb') as f:
        return pickle.load(f)
    
def generate_glove_dict(word_embedding_path):
    glove_wordmap = {}
    with open(word_embedding_path, "r") as glove:
        for line in glove:
            name, vector = tuple(line.split(" ", 1))
            glove_wordmap[name] = np.fromstring(vector, sep=" ")
    return glove_wordmap
from nltk.corpus import stopwords

class WordEmbedding:
    
    def __init__(self, word_embedding_path, unknown_strategy='default'):
        self.stopwords = set(stopwords.words('english'))
        self.embed_model = self.get_glove_dict(word_embedding_path)
        self.embed_dim = len(self.embed_model['car'])
        print("Dimension of a word vector: {}".format(len(self.embed_model['car'])))
        
        if unknown_strategy == 'default':
            unknown = np.zeros(self.embed_dim)
            unknown[1] = 1
            self.embed_model['<UNK>'] = unknown
        elif unknown_strategy == 'random':
            np.random.seed(7)
            self.embed_model['<UNK>'] = np.random.uniform(-0.01, 0.01, 300).astype("float32")
        elif unknown_strategy == 'zeros':
            self.embed_model['<UNK>'] = np.zeros(self.embed_dim)
        else:
            self.embed_model['<UNK>'] = None
            
    def get_glove_dict(self,word_embedding_path):
        return load_pickle(word_embedding_path)
    
    def get_vector(self, word):
        if word in self.embed_model.keys():
            return self.embed_model[word]
        elif word.lower() in self.embed_model.keys():
            return self.embed_model[word.lower()]
        else:
            return self.embed_model['<UNK>']
        
import string
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

class Prepare:
    
    def __init__(self, df):
        self.stopwords = stopwords.words('english') 
        self.df = df
        self.word_tokenizer = TreebankWordTokenizer()
        self.tokenizer = Tokenizer()
        self.le = preprocessing.LabelEncoder()
        self.premise = self.valid_tokens(df['premise'].values)
        self.hypothesis = self.valid_tokens(df['hypothesis'].values)
        self.tokenizer.fit_on_texts(self.premise + self.hypothesis)
        self.vocab_size = len(self.tokenizer.word_index)+1
        self.vocab = self.tokenizer.word_index.items()
        self.max_len = 100
        self.premise, self.hypothesis, self.target = self.prep_df()
        
    def valid_tokens(self,texts):
        valid = []
        for text in texts:
            valid_text = [token for token in self.word_tokenizer.tokenize(text) if token.lower() not in self.stopwords ]
            valid.append(valid_text)
        return valid
                
    def prep_target(self, score):
        self.le.fit(score.unique())
        score = self.le.transform(score)
        return to_categorical(score)
        
    def prep_df(self):
        premise = self.tokenizer.texts_to_sequences(self.premise) 
        hypothesis = self.tokenizer.texts_to_sequences(self.hypothesis)
        #max1 = len(max(premise, key=len))
        #max2 = len(max(hypothesis, key=len))
        #self.max_len = max(max1, max2)
        premise = pad_sequences(premise, maxlen = self.max_len, padding='post')
        hypothesis = pad_sequences(hypothesis, maxlen = self.max_len, padding='post')
        target = self.prep_target(self.df['target'])
        return premise,hypothesis,target    
from tensorflow.compat.v1.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant

class SiameseModel:
    
    def __init__(self, word_embedding, data, use_cudnn_lstm=False, plot_model_architecture=True):
        self.hidden_units = 300
        self.embed_model = word_embedding
        self.input_dim = word_embedding.embed_dim
        self.vocab_size = data.vocab_size
        self.left = data.premise
        self.right = data.hypothesis
        self.target = data.target
        self.max_len = data.max_len
        self.dense_units = 300
        self.drop = 0.2
        self.name = '{}_glove{}_lstm{}_dense{}'.format(str(int(time.time())),
                                                        self.input_dim,self.hidden_units,self.dense_units)
        
        
        embedding_matrix = np.zeros((self.vocab_size, self.input_dim))
        for word, i in data.vocab:
            embedding_vector = self.embed_model.get_vector(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embed = layers.Embedding(input_dim=self.vocab_size, output_dim=self.input_dim, 
                                 embeddings_initializer=Constant(embedding_matrix), 
                                 input_length=self.max_len, mask_zero=True, trainable=False)
        
        translate = layers.TimeDistributed(layers.Dense(self.hidden_units, activation='relu'))

        if use_cudnn_lstm:
            lstm = layers.CuDNNLSTM(self.hidden_units, input_shape=(None, self.input_dim),unit_forget_bias=True,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer='l2', name='lstm_layer')
        else:
            lstm = layers.LSTM(self.hidden_units, input_shape=(None, self.input_dim), unit_forget_bias=True,
                               activation = 'relu',
                               kernel_initializer='he_normal',
                               kernel_regularizer='l2', name='lstm_layer', return_sequences=False)
            
        left_input = Input(shape=(self.max_len), name='input_1')
        right_input = Input(shape=(self.max_len), name='input_2')        

        embed_left = embed(left_input)
        embed_right = embed(right_input)
        print('embed:',embed_right.shape)
        
        prem = translate(embed_left)
        hypo = translate(embed_right)
        print('translate:',prem.shape)

        prem = lstm(prem)
        hypo = lstm(hypo)
        print('lstm:',hypo.shape)
        prem = layers.BatchNormalization()(prem)
        hypo = layers.BatchNormalization()(hypo)
        
        #l1_norm = lambda x: 1 - K.abs(x[0]-x[1])
        #merged = layers.Lambda(function=l1_norm, output_shape=lambda x: x[0],name='L1_distance')([prem, hypo])
        merged = layers.concatenate([prem, hypo])
        merged = layers.Dropout(self.drop)(merged)
        print('merged:', merged.shape)
        
        for i in range(3):
            merged = layers.Dense(2*self.dense_units, activation='relu')(merged)
            merged = layers.Dropout(self.drop)(merged)
            merged = layers.BatchNormalization()(merged)
            print('dense:',merged.shape)
            
        output = layers.Dense(3, activation='softmax', name='output_layer')(merged)
        print('output:',output.shape)
        
        self.model = Model(inputs=[left_input, right_input], outputs=output)

        self.compile()
                
        if plot_model_architecture:
            plot_model(self.model, show_shapes=True, to_file=self.name+'.png')
        
    def compile(self):
        optimizer = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer, metrics=['accuracy'])
        
    def fit(self, validation_split=0.3, epochs=5, batch_size=128, patience=2):
        left_data = self.left
        right_data = self.right
        target = self.target
        early_stopping = EarlyStopping(patience=patience, monitor='val_loss')
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1,  
                                            factor=0.5,
                                            min_lr=0.00001)
        callbacks = [early_stopping, learning_rate_reduction]
        
        self.history = self.model.fit([left_data, right_data], target, 
                                 validation_split=validation_split,
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks)        
        
    def predict(self, left_data, right_data):
        return self.model.predict([left_data, right_data])
    
    def evaluate(self, left_data, right_data, target, batch_size=128):
        return self.model.evaluate([left_data, right_data], target, batch_size=batch_size)
    
    def save_pretrained_weights(self, path='./model/pretrained_weights.h5'):
        self.model.save_weights(path)
        print('Save pretrained weights at location: ', path)
        
    def load_pretrained_weights(self, path='./model/pretrained_weights.h5'):
        self.model.load_weights(path, by_name=True, skip_mismatch=True)
        print('Loaded pretrained weights')
        self.compile()
        
    def save(self, model_folder=None):
        print('Saving model in SavedModel format ...')    
        if model_folder==None or not os.path.isdir(model_folder):
            model_folder = self.name
        os.mkdir(model_folder)
        self.model.save(model_folder)
        print('Saved model to disk')
        
    def load_activation_model(self):
        self.activation_model = Model(inputs=self.model.input[0], 
                                      outputs=self.model.get_layer('lstm_layer').output)
        
    def load(self, model_folder='./model/'):
        #use for encoder decoder alontg with load_activation
        # load json and create model
        json_file = open(model_folder + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_folder + 'model.h5')
        print('Loaded model from disk')
        
        self.model = loaded_model
        # loaded model should be compiled
        self.compile()
        self.load_activation_model()
        
    def visualize_metrics(self):
        epochs = len(self.history.history['accuracy'])
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        t = f.suptitle('Performance', fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)
        epoch_list = list(range(1,epochs+1))
        ax1.plot(epoch_list, self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(epoch_list, self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xticks(np.arange(0, epochs+1, 5))
        ax1.set_ylabel('Accuracy Value')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Accuracy')
        l1 = ax1.legend(loc="best")
        
        ax2.plot(epoch_list, self.history.history['loss'], label='Train Loss')
        ax2.plot(epoch_list, self.history.history['val_loss'], label='Validation Loss')
        ax2.set_xticks(np.arange(0, epochs+1, 5))
        ax2.set_ylabel('Loss Value')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss')
        l2 = ax2.legend(loc="best")
        plt.show()
        
word_embedding_path = '/kaggle/input/word-embedding/glove_vector_dict_300d.pickle'
snli_path = '/kaggle/input/word-embedding/snli_train_5l_df.pickle'
sra_path = '/kaggle/input/word-embedding/student_response_8k_df.pickle'
df = load_pickle(snli_path)
main_df = load_pickle(sra_path)

word_embedding = WordEmbedding(word_embedding_path)
prep = Prepare(df)
prep_main = Prepare(main_df)
gc.collect()
history = model.history
epochs = len(history.history['accuracy'])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
t = f.suptitle('Performance - Training on SRA', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)
epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 1))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 1))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.grid(True)
plt.savefig(str(round(time.time()))+'_Training_SRA.png')
plt.show()
print('Pretraining model on SNLI data...')
t1 = time.time()
siamese = SiameseModel(word_embedding, prep)
print(siamese.model.summary())
siamese.fit(epochs=10,batch_size=512)
t2 = time.time()
print('\tTook %f seconds'%(t2-t1))
siamese.save_pretrained_weights(path='snli_pretrained_weights.h5')
pretrained_weights = '/kaggle/input/word-embedding/snli_pretrained_weights.h5'
print('Training model on SRA data ...')
model = SiameseModel(word_embedding, prep_main)
model.load_pretrained_weights(path=pretrained_weights)
t1 = time.time()
model.fit(epochs=10, patience=3, batch_size=32)
t2 = time.time()
print('\tTooke %f seconds'%(t2 - t1))
model.model.save_weights('SRA_trained_model.h5')
test1a = '/kaggle/input/test-data/3waySciEnts_test_unseen_ans.csv'
test1b = '/kaggle/input/test-data/3wayBeetle_test_unseen_ans.csv'
test2a = '/kaggle/input/test-data/3waySciEnts_test_unseen_domain.csv'
test3a = '/kaggle/input/test-data/3wayBeetle_test_unseen_ques.csv'
test3b = '/kaggle/input/test-data/3waySciEnts_test_unseen_ques.csv'

test1 = read_SRA_data(test1a, test1b)
test3 = read_SRA_data(test3a, test3b)
test1 = Prepare(test1)
test3 = Prepare(test3)
model.evaluate(test1.premise, test1.hypothesis, test1.target)
model.evaluate(test3.premise, test3.hypothesis, test3.target)
p = Prepare(test)
model.predict(p.premise, p.hypothesis)
p.target
