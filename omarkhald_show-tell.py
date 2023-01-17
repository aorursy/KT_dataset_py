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
import pandas as pd 
import cv2
import numpy as np
import os
from glob import glob
import math
import matplotlib.pyplot as plt

import re
import html
import string
import unicodedata
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from nltk.translate.bleu_score import corpus_bleu
df  =pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv")
df2 = pd.read_csv("/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv")

def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    return re.sub(r'\d+', '', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
    """
    :param words:
    :type words:
    :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    or
    from spacy.lang.en.stop_words import STOP_WORDS
    :type stop_words:
    :return:
    :rtype:
    """
    return [word for word in words if word not in stop_words]


def stem_words(words):
    """Stem words in text"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_words(words):
    """Lemmatize words in text"""

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in text"""

    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
    return word_tokenize(text)

def normalize_text( text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    #words = text2words(text)
    #stop_words = stopwords.words('english')
    #words = remove_stopwords(words, stop_words)
    #words = stem_words(words)# Either stem ovocar lemmatize
    #words = lemmatize_words(words)
    #words = lemmatize_verbs(words)

    return text
  
def normalize_corpus(corpus):
    return [normalize_text(t) for t in corpus]
  
df['report'] = df[df.columns[1:]].apply(
    lambda x: ','.join(x.astype(str)),
    axis=1)
df['report'] = df['report'].apply(normalize_text)
df['report'] = 'starts '+df['report']+' end'
vocab_size = 10000
max_len = 260

tok = Tokenizer(num_words=vocab_size,  oov_token='UNK', )
tok.fit_on_texts(df['report'].tolist())
df2 = df2[df2['projection']=='Frontal']
df  =pd.merge(df,df2,  on=['uid'])
class det_gen(tensorflow.keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self,df, tok, max_len,images_path, dim=(256,256), batch_size=8):
        self.df=df
        self.dim = dim
        self.images_path = images_path
        self.tok= tok
        self.max_len = max_len
        self.batch_size = batch_size
        self.nb_iteration = math.ceil((self.df.shape[0])/self.batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.df=self.df.sample(frac=1)
    
    def load_img(self, img_path):
        
        img = cv2.imread(img_path)
        img =cv2.resize(img,(self.dim))
        
        
        return img
        
    
    def __getitem__(self, index):
        'Generate one batch of data'
        
        indicies = list(range(index*self.batch_size, min((index*self.batch_size)+self.batch_size ,(self.df.shape[0]))))
        
        images = []
        for img_path in self.df['filename'].iloc[indicies].tolist():
            img = self.load_img(os.path.join(self.images_path,img_path))
            images.append(img)
            
            
        
        
        x_batch = self.df['report'].iloc[indicies].tolist()
        
        x_batch_input = [sample[:-len(" <end>")] for sample in x_batch]
        
        x_batch_gt = [sample[len(" <start>"): ] for sample in x_batch]
        
        
        x_batch_input = np.array(pad_sequences( self.tok.texts_to_sequences (x_batch_input),
                          maxlen=self.max_len-1 ,
                          padding='post',
                          truncating='post'))
        
        x_batch_gt = np.array(pad_sequences( self.tok.texts_to_sequences (x_batch_gt),
                          maxlen=self.max_len-1 ,
                          padding='post',
                          truncating='post'))
        
        
        
        
        
        
        return [np.array(images), np.array(x_batch_input)] , np.array(x_batch_gt)   
validation_split= 0.2
images_path = "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/"
df = df.sample(frac=1)
df_train = df.iloc[:-int(df.shape[0]*validation_split)]
df_val   = df.iloc[-int(df.shape[0]*validation_split):]
train_dataloader =  det_gen(df_train, tok, max_len,images_path)
val_dataloader =  det_gen(df_val, tok, max_len,images_path)
## Input layers
img_input = layers.Input(shape= (256,256,3)) 
report_input= layers.Input(shape= (max_len-1,))

## Encoder ######################

Densenet_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights=None,
            input_shape=(256,256,3),
        )
number_of_encoder_layers=  len(Densenet_model.layers)

encoder_output = Densenet_model(img_input)
encoder_output = layers.Flatten()(encoder_output)

X_img = layers.Dropout(0.5)(encoder_output)
X_img = layers.Dense(300, use_bias = False, 
                        kernel_regularizer=regularizers.l2(1e-4),
                        name = 'dense_img')(X_img)
X_img = layers.BatchNormalization(name='batch_normalization_img')(X_img)
X_img = layers.Lambda(lambda x : K.expand_dims(x, axis=1))(X_img)

##decoder ########################

X_text = layers.Embedding(vocab_size, 300, mask_zero = True, name = 'emb_text')(report_input)
X_text = layers.Dropout(0.5)(X_text)

# Initial States


LSTMLayer = layers.LSTM(300, return_sequences = True, return_state = True, dropout=0.5, name = 'lstm')

# Take image embedding as the first input to LSTM
_, a, c = LSTMLayer(X_img)

A, _, _ = LSTMLayer(X_text, initial_state=[a, c])
output = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax',
                                 kernel_regularizer = regularizers.l2(1e-4), 
                                 bias_regularizer = regularizers.l2(1e-4)), name = 'time_distributed_softmax')(A)




model  = Model(inputs=[img_input, report_input], outputs=output, name='NIC_greedy_inference_v2')


##Inference models ################

#encoder_inference model
encoder_model = Model(img_input,[a,c])

# Decoder model ###################

a0 = layers.Input(shape=(300,))
c0 = layers.Input(shape=(300,))

A, alast, clast = LSTMLayer(X_text, initial_state=[a0, c0])
output = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax',
                                 kernel_regularizer = regularizers.l2(1e-4), 
                                 bias_regularizer = regularizers.l2(1e-4)), name = 'time_distributed_softmax')(A)



decoder_model = Model([report_input,a0,c0],[output,alast,clast])
model.summary()
plot_model(model)
epochs =5
lr=1e-4
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr))
hist = model.fit_generator( train_dataloader,validation_data = val_dataloader,epochs = epochs)
def tokens_to_text(tokens,tok,end_token='end'):
    sentence=""
    for token in tokens:
        if token ==0:
            break
        
        word = tok.index_word[token]
        
        if word==end_token:
            break
            
        sentence+= word+" "
        
    sentence = sentence.strip()
    
    return sentence


def greedy_inference(input_img, tok,encoder_model, decoder_model,max_len,start_token="start",end_token='end',decoder_type="GRU"):
    if decoder_type=='LSTM':
        a0,c0  =encoder_model(np.expand_dims(input_img,axis=0))
    elif decoder_type=='GRU': 
        hidden_layer  =encoder_model(np.expand_dims(input_img,axis=0))
        
    word = tok.word_index[start_token]
    
    words = []
    
    for index in range(max_len):
        if decoder_type=='LSTM':
            word_probs , a0,c0 = decoder_model.predict([[np.array([word]),a0,c0]])
        elif decoder_type=='GRU': 
            word_probs , hidden_layer = decoder_model.predict([[np.array([word]),hidden_layer]])
            hidden_layer=hidden_layer[0]
        
        word = np.argmax(word_probs)
        
        try:
            if tok.index_word[word]==end_token:
                break
        except:
            pass
        
        words.append(word)
        
    words = tokens_to_text(words,tok,end_token)
    return words
def get_predictions_from_data_loader(data_loader,tok,encoder_model, decoder_model,max_len,start_token="start"
                                     ,end_token='end', inference_type='greedy',decoder_type='GRU'):
    
    data_loader_iterator = data_loader.__iter__()
    
    pred_sentences = []
    Gt_sentences = []
    for index, (X,Y) in enumerate(data_loader_iterator):
        for img,_,sample_y in zip(X[0],X[1],Y):
            
            if inference_type=='greedy':
                pred_sentence = greedy_inference(img, tok,encoder_model, decoder_model,max_len,
                                                 start_token=start_token,end_token=end_token,decoder_type=decoder_type)
            
            GT_sentence   = tokens_to_text(sample_y,tok)
            
            pred_sentences.append(pred_sentence)
            Gt_sentences.append(GT_sentence)
        
        if index == data_loader.nb_iteration -1:
            break
        print("Done with batch number: {} ", index)
        
    return Gt_sentences, pred_sentences
def calculate_bleu_evaluation(GT_sentences, predicted_sentences):
    BLEU_1 = corpus_bleu(GT_sentences, predicted_sentences, weights=(1.0, 0, 0, 0))
    BLEU_2 = corpus_bleu(GT_sentences, predicted_sentences, weights=(0.5, 0.5, 0, 0))
    BLEU_3 = corpus_bleu(GT_sentences, predicted_sentences, weights=(0.3, 0.3, 0.3, 0))
    BLEU_4 = corpus_bleu(GT_sentences, predicted_sentences, weights=(0.25, 0.25, 0.25, 0.25))
    
    return BLEU_1,BLEU_2,BLEU_3,BLEU_4
def evaluate_from_dataloader(data_loader,tok,encoder_model, decoder_model,max_len,start_token='starts',end_token=' end', inference_type='greedy',decoder_type="LSTM"):
    Gt_sentences, pred_sentences = get_predictions_from_data_loader(data_loader,tok,encoder_model, decoder_model,max_len,start_token=start_token,end_token=end_token, inference_type=inference_type,decoder_type=decoder_type)
    BLEU_1,BLEU_2,BLEU_3,BLEU_4 = calculate_bleu_evaluation(Gt_sentences, pred_sentences)
    
    return BLEU_1,BLEU_2,BLEU_3,BLEU_4
BLEU_1,BLEU_2,BLEU_3,BLEU_4 =  evaluate_from_dataloader(val_dataloader,tok,encoder_model, decoder_model,max_len)
print("bleu 1 :"+BLEU_1)
print('-------')
print("bleu 2: "+BLEU_2)
print('-------')
print("bleu 3: "+BLEU_3)
print('-------')
print("bleu 4: "+BLEU_4)