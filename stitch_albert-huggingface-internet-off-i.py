import sys

sys.path.insert(0, "../input/transformers/transformers-master/")

!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



from transformers import *



import pandas as pd

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub

print(tf.__version__)
#import sys

#sys.path.append("../input/albert-tokenization/")

#import albert_tokenization



#BERT_PATH = '../input/albert-base-v2'

#vocab file has <unk> and <pad> while tokenizer outputs [Unk] and [PAD]

#tokenizer = albert_tokenization.FullTokenizer(BERT_PATH+'/30k-clean-Unkfix.vocab', True)



# the below will throw an auto-trackable error if we use hub.module. this is problem due to loading tf1 hub in tf 2. 

# notes that help to solve the problem: https://www.tensorflow.org/hub/common_issues

#albert_model = hub.load(BERT_PATH+'/2')
albert_path = '../input/albertlargev2huggingface/'

tokenizer = AlbertTokenizer.from_pretrained(albert_path, do_lower_case=True)

albert_model = TFAlbertModel.from_pretrained(albert_path)
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
def _get_segments(sentences):

    sentences_segments = []

    for sent in sentences:

      temp = []

      i = 0

      for token in sent.split(" "):

        temp.append(i)

        if token == "[SEP]":

          i += 1

      sentences_segments.append(temp)

    return sentences_segments



def _get_inputs(df,_maxlen,tokenizer,use_keras_pad=False):



    maxqnans = np.int((_maxlen-20)/2)

    pattern = '[^\w\s]+|\n' # remove everything including newline (|\n) other than words (\w) or spaces (\s)

    

    sentences = ["[CLS] " + " ".join(tokenizer.tokenize(qn)[:maxqnans]) +" [SEP] " 

              + " ".join(tokenizer.tokenize(ans)[:maxqnans]) +" [SEP] " 

              + " ".join(tokenizer.tokenize(title)[:10]) + " [SEP] "

              + " ".join(tokenizer.tokenize(cat)[:10]) +" [SEP]" 

                for (title,qn,ans,cat) 

                in 

              zip(df['question_title'].str.replace(pattern, '').values.tolist(),

              df['question_body'].str.replace(pattern, '').values.tolist(),

              df['answer'].str.replace(pattern, '').values.tolist(),

              df['category'].str.replace(pattern, '').values.tolist())]

              #train.head()[['question_title','question_body','answer','category']].values.tolist()]

    



    #generate masks

    # bert requires a mask for the words which are padded. 

    # Say for example, maxlen is 100, sentence size is 90. then, [1]*90 + [0]*[100-90]

    sentences_mask = [[1]*len(sent.split(" "))+[0]*(_maxlen - len(sent.split(" "))) for sent in sentences]

 

    #generate input ids  

    # if less than max length provided then the words are padded

    if use_keras_pad:

      sentences_padded = pad_sequences(sentences.split(" "), dtype=object, maxlen=10, value='[PAD]',padding='post')

    else:

      sentences_padded = [sent + " [PAD]"*(_maxlen-len(sent.split(" "))) if len(sent.split(" "))!=_maxlen else sent for sent in sentences ]

    

    #print([s.split(" ") for s in sentences_padded])

    sentences_converted = [tokenizer.convert_tokens_to_ids(s.split(" ")) for s in sentences_padded]

    

    #generate segments

    # for each separation [SEP], a new segment is converted

    sentences_segment = _get_segments(sentences_padded)



    genLength = set([len(sent.split(" ")) for sent in sentences_padded])

    if _maxlen < 20:

      raise Exception("max length cannot be less than 20")

    elif len(genLength)!=1: 

      print(genLength)

      raise Exception("sentences are not of same size")





    #convert list into tensor integer arrays and return it

    #return sentences_converted,sentences_segment, sentences_mask

    return [np.asarray(sentences_converted, dtype=np.int32), 

           np.asarray(sentences_mask, dtype=np.int32),

        np.asarray(sentences_segment, dtype=np.int32)]

    #return [tf.cast(sentences_converted,tf.int32), tf.cast(sentences_segment,tf.int32), tf.cast(sentences_mask,tf.int32)]
maxlen = 200

Xtr = _get_inputs(df=train,tokenizer=tokenizer,_maxlen=maxlen)

ytr = np.asarray(train.iloc[:,11:])
Xte = _get_inputs(df=test,_maxlen=maxlen, tokenizer = tokenizer )
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout, GlobalAveragePooling1D, LeakyReLU

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import regularizers

from sklearn.model_selection import KFold

from scipy.stats import spearmanr

import tensorflow.keras.backend as K



import warnings; warnings.simplefilter('ignore')
#method1: using tfhub - didnt work

#albert_inputs = dict(

#    input_ids=Xtr[0],

#    input_mask=Xtr[1],

#    segment_ids=Xtr[2])



#albert_model.signatures['tokens']
def build_model():

    token_inputs = Input((maxlen), dtype=tf.int32, name='input_word_ids')



    X = albert_model(token_inputs)[1] #pooled output

    #X = GlobalAveragePooling1D()(X)

    #X = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.01))(pooled_output)

    #X = LeakyReLU(alpha=0.01)(X)

    X = Dropout(0.2)(X)

    output_= Dense(30, activation='sigmoid', name='output')(X)



    bert_model2 = Model(token_inputs,output_)

    print(bert_model2.summary())

    

    bert_model2.compile(optimizer=Adam(learning_rate=0.000001), loss='binary_crossentropy')

    

    return bert_model2
### Albert - all parameters in the model summary are shown to be trainable. So finetuning is possible.

### Albert only requires input ids. Mask & Segments are not required.



class Rho_Calculator(tf.keras.callbacks.Callback):

    

    ## Not predicting on test data for each epoch. Its a bit of overkill and slows down the epoch completion

    

    def __init__(self, valid_data, batch_size=16, fold=None):



        self.valid_inputs = valid_data[0]

        self.valid_outputs = valid_data[1]

       

        self.batch_size = batch_size

        self.fold = fold

        

    def on_train_begin(self, logs={}):

        self.valid_predictions = []

        

    def on_epoch_end(self, epoch, logs={}):

        pred_output = self.model.predict(self.valid_inputs, batch_size=self.batch_size)

        

        self.valid_predictions.append(pred_output)

        

        pred_ = pd.DataFrame(pred_output)

        val_ = pd.DataFrame(self.valid_outputs)

        # take each column at a time. carry out correlation. average correlation for all 30 columns ignoring nan values

        rho_val = np.nanmean([spearmanr(val_.iloc[:,i].values,pred_.iloc[:,i].values).correlation for i in np.arange(len(pred_.columns))])

        

        #rho_val = compute_spearmanr( self.valid_outputs, np.average(self.valid_predictions, axis=0))

        

        print("\nvalidation rho: %.4f" % rho_val)

        

        if self.fold is not None:

            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
i=0

num_folds = 3

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

fold_score = []

#test_preds = np.zeros((Xte[0].shape[0],ytr.shape[1])) # mimic rows shape of test data, columns shape from train since, test will have any column for outputs

test_preds = []

for train_index,val_index in kfold.split(ytr):

    K.clear_session()

    #print(train_index)

    #print('\n')

    #print(val_index)

    i= i+1

    print('executing fold no: {}'.format(i))

    

    # train_index gets a random sample of rows for training

    # Xtr is a list contains 3 np arrays - ids, masks, segments so, using list comprehension to get the splits

    Xtr_fold = [arr[train_index] for arr in Xtr]

    ytr_fold = ytr[train_index]

    

    Xtr_val = [arr[val_index] for arr in Xtr]

    ytr_val = ytr[val_index]

    

    model = build_model()

    rho = Rho_Calculator(valid_data=(Xtr_val[0], ytr_val),batch_size=8,fold=None)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    model.fit(Xtr_fold[0],ytr_fold,epochs=10,batch_size = 8,validation_split=0.2,callbacks=[es,rho]) #,validation_data = (Xtr_val[0],ytr_val)

    



    # calcuate scores for test data

    test_preds.append(model.predict(Xte[0][:],batch_size=8))

sub.iloc[:, 1:] = np.average(test_preds,axis=0)

sub.to_csv('submission.csv', index=False)