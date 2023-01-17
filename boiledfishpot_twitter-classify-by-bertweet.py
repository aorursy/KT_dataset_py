!pip install fairseq

!pip install fastBPE

!pip install transformers
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import tensorflow as tf

import tensorflow.keras.backend as K

from transformers import * 

from tensorflow.keras.layers import *
from nltk.tokenize import TweetTokenizer

from emoji import demojize

import re



tokenizer = TweetTokenizer()



def normalizeToken(token):

    lowercased_token = token.lower()

    if token.startswith("@"):

        return "@USER"

    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):

        return "HTTPURL"

    elif len(token) == 1:

        return demojize(token)

    else:

        if token == "’":

            return "'"

        elif token == "…":

            return "..."

        else:

            return token



def normalizeTweet(tweet):

    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))

    normTweet = " ".join([normalizeToken(token) for token in tokens])



    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")

    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")

    normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")



    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)

    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)

    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    

    return " ".join(normTweet.split())
from types import SimpleNamespace

from fairseq.data.encoders.fastbpe import fastBPE

from fairseq.data import Dictionary





class BERTweetTokenizer():

    

    def __init__(self,pretrained_path):

        bpe_dir = os.path.join(pretrained_path,"bpe.codes")

        vocab_dir = os.path.join(pretrained_path,"dict.txt")

        

        self.bpe = fastBPE(SimpleNamespace(bpe_codes= bpe_dir))

        self.vocab = Dictionary()

        self.vocab.add_from_file(vocab_dir)

        

        self.cls_token_id = 0

        self.pad_token_id = 1

        self.sep_token_id = 2

        

        self.pad_token = '<pad>'

        self.cls_token = '<s>'

        self.sep_token = '</s>'

        

    def bpe_encode(self,text):

        return self.bpe.encode(text)

    

    def encode(self,text,add_special_tokens=False):

        subwords = self.bpe.encode(text)

        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

        return input_ids

    

    def tokenize(self,text):

        return self.bpe_encode(text).split()

    

    def convert_tokens_to_ids(self,tokens):

        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()

        return input_ids

    

    #from: https://www.kaggle.com/nandhuelan/bertweet-first-look

    def decode_id(self,id):

        return self.vocab.string(id, bpe_symbol = '@@')

    

    def decode_id_nospace(self,id):

        return self.vocab.string(id, bpe_symbol = '@@ ')



tokenizer = BERTweetTokenizer('/kaggle/input/bertweet-base-transformers')
def read_train():

    train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

    train['text']=train['text'].astype(str)

    return train



def read_test():

    test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

    test['text']=test['text'].astype(str)

    return test



def read_submission():

    test=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

    return test

    

train_df = read_train()

test_df = read_test()

submission_df = read_submission()
def Data_encode(Dataset,MAX_LEN):

    ct = Dataset.shape[0]

    tokens = np.ones((ct,MAX_LEN),dtype='int32')

    masks = np.zeros((ct,MAX_LEN),dtype='int32')

    segs = np.zeros((ct,MAX_LEN),dtype='int32')



    for k in range(ct):        

        # INPUT_IDS

        text = normalizeTweet(Dataset.loc[k,'text'])

        enc = tokenizer.encode(text)                   

        if len(enc)<MAX_LEN-2:

            tokens[k,:len(enc)+2] = [0] + enc + [2]

            masks[k,:len(enc)+2] = 1

        else:

            tokens[k,:MAX_LEN] = [0] + enc[:MAX_LEN-2] + [2]

            masks[k,:MAX_LEN] = 1 



    return tokens,masks,segs



train_tokens,train_masks,train_segs = Data_encode(train_df,128)

test_tokens,test_masks,test_segs = Data_encode(test_df,128)
def f1(y_true,y_pred):

    def recall(y_true,y_pred):

        #TP:true==1&pred==1

        true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))

        #TP+FN:true==1

        possible_positives = K.sum(K.round(K.clip(y_true,0,1)))

        #recall = TP/(TP+FN)

        recall = true_positives / (possible_positives + K.epsilon())

        return recall

    

    def precision(y_true,y_pred):

        #TP:true==1&pred==1

        true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))

        #TP+FP:pred==1

        prediction_positives = K.sum(K.round(K.clip(y_pred,0,1)))

        #precision = TP/(TP+FP)

        precision = true_positives / (prediction_positives + K.epsilon())

        return precision

    

    precision = precision(y_true,y_pred)

    recall = recall(y_true,y_pred)

    return 2*(precision * recall)/(precision + recall + K.epsilon())
def build_model(MAX_LEN,PATH = '/kaggle/input/bertweet-base-transformers/'):

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    

    config_dir = os.path.join(PATH,'config.json')

    model_dir = os.path.join(PATH,'model.bin')

    config = RobertaConfig.from_pretrained(config_dir)

    bert_model = TFRobertaModel.from_pretrained(model_dir,config=config,from_pt=True)

    x,_ = bert_model(ids,attention_mask=att,token_type_ids=tok)



    out=Dense(1,activation='sigmoid')(x[:,0,:])

    

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=out)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(loss='binary_crossentropy',

                  optimizer=optimizer,

                  metrics=['accuracy',f1])



    return model



model = build_model(128)

model.summary()
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True, verbose=1)
train_inp = [train_tokens,train_masks,train_segs]

train_labels = train_df['target']

train_history = model.fit(

    train_inp, train_labels,

    validation_split=0.2,

    epochs=5,

    batch_size=16,

    verbose = 2,

    callbacks = [es]

)
test_inp = [test_tokens,test_masks,test_segs]

test_pred = model.predict(test_inp)

submission_df['target']=test_pred.round().astype(int)

submission_df.to_csv("submission.csv",index=False)