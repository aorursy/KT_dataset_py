import os, re, string

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import random



#!pip install pySpellChecker

#from spellchecker import SpellChecker               # SpellChecker

from tqdm._tqdm_notebook import tqdm_notebook       # Get progress bar when using pandas

tqdm_notebook.pandas()



seed = 9870

def seeder(seed):

    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)



print(os.listdir('../input'))
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

sample = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train.head()
test.head()
train.describe(include=['O','float','int'])
test.describe(include=['O','float','int'])
train.info()
test.info()
sns.countplot(train.text.duplicated())
# Deleting duplicate tweets



duplicate_index = train[train.text.duplicated()].index

train.drop(index = duplicate_index, inplace = True)

train.reset_index(drop = True, inplace = True)
shortforms =   {"ain't": "am not",

    "aren't": "are not",

    "can't": "cannot",

    "can't've": "cannot have",

    "'cause": "because",

    "could've": "could have",

    "couldn't": "could not",

    "couldn't've": "could not have",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hadn't've": "had not have",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "He had",

    "he'd've": "He would have",

    "he'll": "He will",

    "he'll've": "He will have",

    "he's": "He is",

    "how'd": "How did",

    "how'd'y": "How do you",

    "how'll": "How will",

    "how's": "How is",

    "i'd": "I had",

    "i'd've": "I would have",

    "i'll": "I will",

    "i'll've": "I will have",

    "i'm": "I am",

    "i've": "I have",

    "isn't": "is not",

    "it'd": "It had",

    "it'd've": "It would have",

    "it'll": "It will",

    "it'll've": "It will have",

    "it's": "It is",

    ".it's": "It is",

    "let's": "Let us",

    "ma'am": "Madam",

    "mayn't": "may not",

    "might've": "might have",

    "mightn't": "might not",

    "mightn't've": "might not have",

    "must've": "must have",

    "mustn't": "must not",

    "mustn't've": "must not have",

    "needn't": "need not",

    "needn't've": "need not have",

    "o'clock": "of the clock",

    "oughtn't": "ought not",

    "oughtn't've": "ought not have",

    "shan't": "shall not",

    "sha'n't": "shall not",

    "shan't've": "shall not have",

    "she'd": "She had",

    "she'd've": "She would have",

    "she'll": "She will",

    "she'll've": "She will have",

    "she's": "She is",

    "should've": "should have",

    "shouldn't": "should not",

    "shouldn't've": "should not have",

    "so've": "so have",

    "so's": "so is",

    "that'd": "that had",

    "that'd've": "that would have",

    "that's": "that is",

    "there'd": "There had",

    "there'd've": "There would have",

    "there's": "There has",

    "they'd": "They had",

    "they'd've": "They would have",

    "they'll": "They will",

    "they'll've": "They will have",

    "they're": "They are",

    "they've": "They have",

    "to've": "to have",

    "wasn't": "was not",

    "we'd": "We had",

    "we'd've": "We would have",

    "we'll": "We will",

    "we'll've": "We will have",

    "we're": "We are",

    "we've": "We have",

    "weren't": "were not",

    "what'll": "What will",

    "what'll've": "What will have",

    "what're": "What are",

    "what's": "What is",

    "what've": "What have",

    "when's": "When is",

    "when've": "When have",

    "where'd": "Where did",

    "where's": "Where is",

    "where've": "Where have",

    "who'll": "Who will",

    "who'll've": "Who will have",

    "who's": "Who is",

    "who've": "Who have",

    "why's": "Why is",

    "why've": "Why have",

    "will've": "ill have",

    "won't": "will not",

    "won't've": "will not have",

    "would've": "would have",

    "wouldn't": "would not",

    "wouldn't've": "would not have",

    "y'all": "You all",

    "y'all'd": "You all would",

    "y'all'd've": "You all would have",

    "y'all're": "You all are",

    "y'all've": "You all have",

    "you'd": "You had",

    "you'd've": "You would have",

    "you'll": "You will",

    "you'll've": "You will have",

    "you're": "You are",

    "you've": "You have"

}
def cleaner(text):

    text = str(text).lower()                                                    # LowerCase

    text = re.sub(r'<*?>',' ',text)                                             # Removing HTML tag

    text = re.sub(r'https?://\S+|www\.\S+',' ',text)                            # Removing hyperlink related entries

    text = ' '.join([shortforms[word] 

                     if word in shortforms.keys() else word

                     for word in text.split()])

    text = str(text).lower()                                                    # LowerCase

    #text = str(text).translate(str.maketrans('','',string.punctuation))         # Removing punctuation

   

    text = re.sub(r'^\s','',text)

    text = re.sub(r'\s+',' ',text)

    

    # Spelling Checker

    

    #spell = SpellChecker()

    #correct = []

    #wrong_words = spell.unknown(text.split())



    #for w in text.split():

    #    if w in wrong_words :

    #        correct.append(spell.correction(w))

    #    else :

    #        correct.append(w)

    

    return(text)
%%time

train['cleaner_text'] = train.text.progress_apply(lambda x: cleaner(x))

test['cleaner_text'] = test.text.progress_apply(lambda x: cleaner(x))
from transformers import RobertaTokenizer, TFAutoModel, AutoConfig, TFRobertaMainLayer



case = 'roberta-base'



tokenizer = RobertaTokenizer.from_pretrained(case)

config = AutoConfig.from_pretrained(case, output_attentions = True, output_hidden_states = True)

model = TFAutoModel.from_pretrained(case, config = config)

bert = TFRobertaMainLayer(config)
%%time

import tqdm



def convert2token(all_text):

    token_id, attention_id = [], []

    for i, sent in tqdm.tqdm(enumerate(all_text)):

        token_dict = tokenizer.encode_plus(sent, max_length=60, pad_to_max_length=True, return_attention_mask=True, 

                                           return_tensors='tf', add_special_tokens= True)

        token_id.append(token_dict['input_ids'])

        attention_id.append(token_dict['attention_mask'])

    

    token_id = np.array(token_id, dtype='int32')

    attention_id = np.array(attention_id, dtype='int32')

    return(token_id, attention_id)



train_token_id, train_attention_id = convert2token(train.cleaner_text.values)

test_token_id, test_attention_id = convert2token(test.cleaner_text.values)
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint



def building_model(need_emb):

    inp_1 = tf.keras.layers.Input(shape = (60,), name = 'token_id', dtype = 'int32')

    inp_2 = tf.keras.layers.Input(shape = (60,), name = 'mask_id', dtype = 'int32')



    x1 = tf.keras.layers.Reshape((60,))(inp_1)

    x2 = tf.keras.layers.Reshape((60,))(inp_2)

    

    if need_emb:

        

        emb = model(x1, attention_mask = x2)[0]     # Give output in form batch_size * max_len_token * hidden_dim, Using only CLS token hidden dimension        

        x = tf.keras.layers.Dense(256, activation = 'relu')(emb[:,0,:])

        x = tf.keras.layers.BatchNormalization() (x)

        x = tf.keras.layers.Dropout(0.3) (x)

        x = tf.keras.layers.Dense(32, activation = 'relu') (x)

        x = tf.keras.layers.BatchNormalization() (x)

        x = tf.keras.layers.Dropout(0.3) (x)



    

    else :

        

        emb = bert(x1, attention_mask = x2)[0]

        x = tf.keras.layers.Dropout(0.2) (emb[:,0,:])





    out = tf.keras.layers.Dense(1, activation = 'sigmoid') (x)



    Emb_Model = tf.keras.models.Model(inputs = [inp_1, inp_2], outputs = out)

    callback = ModelCheckpoint(filepath = 'best.hdf5', monitor = 'val_loss', save_best_only = True, verbose = 1)

    

    # Further more we need to set Bert layer as it is. We don't want to train Bert layer as we are using it as pretrained layer.

    # Setting Bert Layer trainable = False

    

    if need_emb :

        

        for layer in Emb_Model.layers[:5]:

            layer.trainable = False

    

    return(Emb_Model, callback)
Emb_Model, callback = building_model(need_emb=True)

Emb_Model.summary()
Emb_Model.compile(metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate = 4e-5), loss='binary_crossentropy')

Emb_Model.fit([np.reshape(train_token_id, (7503,60)), np.reshape(train_attention_id, (7503,60))],  train.target, epochs=10, 

      batch_size=64, validation_split=0.20, shuffle = True)
%%time

Emb_Model_Answer = Emb_Model.predict([np.reshape(test_token_id, (3263,60)), np.reshape(test_attention_id, (3263,60))])
Tune_Bert,callback = building_model(need_emb = False)

Tune_Bert.summary()
Tune_Bert.compile(metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy')

Tune_Bert.fit([np.reshape(train_token_id, (7503,60)), np.reshape(train_attention_id, (7503,60))],  train.target, epochs=10, 

      batch_size=64, validation_split=0.20, shuffle = True, callbacks = [callback])
Tune_Bert.load_weights('best.hdf5')

Tune_answer = Tune_Bert.predict([np.reshape(test_token_id, (3263,60)), np.reshape(test_attention_id, (3263,60))])
sample.head(2)
answer_Emb = pd.DataFrame({'id': sample.id, 'target': np.where(Emb_Model_Answer>0.5,1,0).reshape(Emb_Model_Answer.shape[0])})

answer_tune = pd.DataFrame({'id': sample.id, 'target': np.where(Tune_answer>0.5,1,0).reshape(Tune_answer.shape[0])})
answer_Emb.to_csv('submission_emb.csv', index = False)

answer_tune.to_csv('submission_tune.csv', index = False)