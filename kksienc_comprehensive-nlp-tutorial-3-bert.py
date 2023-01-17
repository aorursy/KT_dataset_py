from IPython.display import Image
Image("/kaggle/input/images/bert.png",  width=350)
# install tesorflow bert package
!pip install bert-for-tf2

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert

#Loding pretrained bert layer
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)


# Loading tokenizer from the bert layer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocab_file, do_lower_case)

text = 'Encoding will be clear with tihs example'
# tokenize
tokens_list = tokenizer.tokenize(text)
print('Text after tokenization')
print(tokens_list)

# initilize dimension
max_len =25
text = tokens_list[:max_len-2]
input_sequence = ["[CLS]"] + text + ["[SEP]"]
print("After adding  flasges -[CLS] and [SEP]: ")
print(input_sequence)


tokens = tokenizer.convert_tokens_to_ids(input_sequence )
print("tokens to id ")
print(tokens)

pad_len = max_len -len(input_sequence)
tokens += [0] * pad_len
print("tokens: ")
print(tokens)

print(pad_len)
pad_masks = [1] * len(input_sequence) + [0] * pad_len
print("Pad Masking: ")
print(pad_masks)

segment_ids = [0] * max_len
print("Segment Ids: ")
print(segment_ids)
# fetch & cleaning  datsset
!pip install pyspellchecker
import pandas as pd
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer 
import nltk 
import re

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

def convert_to_antonym(sentence):
    words = nltk.word_tokenize(sentence)
    new_words = []
    temp_word = ''
    for word in words:
        antonyms = []
        if word == 'not':
            temp_word = 'not_'
        elif temp_word == 'not_':
            for syn in wordnet.synsets(word):
                for s in syn.lemmas():
                    for a in s.antonyms():
                        antonyms.append(a.name())
            if len(antonyms) >= 1:
                word = antonyms[0]
            else:
                word = temp_word + word # when antonym is not found, it will
                                    # remain not_happy
            
            temp_word = ''
        if word != 'not':
            new_words.append(word)
    return ' '.join(new_words)


def correct_spellings(text):
    spell = SpellChecker()
    corrected_words = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)
        

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
  """
    text = text.lower() # lowercase text
    text= re.sub(r'[^\w\s#]',' ',text) #Removing every thing other than space, word and hash
    text  = re.sub(r"https?://\S+|www\.\S+", "", text )
    text= re.sub(r'[0-9]',' ',text)
    #text = correct_spellings(text)
    text = convert_to_antonym(text)
    text = re.sub(' +', ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text    
    return text


train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)


# function to encode the text into tokens, masks, and segment flags
import numpy as np
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

MAX_LEN = 64

# encode train set 
train_input = bert_encode(train_df.text.values, tokenizer, max_len=MAX_LEN)
# encode  test set 
test_input = bert_encode(test_df.text.values, tokenizer, max_len= MAX_LEN )
train_labels = train_df.target.values
# lets see encoded train set 
train_input
# first define input for token, mask and segment id  
from tensorflow.keras.layers import  Input
input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="segment_ids")

#  output  
from tensorflow.keras.layers import Dense
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])  
clf_output = sequence_output[:, 0, :]
out = Dense(1, activation='sigmoid')(clf_output)   

# intilize model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# train
train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=2,
    batch_size=32
)

model.save('model.h5')

test_pred = model.predict(test_input)
preds = test_pred.round().astype(int)
preds