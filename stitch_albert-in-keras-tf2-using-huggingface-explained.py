import sys

sys.path.insert(0, "../input/transformers/transformers-master/")

!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



from transformers import *



import pandas as pd

import numpy as np

import re

from tqdm.notebook import tqdm

from math import floor, ceil



import tensorflow as tf

print(tf.__version__)
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
albert_path = '../input/albertlargev2huggingface/'

tokenizer = AlbertTokenizer.from_pretrained(albert_path, do_lower_case=True)

albert_model = TFAlbertModel.from_pretrained(albert_path)



#bert_path = '../input/bert-base-uncased-huggingface/'

#tokenizer = BertTokenizer.from_pretrained(bert_path+'vocab.txt', do_lower_case=True)

#bert_model = TFBertModel.from_pretrained(bert_path)
print("Data cleaning started........")

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"couldnt" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"doesnt" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"havent" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"shouldnt" : "should not",

"that's" : "that is",

"thats" : "that is",

"there's" : "there is",

"theres" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"theyre":  "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"}





def clean_text(text):

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    #text = text.lower().split()

    #stops = set(stopwords.words("english"))

    #text = [w for w in text if not w in stops]    

    #text = " ".join(text)

    return(text)



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



def replace_typical_misspell(text):

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)



def clean_data(df, columns: list):

    for col in columns:

        df[col] = df[col].apply(lambda x: clean_text(x.lower()))

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))



    return df
columns = ['question_title','question_body','answer']

train= clean_data(train, columns)

test = clean_data(test, columns)

print("Data cleaning Done........")
def _get_segments(tokens,_maxlen):

    sentences_segments = []

    i=0

    for token in tokens:

        sentences_segments.append(i)

        if token == "[SEP]":

            i += 1

    return sentences_segments + [0] * (_maxlen - len(tokens))



def _get_inputs(df,_maxlen,tokenizer,use_keras_pad=False):

    

    #generate input ids  

    maxqnans = np.int((_maxlen-34)/2) #approx token size : qn = 230, ans = 230, title =30, special_words = 4

    pattern = '[^\w\s]+|\n' # remove everything including newline (|\n) other than words (\w) or spaces (\s)

    

    sentences = [["[CLS] "] + tokenizer.tokenize(title)[:20]+ ["[SEP]"]

                  + tokenizer.tokenize(qn)[:maxqnans] + ["[SEP]"]

                  + tokenizer.tokenize(ans)[:maxqnans] + ["[SEP]"]

                  for (title,qn,ans) 

                in 

              zip(df['question_title'].str.replace(pattern, '').values.tolist(),

              df['question_body'].str.replace(pattern, '').values.tolist(),

              df['answer'].str.replace(pattern, '').values.tolist())]

    

    

    # if less than max length provided then the words are padded

    if use_keras_pad:

        sentences_padded = pad_sequences(sentences, dtype=object, maxlen=_maxlen, value=['[PAD]'],padding='post')

    else:

        sentences_padded = [tokens + ["[PAD]"]*(_maxlen-len(tokens)) if len(tokens)!=_maxlen else tokens for tokens in sentences ]



    sentences_converted = [tokenizer.convert_tokens_to_ids(s) for s in sentences_padded]

    

    

    #generate masks

    # bert requires a mask for the words which are padded. 

    # Say for example, maxlen is 100, sentence size is 90. then, [1]*90 + [0]*[100-90]

    sentences_mask = [[1]*len(tokens)+[0]*(_maxlen - len(tokens)) for tokens in sentences]

 

    

    #generate segments

    # for each separation [SEP], a new segment is converted

    sentences_segments = [_get_segments(tokens,_maxlen=_maxlen) for tokens in sentences]



    genLength = set([len(tokens) for tokens in sentences_padded])

    if _maxlen < 20:

        raise Exception("max length cannot be less than 20")

    elif len(genLength)!=1: 

        print(genLength)

        raise Exception("sentences are not of same size \n {}".format(genLength))



    #convert list into tensor integer arrays and return it

    #return sentences_converted,sentences_segment, sentences_mask

    #return sentences

    return [np.asarray(sentences_converted, dtype=np.int32), 

            np.asarray(sentences_mask, dtype=np.int32), 

            np.asarray(sentences_segments, dtype=np.int32)]

    #return [tf.cast(sentences_converted,tf.int32), tf.cast(sentences_segment,tf.int32), tf.cast(sentences_mask,tf.int32)]
MAX_SEQUENCE_LENGTH = 512

Xtr = _get_inputs(train,_maxlen=MAX_SEQUENCE_LENGTH,tokenizer=tokenizer,use_keras_pad=False)

ytr = np.asarray(train.iloc[:,11:])
Xte = _get_inputs(test,_maxlen=MAX_SEQUENCE_LENGTH,tokenizer=tokenizer,use_keras_pad=False)
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout, GlobalAveragePooling1D

from tensorflow.keras import Sequential

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import regularizers

import tensorflow.keras.backend as K



from sklearn.model_selection import KFold, GroupKFold

from scipy.stats import spearmanr



import warnings; warnings.simplefilter('ignore')
def build_model():

    

    token_inputs = Input((MAX_SEQUENCE_LENGTH), dtype=tf.int32, name='input_word_ids')

    #mask_inputs = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')

    #seg_inputs = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')



    # going with pooled output since seq_output results in ResourceExhausted Error even with GPU

    _,pooled_output = albert_model(token_inputs)#[token_inputs, mask_inputs, seg_inputs])

    #X = GlobalAveragePooling1D()(pooled_output)

    X = Dropout(0.2)(pooled_output)

    output_= Dense(30, activation='sigmoid', name='output')(X)



    #bert_model2 = Model([token_inputs, mask_inputs, seg_inputs],output_)

    bert_model2 = Model(token_inputs,output_)

    

    print(bert_model2.summary())

    

    bert_model2.compile(optimizer= Adam(learning_rate=0.0001), loss='binary_crossentropy')

    

    return bert_model2
i=0

num_folds = 3

#kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

gkf = GroupKFold(n_splits=3).split(X=train.question_body, groups=train.question_body)

fold_score = []

#test_preds = np.zeros((Xte[0].shape[0],ytr.shape[1])) # mimic rows shape of test data, columns shape from train since, test will have any column for outputs

test_preds = []



#for train_index,val_index in kfold.split(ytr):

for fold, (train_index, val_index) in enumerate(gkf):



    i= i+1

    print('executing fold no: {}'.format(i))

    

    K.clear_session()

    # train_index gets a random sample of rows for training

    # Xtr is a list contains 3 np arrays - ids, masks, segments so, using list comprehension to get the splits

    Xtr_fold = [arr[train_index] for arr in Xtr]

    ytr_fold = ytr[train_index]

    

    Xtr_val = [arr[val_index] for arr in Xtr]

    ytr_val = ytr[val_index]

    

    model = build_model()

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    #model.fit(Xtr_fold,ytr_fold,epochs=5,batch_size = 8,validation_data = (Xtr_val,ytr_val),callbacks=[es])

    model.fit(Xtr_fold[0],ytr_fold,epochs=5,batch_size = 4 ,validation_split = 0.2,callbacks=[es])

    

    #calculate spearman score

    pred_ = pd.DataFrame(model.predict(Xtr_val[0]))

    val_ = pd.DataFrame(ytr_val)

    spearman_score = np.nanmean([spearmanr(pred_.iloc[:,i].values,val_.iloc[:,i].values).correlation for i in np.arange(len(pred_.columns))])

    print("Spearman Score on validation data : {}".format(spearman_score))

    fold_score.append(spearman_score)

    test_preds.append(model.predict(Xte))



print("Spearman Score on validation data : {}".format(np.mean(fold_score)))

#average of across arrays by row

sub.iloc[:, 1:] = np.average(test_preds,axis=0)

sub.to_csv('submission.csv', index=False)