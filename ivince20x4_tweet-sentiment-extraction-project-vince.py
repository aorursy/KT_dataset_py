import numpy as np    # linear algebra

import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import math



# plot visualization

import matplotlib.pyplot as plt

import seaborn as sns 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# nlp libraries

import re

import string

import nltk

import spacy



import tensorflow as tf

from transformers import *

import tokenizers

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

print('TF version',tf.__version__)
datapath= "/kaggle/input/tweet-sentiment-extraction"

train   = pd.read_csv(datapath+"/train.csv").fillna('')

test    = pd.read_csv(datapath+"/test.csv").fillna('')
print(train.head())

print(train.shape)

print("Unique sentiment", train.sentiment.unique())
print(test.head())

print(test.shape)

print("Unique sentiment", test.sentiment.unique())
print(train[train.isna().any(axis=1)])

print(test[test.isna().any(axis=1)])
train.dropna(inplace=True)

print(train.shape)

print(test.shape)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,8))

fig.suptitle('Distribution of sentiment for train and test set')



ax1= sns.countplot(x='sentiment', data=train, ax=ax1)

ax2= sns.countplot(x='sentiment', data=test,  ax=ax2)



for p in ax1.patches:

    ax1.annotate(format(p.get_height(), '.2f'), 

                 (p.get_x() + p.get_width() / 2., p.get_height()), 

                 ha = 'center', va = 'center', xytext = (0, 10), 

                 textcoords = 'offset points')

for p in ax2.patches:

    ax2.annotate(format(p.get_height(), '.2f'), 

                 (p.get_x() + p.get_width() / 2., p.get_height()), 

                 ha = 'center', va = 'center', xytext = (0, 10), 

                 textcoords = 'offset points')



fig.show()
eda= train.copy()

eda['words_text']       = train.text.apply(lambda x:len(str(x).split()))

eda['words_selectedtxt']= train.selected_text.apply(lambda x:len(str(x).split()))



fig, ax = plt.subplots(3,2, figsize=(13,15))

fig.suptitle('Distribution of words by sentiment in training set')



ax1= sns.distplot(eda[eda['sentiment']=='positive']['words_text'],       ax=ax[0][0])

ax1.set_title('Positive: Text')

ax2= sns.distplot(eda[eda['sentiment']=='positive']['words_selectedtxt'],ax=ax[0][1])

ax2.set_title('Positive: Selected_Text')

ax3= sns.distplot(eda[eda['sentiment']=='negative']['words_text'],       ax=ax[1][0], color='r')

ax3.set_title('Negative: Text')

ax4= sns.distplot(eda[eda['sentiment']=='negative']['words_selectedtxt'],ax=ax[1][1], color='r')

ax4.set_title('Negative: Selected_Text')

ax5= sns.distplot(eda[eda['sentiment']=='neutral']['words_text'],       ax=ax[2][0], color='g')

ax5.set_title('Neutral: Text')

ax6= sns.distplot(eda[eda['sentiment']=='neutral']['words_selectedtxt'],ax=ax[2][1], color='g')

ax6.set_title('Neutral: Selected_Text')



fig.show()
# function to pre-process text



def pre_process(text):

    '''Lowercase text, 

    remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
eda['text2']        = eda['text'].apply(lambda x: pre_process(x))

eda['selectedtxt2'] = eda['selected_text'].apply(lambda x: pre_process(x))

eda.head(10)
test['text2'] = test['text'].apply(lambda x: pre_process(x))

test.head(10)
# function to plot wordcloud



def word_cloud(text, mask=None, max_words=200, max_font_size=100, 

                   figure_size=(24.0,16.0), color = 'white',

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {"u", "im", "i'll","we're", "i'm", "wat", "about", "oh",'got','one'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=400, 

                    height=200,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  
word_cloud(eda[eda['sentiment']=='neutral'].text2, color='white',

               max_font_size=60,title_size=30,

               title="Wordcloud_NeutralTweets")
word_cloud(eda[eda['sentiment']=='positive'].text2, color='white',

               max_font_size=60,title_size=30,

               title="Wordcloud_PositiveTweets")
word_cloud(eda[eda['sentiment']=='negative'].text2, color='white',

               max_font_size=60,title_size=30,

               title="Wordcloud_NegativeTweets")
model_path = '/kaggle/input/tf-roberta/'

MAX_LEN    = 96



tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=model_path+'vocab-roberta-base.json', 

    merges_file=model_path+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
trainrows     = train.shape[0]

input_ids     = np.ones((trainrows, MAX_LEN),dtype='int32')

attention_mask= np.zeros((trainrows,MAX_LEN),dtype='int32')

token_type_ids= np.zeros((trainrows,MAX_LEN),dtype='int32')

start_tokens  = np.zeros((trainrows,MAX_LEN),dtype='int32')

end_tokens    = np.zeros((trainrows,MAX_LEN),dtype='int32')



for k in range(trainrows):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    idx   = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    # ID_OFFSETS

    offsets = []; idx=0

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask[k,:len(enc.ids)+3] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+2] = 1

        end_tokens[k,toks[-1]+2] = 1
testrows        = test.shape[0]

input_ids_t     = np.ones((testrows, MAX_LEN), dtype='int32')

attention_mask_t= np.zeros((testrows, MAX_LEN), dtype='int32')

token_type_ids_t= np.zeros((testrows, MAX_LEN), dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask_t[k,:len(enc.ids)+3] = 1
import pickle



def save_weights(model, dst_fn):

    weights = model.get_weights()

    with open(dst_fn, 'wb') as f:

        pickle.dump(weights, f)





def load_weights(model, weight_fn):

    with open(weight_fn, 'rb') as f:

        weights = pickle.load(f)

    model.set_weights(weights)

    return model



def loss_fn(y_true, y_pred):

    # adjust the targets for sequence bucketing

    ll = tf.shape(y_pred)[1]

    y_true = y_true[:, :ll]

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss = tf.reduce_mean(loss)

    return loss



def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)



    lens = MAX_LEN - tf.reduce_sum(padding, -1)

    max_len = tf.reduce_max(lens)

    ids_ = ids[:, :max_len]

    att_ = att[:, :max_len]

    tok_ = tok[:, :max_len]



    config = RobertaConfig.from_pretrained(model_path+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(model_path+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

    

    x1 = tf.keras.layers.Dropout(0.1)(x[0])

    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 

    model.compile(loss=loss_fn, optimizer=optimizer)



    # this is required as `model.predict` needs a fixed size!

    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
SEED   = 42

EPOCHS = 3

BATCH_SIZE = 32

PAD_ID = 1

LABEL_SMOOTHING = 0.1

np.random.seed(SEED)

jac = []

VER='v0'

DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start   = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end     = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end   = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    model, padded_model = build_model()



    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [start_tokens[idxT,], end_tokens[idxT,]]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]



    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '%s-roberta-%i.h5'%(VER,fold)

    

    for epoch in range(1, EPOCHS + 1):

        # sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), 

                                   key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), 

                                   reverse=True))

        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        # reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY, callbacks=[],

            validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

        save_weights(model, weight_fn)



    print('Loading model...')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    load_weights(model, weight_fn)



    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],

                                                            token_type_ids[idxV,]],verbose=DISPLAY)

    

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b: 

            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[a-2:b-1])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
selectedtext = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-2:b-1])

    selectedtext.append(st)
test['selected_text'] = selectedtext

test[['textID','selected_text']].to_csv('submission.csv',index=False)

#pd.set_option('max_colwidth', 60)

test.sample(10)