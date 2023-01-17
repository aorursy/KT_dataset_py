import numpy as np

import pandas as pd

import matplotlib

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

%matplotlib inline

import math



import tensorflow as tf

import tensorflow.keras.backend as K

from transformers import *

import tokenizers

from sklearn.model_selection import StratifiedKFold



pd.set_option('max_colwidth', 40)
IS_PSEUDO_LABELLING = False

IS_USING_PSEUDO_LABELS = False # currently cannot be used with IS_ADDING_SENTIMENT_WORDS

IS_SUBSTITUTING_URLS = False

IS_GETTING_BEST_LOGITS = False

IS_PREDICTING_TRAIN = False

IS_POSTPROCESSING = False

IS_AUGMENT_SYNONYM = False

IS_ADDING_SENTIMENT_WORDS = True # currently cannot be used with IS_USING_PSEUDO_LABELS

IS_INCLUDING_EXTRA_DATA = False

IS_EXPECTATION_MAXIMISE = True
MAX_LEN = 192

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

PAD_ID = 1

SEED = 88888

tf.random.set_seed(SEED)

np.random.seed(SEED)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')

train.head()
if IS_INCLUDING_EXTRA_DATA:

    extra_data = pd.read_csv('../input/complete-tweet-sentiment-extraction-data/tweet_dataset.csv').fillna('')

    extra_data = extra_data[(~extra_data.aux_id.isin(train['textID'].values)) & (extra_data.selected_text != '')].reset_index(drop=True)

    extra_data.textID = extra_data.aux_id

    extra_data.sentiment = extra_data.new_sentiment

    extra_data = extra_data[['textID', 'text', 'selected_text', 'sentiment']]

    train = pd.concat([train, extra_data]).reset_index(drop=True)
if IS_AUGMENT_SYNONYM:

    import json

    import random

    random.seed(SEED)

    with open('../input/englishengen-synonyms-json-thesaurus/eng_synonyms.json') as json_file:  

        synonyms_dict = json.load(json_file)



    def get_synonym_word(text, first_selected, last_selected):

        attempts = 0

        old_word = ''

        new_word = ''

        possible_words = text.split()

        while len(possible_words) > 0 and attempts < 5:

            word_choice = random.choice(possible_words)

            if word_choice in synonyms_dict and len(synonyms_dict[word_choice]) > 0:

                if not ((word_choice in first_selected and len(word_choice) != len(first_selected)) and (word_choice in last_selected and len(word_choice) != len(last_selected))):

                    old_word = word_choice

                    new_word = random.choice(synonyms_dict[old_word])

                    break

            attempts += 1

        return old_word, new_word



    def get_synonym_row(row, changes):

        text = row['text']

        selected_text = row['selected_text']

        sentiment = row['sentiment']

        point_id = row['textID'] + 'aug' + str(changes)



        if len(text) > 0:

            for i in range(changes):

                # Get a synonym

                word_to_replace, replacement_word = get_synonym_word(text, selected_text.split()[0], selected_text.split()[-1])

                # Make the replacement

                if word_to_replace in selected_text.split():

                    old_selected_text = selected_text

                    selected_text = selected_text.replace(word_to_replace, replacement_word)

                    text = text.replace(old_selected_text, selected_text, 1)

                else:

                    text = text.replace(word_to_replace, replacement_word)



        return text, selected_text, sentiment, point_id



    aug_rows = [train]

    for changes in [1]:

        new_rows = []

        for index, row in train.iterrows():

            text, selected_text, sentiment, point_id = get_synonym_row(row, changes)

            new_rows.append([point_id, text, selected_text, sentiment])

        aug_train = pd.DataFrame(new_rows, columns=['textID', 'text', 'selected_text', 'sentiment'])

        aug_rows.append(aug_train)



    train = pd.concat(aug_rows).sort_index(kind='merge').reset_index(drop=True)
if IS_USING_PSEUDO_LABELS:

    pseudo_labelled_set = pd.read_csv('../input/tweetsentimentextraction2020completepseudo/extra_data.csv').fillna('')[['textID', 'text', 'selected_text', 'sentiment']]

    pseudo_labelled_folds = []

    for k in range(5):

        pseudo_labelled_folds.append(pseudo_labelled_set.iloc[k::5].reset_index(drop=True))

    pseudo_labelled_folds[4].head()
if IS_PSEUDO_LABELLING:

    test = pd.read_csv('../input/complete-tweet-sentiment-extraction-data/tweet_dataset.csv').fillna('')

    test = test[~test.aux_id.isin(train['textID'].values)].reset_index(drop=True)

    test['textID'] = test['aux_id']

    sent_con = {'empty' : 'neutral', 'sadness' : 'negative', 'worry' : 'negative', 'neutral' : 'neutral', 'fun' : 'positive', 'happiness' : 'positive', 'hate' : 'negative', 'surprise' : 'neutral', 'relief' : 'positive', 'enthusiasm' : 'neutral', 'anger' : 'negative', 'boredom' : 'negative', 'love' : 'positive'}

    test['sentiment'] = test.apply(lambda x: x.new_sentiment if len(x.new_sentiment) > 0 else sent_con[x.sentiment], axis=1)

    test[['textID', 'text', 'sentiment']]

else:

    test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')



test.head()
DROPOUT = 0.1 # 0.1

N_SPLIT = 5 # 5

LEARNING_RATE = 3e-5 # 3e-5

LEAKY_RELU_ALPHA = 0.1 # 0.3

LABEL_SMOOTHING = 0.1 # 0

EPOCHS = 3 # 3

BATCH_SIZE = 32 # 32
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



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

    

    # The first output (for the start token)

    x1 = tf.keras.layers.Dropout(DROPOUT)(x[0])

    x1 = tf.keras.layers.Conv1D(768, 2, padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x1)

    x1 = tf.keras.layers.Conv1D(128, 2, padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x1)

    x1 = tf.keras.layers.Dense(32)(x1)

    x1 = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    # The second output (for the end token)

    x2 = tf.keras.layers.Dropout(DROPOUT)(x[0]) 

    x2 = tf.keras.layers.Conv1D(768, 2, padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x2)

    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x2)

    x2 = tf.keras.layers.Dense(32)(x2)

    x2 = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)

    

    # Create model

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) 

    model.compile(loss=loss_fn, optimizer=optimizer)

    

    # this is required as `model.predict` needs a fixed size!

    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model
if IS_POSTPROCESSING:

    def post_processing(text):

        if len(text.split()) > 0 and 'http' in text.split()[-1]:

            return  ' '.join(text.split()[:-1])

        else:

            return text
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
if IS_SUBSTITUTING_URLS:

    import re



    def preprocess_urls(replacements, full_text, sub_text, k):

        text_split = full_text.split()

        urls_done = 0

        for word in text_split:

            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', word)

            if len(urls) > 0:

                replacements.append((k, urls[0], 'website' + str(urls_done)))

                full_text = full_text.replace(urls[0], 'website' + str(urls_done), 1)

                sub_text = sub_text.replace(urls[0], 'website' + str(urls_done), 1)

                urls_done += 1

        return replacements, full_text, sub_text



    def deprocess_urls(replacements, full_text, sub_text, k):

        replaces = [x for x in replacements if x[0] == k]

        for replacer in replaces:

            full_text = full_text.replace(replacer[2], replacer[1], 1)

            sub_text = sub_text.replace(replacer[2], replacer[1], 1)

        return full_text, sub_text



    replacements, test, sub_test = preprocess_urls(replacements, test, sub_test, k)



    test, sub_test = deprocess_urls(replacements, test, sub_test, k)
if IS_SUBSTITUTING_URLS:

    train_web_replacements = []

    for k in range(train.shape[0]):

        text = train.loc[k, 'text']

        sub_text = train.loc[k, 'selected_text']

        train_web_replacements, train.loc[k, 'text'], train.loc[k, 'selected_text'] = preprocess_urls(train_web_replacements, text, sub_text, k)

    

    if IS_USING_PSEUDO_LABELS:

        pseudo_web_replacements = []

        for fold in range(5):

            pseudo_replacements = []

            for k in range(train.shape[0]):

                text = pseudo_labelled_folds[fold].loc[k, 'text']

                sub_text = pseudo_labelled_folds[fold].loc[k, 'selected_text']

                pseudo_replacements, pseudo_labelled_folds[fold].loc[k, 'text'], pseudo_labelled_folds[fold].loc[k, 'selected_text'] = preprocess_urls(pseudo_replacements, text, sub_text, k)

            pseudo_web_replacements.append(pseudo_replacements)



    test_web_replacements = []

    for k in range(test.shape[0]):

        text = test.loc[k, 'text']

        sub_text = ''

        test_web_replacements, test.loc[k, 'text'], throwaway = preprocess_urls(train_web_replacements, text, sub_text, k)

        del(throwaway)
if IS_ADDING_SENTIMENT_WORDS:

    from nltk.sentiment.vader import SentimentIntensityAnalyzer



    polarity_threshold = 0.3

    pos_exist_token = 8000

    npos_exist_token = 8001

    neg_exist_token = 9000

    nneg_exist_token = 9001



    sid = SentimentIntensityAnalyzer()



    # Find the highly positive and negative words in the dataset

    def get_high_polarity_words(data):

        high_polarity_words = []

        for index, row in data.iterrows():

            row_polarising = []

            for word in row['text'].split():

                if (sid.polarity_scores(word)['compound'] >= polarity_threshold) or (sid.polarity_scores(word)['compound'] <= -polarity_threshold):

                    row_polarising += tokenizer.encode(word).ids

                    #row_polarising.append(word)

            high_polarity_words.append(row_polarising)

        return np.array(high_polarity_words)



    # Create tokens to specify whether highly positive and negative words exist

    def get_high_polarity_tokens(data):

        high_polarity_tokens = []

        for index, row in data.iterrows():

            pos_words = [word for word in row['text'].split() if sid.polarity_scores(word)['compound'] >= polarity_threshold]

            neg_words = [word for word in row['text'].split() if sid.polarity_scores(word)['compound'] <= -polarity_threshold]

            high_polarity_tokens.append([pos_exist_token if len(pos_words) > 0 else npos_exist_token,

                                         neg_exist_token if len(neg_words) > 0 else nneg_exist_token])

        return np.array(high_polarity_tokens)



    train_sentiment_words = get_high_polarity_tokens(train)

    test_sentiment_words = get_high_polarity_tokens(test)
ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    idx = text1.find(text2)

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

    if IS_ADDING_SENTIMENT_WORDS:

        input_ids[k,:len(enc.ids)+len(train_sentiment_words[k])+3] = [0, s_tok] + list(train_sentiment_words[k]) + enc.ids + [2]

        attention_mask[k,:len(enc.ids)+len(train_sentiment_words[k])+3] = 1

    else:

        input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

        attention_mask[k,:len(enc.ids)+3] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+2] = 1

        end_tokens[k,toks[-1]+2] = 1
if IS_USING_PSEUDO_LABELS:

    pseudo_labelled_folds_tokens = []

    for fold in range(5):

        pseudo_labels = pseudo_labelled_folds[fold]

        ct = pseudo_labels.shape[0]

        input_ids_ps = np.ones((ct,MAX_LEN),dtype='int32')

        attention_mask_ps = np.zeros((ct,MAX_LEN),dtype='int32')

        token_type_ids_ps = np.zeros((ct,MAX_LEN),dtype='int32')

        start_tokens_ps = np.zeros((ct,MAX_LEN),dtype='int32')

        end_tokens_ps = np.zeros((ct,MAX_LEN),dtype='int32')



        for k in range(pseudo_labels.shape[0]):



            # FIND OVERLAP

            text1 = " "+" ".join(pseudo_labels.loc[k,'text'].split())

            text2 = " ".join(pseudo_labels.loc[k,'selected_text'].split())

            idx = text1.find(text2)

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



            s_tok = sentiment_id[pseudo_labels.loc[k,'sentiment']]

            input_ids_ps[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

            attention_mask_ps[k,:len(enc.ids)+3] = 1

            if len(toks)>0:

                start_tokens_ps[k,toks[0]+2] = 1

                end_tokens_ps[k,toks[-1]+2] = 1



            pseudo_labelled_folds_tokens.append((input_ids_ps, attention_mask_ps, token_type_ids_ps, start_tokens_ps, end_tokens_ps))
ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    if IS_ADDING_SENTIMENT_WORDS:

        input_ids_t[k,:len(enc.ids)+len(test_sentiment_words[k])+3] = [0, s_tok] + list(test_sentiment_words[k]) + enc.ids + [2]

        attention_mask_t[k,:len(enc.ids)+len(test_sentiment_words[k])+3] = 1

    else:

        input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

        attention_mask_t[k,:len(enc.ids)+3] = 1
if IS_EXPECTATION_MAXIMISE:

    jem_preds = np.zeros((input_ids_t.shape[0], N_SPLIT * 3))

    jem_preds = pd.DataFrame(jem_preds, columns = ['start0', 'end0', 'string0', 'start1', 'end1', 'string1', 'start2', 'end2', 'string2', 'start3', 'end3', 'string3', 'start4', 'end4', 'string4'])
%%time

jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE



# The start and end tokens will be stored in these

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))

#preds_start_train = np.zeros((input_ids.shape[0],MAX_LEN))

#preds_end_train = np.zeros((input_ids.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=N_SPLIT,shuffle=True,random_state=SEED)

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):

    # Output the current fold

    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    # Build the model

    K.clear_session()

    model, padded_model = build_model()

    

    # Add pseudo labels to the fold

    if IS_USING_PSEUDO_LABELS:

        input_ids_ps, attention_mask_ps, token_type_ids_ps, start_tokens_ps, end_tokens_ps = pseudo_labelled_folds_tokens[fold]

    input_ids_fold = np.concatenate((input_ids[idxT,], input_ids_ps)) if IS_USING_PSEUDO_LABELS else input_ids[idxT,]

    attention_mask_fold = np.concatenate((attention_mask[idxT,], attention_mask_ps)) if IS_USING_PSEUDO_LABELS else attention_mask[idxT,]

    token_type_ids_fold = np.concatenate((token_type_ids[idxT,], token_type_ids_ps)) if IS_USING_PSEUDO_LABELS else token_type_ids[idxT,]

    start_tokens_fold = np.concatenate((start_tokens[idxT,], start_tokens_ps)) if IS_USING_PSEUDO_LABELS else start_tokens[idxT,]

    end_tokens_fold = np.concatenate((end_tokens[idxT,], end_tokens_ps)) if IS_USING_PSEUDO_LABELS else end_tokens[idxT,]

        

    #sv = tf.keras.callbacks.ModelCheckpoint(

    #    '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,

    #    save_weights_only=True, mode='auto', save_freq='epoch')

    inpT = [input_ids_fold, attention_mask_fold, token_type_ids_fold]

    targetT = [start_tokens_fold, end_tokens_fold]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]

    

    # Sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '%s-roberta-%i.h5'%(VER,fold)

    for epoch in range(1, EPOCHS + 1):

        # Sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))

        

        # Shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        

        # Reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY, callbacks=[],

            validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

        save_weights(model, weight_fn)

    

    # Load weights

    print('Loading model...')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    load_weights(model, weight_fn)

    

    # Make fold predictions

    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

    

    # Predict on the test set (which will only be 1/fold of the prediction)

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    if IS_PSEUDO_LABELLING:

        preds[0][fold::5] = 0

        preds[1][fold::5] = 0

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # JEM prediction storage

    if IS_EXPECTATION_MAXIMISE:

        jem_preds['start' + str(fold)] = [preds[0][index][token] for index, token in enumerate(np.argmax(preds[0], axis=1))]

        jem_preds['end' + str(fold)] = [preds[1][index][token] for index, token in enumerate(np.argmax(preds[1], axis=1))]

        jem_preds['string' + str(fold)] = [tokenizer.decode(tokenizer.encode(" "+" ".join(test.loc[i,'text'].split())).ids[np.argmax(preds[0][i])-2:np.argmax(preds[1][i])-1]) for i in range(input_ids_t.shape[0])]

    

    if IS_PREDICTING_TRAIN:

        print('Predicting Train...')

        preds_train = padded_model.predict([input_ids,attention_mask,token_type_ids],verbose=DISPLAY)

        preds_start_train += preds_train[0]/skf.n_splits

        preds_end_train += preds_train[1]/skf.n_splits

    

    # Convert the predicted start and end tokens into strings

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

        if IS_SUBSTITUTING_URLS:

            throwaway, st = deprocess_urls(train_web_replacements, '', st, k)

        if IS_POSTPROCESSING:

            st = post_processing(st)

        all.append(jaccard(st,train.loc[k,'selected_text']))

    

    # Output fold score

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))

for j in jac:

    print('>>', j)
jem_all = []



if IS_EXPECTATION_MAXIMISE:

    def jem_expectation_maximisation(confidences, strings, check_string):

        res = 0

        for i in range(len(confidences)):

            res += confidences[i] * jaccard(strings[i], check_string)

        return res



    for k in range(input_ids_t.shape[0]):

        # Calculate the prediction for each fold

        #for fold in range(N_SPLIT):

        #    jem_preds.loc[k,'string' + str(fold)] = tokenizer.decode(enc.ids[jem_preds.loc[k,'start' + str(fold)]-2:jem_preds.loc[k,'end' + str(fold)]-1])

        

        # Get predictions and confidences for each fold

        jem_strings = [jem_preds.loc[k,'string' + str(x)] for x in range(N_SPLIT)]

        jem_confidences = [0.5 * (jem_preds.loc[k, 'start' + str(x)] + jem_preds.loc[k, 'end' + str(x)]) for x in range(N_SPLIT)]

        

        # Jaccard Expectation Maximisation

        jem_best = (0, test.loc[k, 'text'])

        for fold in range(N_SPLIT):

            jem_fold_conf = jem_expectation_maximisation(jem_confidences, jem_strings, jem_preds.loc[k, 'string' + str(fold)])

            if jem_fold_conf > jem_best[0]:

                jem_best = (jem_fold_conf, jem_strings[fold])

        

        # Add the best prediction to the list

        jem_all.append(jem_best[1])
if IS_GETTING_BEST_LOGITS:

    def get_best_start_end_idxs(_start_logits, _end_logits):

        best_logit = -1000

        best_idxs = None

        for start_idx, start_logit in enumerate(_start_logits):

            for end_idx, end_logit in enumerate(_end_logits[start_idx:]):

                logit_sum = (start_logit + end_logit).item()

                if logit_sum > best_logit:

                    best_logit = logit_sum

                    best_idxs = (start_idx, start_idx+end_idx)

        return best_idxs
all = []

for k in range(input_ids_t.shape[0]):

    # Get the best start and end tokens

    if IS_GETTING_BEST_LOGITS:

        a, b = get_best_start_end_idxs(preds_start[k,], preds_end[k,])

    else:

        a = np.argmax(preds_start[k,])

        b = np.argmax(preds_end[k,])

    

    # Extract the selected text using the start and end tokens

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-2:b-1])

    

    # Substitute URL if required

    if IS_SUBSTITUTING_URLS:

        throwaway, st = deprocess_urls(test_web_replacements, '', st, k)

    

    # Perform postprocessing

    if IS_POSTPROCESSING:

        st = post_processing(st)

    

    all.append(st)
if IS_EXPECTATION_MAXIMISE:

    test['selected_text'] = jem_all

else:

    test['selected_text'] = all



if IS_PSEUDO_LABELLING:

    test.to_csv('extra_data.csv', index=False)

else:

    test[['textID','selected_text']].to_csv('submission.csv',index=False)

    

test.sample(10)
if IS_PREDICTING_TRAIN:

    all = []

    start = []

    end = []

    start_pred = []

    end_pred = []

    for k in range(input_ids.shape[0]):

        # Get the best start and end tokens

        if IS_GETTING_BEST_LOGITS:

            a, b = get_best_start_end_idxs(preds_start_train[k,], preds_end_train[k,])

        else:

            a = np.argmax(preds_start_train[k,])

            b = np.argmax(preds_end_train[k,])

        

        start.append(np.argmax(start_tokens[k]))

        end.append(np.argmax(end_tokens[k]))

        

        # Extract the selected text using the start and end tokens

        if a>b:

            st = train.loc[k,'text']

            start_pred.append(0)

            end_pred.append(len(st))

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[a-2:b-1])

            start_pred.append(a)

            end_pred.append(b)

            

        # Substitute URL if required

        if IS_SUBSTITUTING_URLS:

            throwaway, st = deprocess_urls(test_web_replacements, '', st, k)



        # Perform postprocessing

        if IS_POSTPROCESSING:

            st = post_processing(st)

            

        all.append(st)

        

    train['start'] = start

    train['end'] = end

    train['start_pred'] = start_pred

    train['end_pred'] = end_pred

    train['selected_text_pred'] = all

    train.to_csv('outliers.csv', index=False)

    train.sample(10)