!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

!pip install pyspellchecker

!python -m spacy download en_vectors_web_lg

!python -m spacy download en_core_web_sm
import gc

import re

import string

import operator

from collections import defaultdict



import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import matplotlib.pyplot as plt

import seaborn as sns



import tokenization

from wordcloud import STOPWORDS



from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import precision_score, recall_score, f1_score



import tensorflow as tf

import tensorflow_hub as hub

from tensorflow import keras

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback



SEED = 1337
from spellchecker import SpellChecker

import pandas as pd

import os

import numpy

from tensorflow.keras.models import Model, Sequential

# from tensorflow.keras.layers import CuDNNLSTM as LSTM

from tensorflow.keras.layers import LSTM as LSTM

from tensorflow.keras.layers import SpatialDropout1D, Input, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Bidirectional, Embedding, TimeDistributed, add, concatenate

from tensorflow.keras.optimizers import Adam

import spacy

import en_vectors_web_lg

import en_core_web_sm

import re

import random



spell = SpellChecker()

print("Loading spaCy")

nlp_ents = en_core_web_sm.load()

nlp_vects = en_vectors_web_lg.load()

nlp_vects.add_pipe(nlp_vects.create_pipe("sentencizer"))
df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})

df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})





for df in [df_train, df_test]:

    for col in ['keyword', 'location']:

        df[col] = df[col].fillna(f'no_{col}')
import re

import os

#import spacy

#nlp_vects = spacy.load('en_vectors_web_lg')

from spellchecker import SpellChecker



def manage_contractions(tweet):

    tweet = tweet.text

    tweet_ = tweet



    # Contractions

    tweet = re.sub(r"\bain'?t\b", "am not", tweet, flags=re.I)

    tweet = re.sub(r"\baren'?t\b", "are not", tweet, flags=re.I)

    tweet = re.sub(r"\bcan'?t\b", "cannot", tweet, flags=re.I)

    tweet = re.sub(r"\bcouldn'?t\b", "could not", tweet, flags=re.I)

    tweet = re.sub(r"\bcould'?ve\b", "could have", tweet, flags=re.I)

    tweet = re.sub(r"\bdidn'?t\b", "did not", tweet, flags=re.I)

    tweet = re.sub(r"\bdoesn'?t\b", "does not", tweet, flags=re.I)

    tweet = re.sub(r"\bdon'?t\b", "do not", tweet, flags=re.I)

    tweet = re.sub(r"\bhasn'?t\b", "has not", tweet, flags=re.I)

    tweet = re.sub(r"\bhaven'?t\b", "have not", tweet, flags=re.I)

    tweet = re.sub(r"\bhe'?ll\b", "he will", tweet, flags=re.I)

    tweet = re.sub(r"\bhere'?s\b", "here is", tweet, flags=re.I)

    tweet = re.sub(r"\bhe'?s\b", "he is", tweet, flags=re.I)

    tweet = re.sub(r"\bi'?d\b", "i would", tweet, flags=re.I)

    tweet = re.sub(r"\bi'?ll\b", "i will", tweet, flags=re.I)

    tweet = re.sub(r"\bi'?m\b", "i am", tweet, flags=re.I)

    tweet = re.sub(r"\bisn'?t\b", "is not", tweet, flags=re.I)

    tweet = re.sub(r"\bit'?ll\b", "it will", tweet, flags=re.I)

    tweet = re.sub(r"\bit'?s\b", "it is", tweet, flags=re.I)

    tweet = re.sub(r"\bit'?s\b", "it is", tweet, flags=re.I)

    tweet = re.sub(r"\bi'?ve\b", "i have", tweet, flags=re.I)

    tweet = re.sub(r"\blet'?s\b", "let us", tweet, flags=re.I)

    tweet = re.sub(r"\bshouldn'?t\b", "should not", tweet, flags=re.I)

    tweet = re.sub(r"\bshould'?ve\b", "should have", tweet, flags=re.I)

    tweet = re.sub(r"\bthat'?s\b", "that is", tweet, flags=re.I)

    tweet = re.sub(r"\bthere'?s\b", "there is", tweet, flags=re.I)

    tweet = re.sub(r"\bthey'?d\b", "they would", tweet, flags=re.I)

    tweet = re.sub(r"\bthey'?ll\b", "they will", tweet, flags=re.I)

    tweet = re.sub(r"\bthey'?re\b", "they are", tweet, flags=re.I)

    tweet = re.sub(r"\bthey'?ve\b", "they have", tweet, flags=re.I)

    tweet = re.sub(r"\bwasn'?t\b", "was not", tweet, flags=re.I)

    tweet = re.sub(r"\bwe'?d\b", "we would", tweet, flags=re.I)

    tweet = re.sub(r"\bwe'?ll\b", "we will", tweet, flags=re.I)

    tweet = re.sub(r"\bwe'?re\b", "we are", tweet, flags=re.I)

    tweet = re.sub(r"\bweren'?t\b", "were not", tweet, flags=re.I)

    tweet = re.sub(r"\bwe'?ve\b", "we have", tweet, flags=re.I)

    tweet = re.sub(r"\bwhat'?s\b", "what is", tweet, flags=re.I)

    tweet = re.sub(r"\bwhere'?s\b", "where is", tweet, flags=re.I)

    tweet = re.sub(r"\bwho'?s\b", "who is", tweet, flags=re.I)

    tweet = re.sub(r"\bwon'?t\b", "will not", tweet, flags=re.I)

    tweet = re.sub(r"\bwouldn'?t\b", "would not", tweet, flags=re.I)

    tweet = re.sub(r"\bwould'?ve\b", "would have", tweet, flags=re.I)

    tweet = re.sub(r"\by'?all\b", "you all", tweet, flags=re.I)

    tweet = re.sub(r"\byou'?d\b", "you would", tweet, flags=re.I)

    tweet = re.sub(r"\byou'?ll\b", "you will", tweet, flags=re.I)

    tweet = re.sub(r"\byou'?re\b", "you are", tweet, flags=re.I)

    tweet = re.sub(r"\byou'?ve\b", "you have", tweet, flags=re.I)



    if tweet.find('woulded') != -1:

        print('oe')



    return tweet



def manage_url_tweet_characters(tweet):

    tweet = tweet.text

    tweet_ = tweet



    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)

    tweet = re.sub('#', '', tweet)

    tweet = re.sub('@.*', 'author', tweet)

    tweet = re.sub('\n', ' ', tweet)



    tweet = re.sub('_', ' ', tweet)



    # Character entity references

    tweet = re.sub(r"&gt;", " ", tweet)

    tweet = re.sub(r"&lt;", " ", tweet)

    tweet = re.sub(r"&amp;", " ", tweet)



    # Extended Characters

    tweet = tweet.encode('iso-8859-1').decode('ascii', 'ignore')



    return tweet





def manage_tricks(tweet):

    tweet = tweet.text

    tweet_ = tweet



    # Three letters

    tweet = re.sub(r'([a-z])\1\1+', r'\1\1', tweet)

    tweet = re.sub(r'([A-Z])\1\1+', r'\1\1', tweet)



    # LetterNumber -> Letter -> Number

    tweet = re.sub("([a-zA-Z])(\d)", r"\1 \2", tweet)

    tweet = re.sub("(\d)([a-zA-Z])", r"\1 \2", tweet)



    # HowAreYou -> How Are You

    tweet = re.sub('([a-z])([A-Z])', r'\1 \2', tweet)



    # numbers & percentage

    re.sub(r'\s[\+~\-]?\d+[\.,\']?\d*\%', ' percentage ', tweet)

    tweet = re.sub(r'\s[\+~\-]?\d+[\.,\']?\d*\s', ' number ', tweet)



    # Dates and times

    tweet = re.sub('\d\d:\d\d:\d\d', 'time', tweet)

    tweet = re.sub('\d\d/\d\d/\d\d', 'date', tweet)

    tweet = re.sub('\d\d/\d\d/\d\d\d\d', 'date', tweet)



    # letter weird letter

    tweet = re.sub('(\d),(\d)', r'\1\2', tweet)

    tweet = re.sub(r'(\d)\.(\d)', r'\1\2', tweet)

    tweet = re.sub('([a-zA-Z0-9])[^a-zA-Z0-9]([a-zA-Z0-9])', r'\1 \2', tweet)



    # dot letter

    tweet = re.sub(r'\b\.([a-zA-Z])', r'\1', tweet)



    # remove more than two spaces

    tweet = re.sub(r' +', r' ', tweet)



    if tweet.find('Rescuers recover') != -1:

        print('eo')



    if tweet.find(r'\.author') != -1:

        print('eo')



    return tweet





def manage_ner(tweet, nlp):

    tweet = tweet.text

    # PERSON    People, including    fictional.

    # NORP    Nationalities or religious or political    groups.

    # FAC    Buildings, airports, highways, bridges, etc.

    # ORG    Companies, agencies, institutions, etc.

    # GPE    Countries, cities, states.

    # LOC     	Non-GPE locations, mountain ranges, bodies of water.

    # MONEY    Monetary    values, including    unit.

    labels = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'MONEY']

    tweet_ = tweet

    tweet = re.sub('[\|\(\)]', '', tweet)

    tweet = re.sub("'", '', tweet)

    doc = nlp(tweet)





    for ent in doc.ents:

        if ent.label_ in labels:

            try:

                if os.name == 'nt':

                    tweet = re.sub('('+ent.text+')', ent.label_, tweet)

                else:

                    tweet = re.sub('('+ent.text+')', ent.label_+' \\1', tweet)

            except:

                pass



    if tweet.find('ORGDemolition') != -1:

        print('person')



    if tweet.find('PERSONDetonate') != -1:

        print('person')





    return tweet





def manage_no_dictionable(tweet, nlp, spell):

    id = tweet.id



    tweet = tweet.text

    doc = nlp(tweet)

    for word in doc:

        if word.has_vector:

            pass

        else:

            if id % 100000000 == 0:

                print(id)

                print(str(word))

            new_words = split_multi_word(str(word), spell, 4)

            if id % 100000000 == 0:

                print(new_words)

            try:

                tweet_ = tweet

                tweet = re.sub(str(word), new_words, tweet)

                manage_no_dictionable.dic[tweet_] = tweet

            except:

                pass

    return tweet





def is_vectorable(word, spell):

    if len(word) < 3:

        return False

    if spell.known([word]):

        return word

    else:

        for i in range(3, len(word)):

            w1 = word[0:i]

            w2 = word[i:]



            w1_r = is_vectorable(w1, spell)

            w2_r = is_vectorable(w2, spell)



            if w1_r and w2_r:

                return w1_r + ' ' + w2_r



        return False





def split_multi_word(word, spell, depth):

    try:

        aux = split_multi_word.dict[word]

        return aux[0], aux[1]

    except:

        if word == '':

            return '', 0

        else:

            if spell.known([word]):

                return word, 1/len(word)

            else:

                if depth > 0:

                    the_score = len(word)

                    the_sub_words = word

                    for l in range(min(12, len(word)), 0, -1):

                        for i in range(0, len(word)-l+1):

                            sub_word = word[i:i+l]

                            if spell.known([sub_word]):

                                score = 1/len(sub_word)



                                word_left = word[0:i]

                                word_right = word[i+l:]



                                sub_words_left, score_left = split_multi_word(word_left, spell, depth - 1)

                                sub_words_right, score_right = split_multi_word(word_right, spell, depth - 1)



                                score = score_left + score + score_right

                                sub_words = sub_words_left + ' ' + sub_word + ' ' + sub_words_right



                                if the_score > score:

                                    the_score = score

                                    the_sub_words = sub_words

                else:

                    the_score = 1/len(word)

                    the_sub_words = word

                split_multi_word.dict[word] = [the_sub_words, the_score]

                return the_sub_words, the_score





split_multi_word.dict = {}
print("Parsing texts...")

print("url characters...")

df_train['text_cleaned'] = df_train.apply(lambda tweet: manage_url_tweet_characters(tweet), axis=1)

df_test['text_cleaned'] = df_test.apply(lambda tweet: manage_url_tweet_characters(tweet), axis=1)

print('contractions')

df_train['text_cleaned'] = df_train.apply(lambda tweet: manage_contractions(tweet), axis=1)

df_test['text_cleaned'] = df_test.apply(lambda tweet: manage_contractions(tweet), axis=1)

print('manage_tricks')

df_train['text_cleaned'] = df_train.apply(lambda tweet: manage_tricks(tweet), axis=1)

df_test['text_cleaned'] = df_test.apply(lambda tweet: manage_tricks(tweet), axis=1)

print('ner')

df_train['text_cleaned'] = df_train.apply(lambda tweet: manage_ner(tweet, nlp_ents), axis=1)

df_test['text_cleaned'] = df_test.apply(lambda tweet: manage_ner(tweet, nlp_ents), axis=1)

print('multi_word')

df_train['text_cleaned'] = df_train.apply(lambda tweet: manage_no_dictionable(tweet, nlp_vects, spell), axis=1)

df_test['text_cleaned'] = df_test.apply(lambda tweet: manage_no_dictionable(tweet, nlp_vects, spell), axis=1)

print('done')
df_mislabeled = df_train.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

df_mislabeled.index.tolist()
df_train['target_relabeled'] = df_train['target'].copy() 



df_train.loc[df_train['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_relabeled'] = 0

df_train.loc[df_train['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_relabeled'] = 0

df_train.loc[df_train['text'] == 'To fight bioterrorism sir.', 'target_relabeled'] = 0

df_train.loc[df_train['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_relabeled'] = 1

df_train.loc[df_train['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_relabeled'] = 1

df_train.loc[df_train['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_relabeled'] = 0

df_train.loc[df_train['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_relabeled'] = 0

df_train.loc[df_train['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_relabeled'] = 1

df_train.loc[df_train['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_relabeled'] = 1

df_train.loc[df_train['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_relabeled'] = 0

df_train.loc[df_train['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_relabeled'] = 0

df_train.loc[df_train['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_relabeled'] = 0

df_train.loc[df_train['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_relabeled'] = 0

df_train.loc[df_train['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_relabeled'] = 0

df_train.loc[df_train['text'] == "Caution: breathing may be hazardous to your health.", 'target_relabeled'] = 1

df_train.loc[df_train['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_relabeled'] = 0

df_train.loc[df_train['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_relabeled'] = 0

df_train.loc[df_train['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_relabeled'] = 0
K = 2

skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)



DISASTER = df_train['target'] == 1

print('Whole Training Set Shape = {}'.format(df_train.shape))

print('Whole Training Set Unique keyword Count = {}'.format(df_train['keyword'].nunique()))

print('Whole Training Set Target Rate (Disaster) {}/{} (Not Disaster)'.format(df_train[DISASTER]['target_relabeled'].count(), df_train[~DISASTER]['target_relabeled'].count()))



for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train['text_cleaned'], df_train['target']), 1):

    print('\nFold {} Training Set Shape = {} - Validation Set Shape = {}'.format(fold, df_train.loc[trn_idx, 'text_cleaned'].shape, df_train.loc[val_idx, 'text_cleaned'].shape))

    print('Fold {} Training Set Unique keyword Count = {} - Validation Set Unique keyword Count = {}'.format(fold, df_train.loc[trn_idx, 'keyword'].nunique(), df_train.loc[val_idx, 'keyword'].nunique()))    
class ClassificationReport(Callback):

    

    def __init__(self, train_data=(), validation_data=()):

        super(Callback, self).__init__()

        

        self.X_train, self.y_train = train_data

        self.train_precision_scores = []

        self.train_recall_scores = []

        self.train_f1_scores = []

        

        self.X_val, self.y_val = validation_data

        self.val_precision_scores = []

        self.val_recall_scores = []

        self.val_f1_scores = [] 

               

    def on_epoch_end(self, epoch, logs={}):

        train_predictions = np.round(self.model.predict(self.X_train, verbose=0))        

        train_precision = precision_score(self.y_train, train_predictions, average='macro')

        train_recall = recall_score(self.y_train, train_predictions, average='macro')

        train_f1 = f1_score(self.y_train, train_predictions, average='macro')

        self.train_precision_scores.append(train_precision)        

        self.train_recall_scores.append(train_recall)

        self.train_f1_scores.append(train_f1)

        

        val_predictions = np.round(self.model.predict(self.X_val, verbose=0))

        val_precision = precision_score(self.y_val, val_predictions, average='macro')

        val_recall = recall_score(self.y_val, val_predictions, average='macro')

        val_f1 = f1_score(self.y_val, val_predictions, average='macro')

        self.val_precision_scores.append(val_precision)        

        self.val_recall_scores.append(val_recall)        

        self.val_f1_scores.append(val_f1)

        

        print('\nEpoch: {} - Training Precision: {:.6} - Training Recall: {:.6} - Training F1: {:.6}'.format(epoch + 1, train_precision, train_recall, train_f1))

        print('Epoch: {} - Validation Precision: {:.6} - Validation Recall: {:.6} - Validation F1: {:.6}'.format(epoch + 1, val_precision, val_recall, val_f1))  
%%time



bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)
class DisasterDetector:

    

    def __init__(self, bert_layer, max_seq_length=128, lr=0.0001, epochs=15, batch_size=32):

        

        # BERT and Tokenization params

        self.bert_layer = bert_layer

        

        self.max_seq_length = max_seq_length        

        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()

        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()

        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

        

        # Learning control params

        self.lr = lr

        self.epochs = epochs

        self.batch_size = batch_size

        

        self.models = []

        self.scores = {}

        

        

    def encode(self, texts):

                

        all_tokens = []

        all_masks = []

        all_segments = []



        for text in texts:

            text = self.tokenizer.tokenize(text)

            text = text[:self.max_seq_length - 2]

            input_sequence = ['[CLS]'] + text + ['[SEP]']

            pad_len = self.max_seq_length - len(input_sequence)



            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)

            tokens += [0] * pad_len

            pad_masks = [1] * len(input_sequence) + [0] * pad_len

            segment_ids = [0] * self.max_seq_length



            all_tokens.append(tokens)

            all_masks.append(pad_masks)

            all_segments.append(segment_ids)



        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    

    

    def build_model(self):

        

        input_word_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_word_ids')

        input_mask = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_mask')

        segment_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='segment_ids')    

        

        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])   

        clf_output = sequence_output[:, 0, :]

        out = Dense(1, activation='sigmoid')(clf_output)

        

        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

        optimizer = SGD(learning_rate=self.lr, momentum=0.8)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        

        return model

    

    

    def train(self, X):

        

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X['text_cleaned'], X['keyword'])):

            

            print('\nFold {}\n'.format(fold))

        

            X_trn_encoded = self.encode(X.loc[trn_idx, 'text_cleaned'].str.lower())

            y_trn = X.loc[trn_idx, 'target_relabeled']

            X_val_encoded = self.encode(X.loc[val_idx, 'text_cleaned'].str.lower())

            y_val = X.loc[val_idx, 'target_relabeled']

        

            # Callbacks

            metrics = ClassificationReport(train_data=(X_trn_encoded, y_trn), validation_data=(X_val_encoded, y_val))

            

            # Model

            model = self.build_model()        

            model.fit(X_trn_encoded, y_trn, validation_data=(X_val_encoded, y_val), callbacks=[metrics], epochs=self.epochs, batch_size=self.batch_size)

            

            self.models.append(model)

            self.scores[fold] = {

                'train': {

                    'precision': metrics.train_precision_scores,

                    'recall': metrics.train_recall_scores,

                    'f1': metrics.train_f1_scores                    

                },

                'validation': {

                    'precision': metrics.val_precision_scores,

                    'recall': metrics.val_recall_scores,

                    'f1': metrics.val_f1_scores                    

                }

            }

                    

                

    def plot_learning_curve(self):

        

        fig, axes = plt.subplots(nrows=K, ncols=2, figsize=(20, K * 6), dpi=100)

    

        for i in range(K):

            

            # Classification Report curve

            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[i].history.history['val_accuracy'], ax=axes[i][0], label='val_accuracy')

            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['precision'], ax=axes[i][0], label='val_precision')

            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['recall'], ax=axes[i][0], label='val_recall')

            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['f1'], ax=axes[i][0], label='val_f1')        



            axes[i][0].legend() 

            axes[i][0].set_title('Fold {} Validation Classification Report'.format(i), fontsize=14)



            # Loss curve

            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['loss'], ax=axes[i][1], label='train_loss')

            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['val_loss'], ax=axes[i][1], label='val_loss')



            axes[i][1].legend() 

            axes[i][1].set_title('Fold {} Train / Validation Loss'.format(i), fontsize=14)



            for j in range(2):

                axes[i][j].set_xlabel('Epoch', size=12)

                axes[i][j].tick_params(axis='x', labelsize=12)

                axes[i][j].tick_params(axis='y', labelsize=12)



        plt.show()

        

        

    def predict(self, X):

        

        X_test_encoded = self.encode(X['text_cleaned'].str.lower())

        y_pred = np.zeros((X_test_encoded[0].shape[0], 1))



        for model in self.models:

            y_pred += model.predict(X_test_encoded) / len(self.models)



        return y_pred
clf = DisasterDetector(bert_layer, max_seq_length=128, lr=0.0001, epochs=10, batch_size=32)



clf.train(df_train)
clf.plot_learning_curve()
y_pred = clf.predict(df_test)



model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

model_submission['target'] = np.round(y_pred).astype('int')

model_submission.to_csv('model_submission.csv', index=False)

model_submission.describe()
TRAIN_FEATURES = ['id', 'keyword', 'location', 'text', 'target', 'text_cleaned', 'target_relabeled']

TEST_FEATURES = ['id', 'keyword', 'location', 'text', 'target', 'text_cleaned']



df_train[TRAIN_FEATURES].to_pickle('train.pkl')

df_test[TEST_FEATURES].to_pickle('test.pkl')



print('Training Set Shape = {}'.format(df_train[TRAIN_FEATURES].shape))

print('Training Set Memory Usage = {:.2f} MB'.format(df_train[TRAIN_FEATURES].memory_usage().sum() / 1024**2))

print('Test Set Shape = {}'.format(df_test[TEST_FEATURES].shape))

print('Test Set Memory Usage = {:.2f} MB'.format(df_test[TEST_FEATURES].memory_usage().sum() / 1024**2))