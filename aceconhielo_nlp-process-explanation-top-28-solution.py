 # Libraries

import pandas as pd

import numpy as np



# Aux Libraries

from cleaning_text_aux import *

from cleaning_text_process import *



# For string analysis and manipulation

import re

import string

from bs4 import BeautifulSoup



import nltk

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



from wordcloud import STOPWORDS





# Visualization

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.figure_factory as ff





# Modelling

from sklearn import feature_extraction, linear_model, model_selection, preprocessing



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



SEED = 42

np.random.seed(SEED)



# CONSTANTS

BAR_COLORS = ['#005e7d','#dd9933']





def dark_theme_layout(fig, title, height):

    fig.update_layout(

        title_text = title,

        barmode='overlay',

        font_color="white",

        legend = dict(font=dict(color='white')),

        plot_bgcolor  ='#161a28',

        paper_bgcolor = '#1e2130',

    height=height)



    fig.update_xaxes(zeroline=False, showline=False,showgrid=False,gridwidth=1)

    fig.update_yaxes(zeroline=False, showline=False,showgrid=False,gridwidth=1)
path_data = '/kaggle/input/nlp-getting-started/'

df_train = pd.read_csv(path_data + 'train.csv')

df_test = pd.read_csv(path_data + 'test.csv')

ssub = pd.read_csv(path_data + 'sample_submission.csv')

# Disaster Tweet Example

print('Tagged as disaster: ',df_train[df_train['target']==1]['text'].values[0])

# Non Disaster Tweet Example

print('Tagged as non disaster: ',df_train[df_train['target']==0]['text'].values[0])

count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(df_train["text"]) # Count the words in each tweet

test_vectors = count_vectorizer.transform(df_test["text"])





print(train_vectors[0].todense().shape) # (1, 21637) => That means that there are 21637 unique words in all tweets

print(train_vectors[0].todense())
# Our vectors are really big, so we want to push the model's weight toward 0 without completely discounting different words

# Ridge Regression is a good way to do this



clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf,train_vectors, df_train['target'], cv=5, scoring = 'f1')

round(np.mean(scores),2)
# Get the predictions

clf.fit(train_vectors, df_train['target'])

ssub['target'] = clf.predict(test_vectors)

ssub.to_csv("submission_tutorial.csv", index=False)
# train target distribution

train_td = df_train[['id','target']].groupby('target').sum().reset_index()

train_td.columns = ['target', 'percentage']





train_td.loc[train_td.target==0,'target'] = 'Not real disaster'

train_td.loc[train_td.target==1,'target'] = 'Real disaster'





train_td['percentage'] = train_td.apply(lambda x: (x['percentage']/ sum(train_td['percentage'])), axis=1)

fig = go.Figure(data=[go.Bar(

    x=train_td['target'],

    y=train_td['percentage'],

    marker_color=BAR_COLORS # marker color can be a single color value or an iterable

)])





title = 'Train Class Distribution'

height = 500

dark_theme_layout(fig, title, height)





fig.update_layout(

    yaxis_tickformat = '%',

    yaxis= dict(title_text='% Tweets from total'),

    xaxis=dict(title_text='Type of tweet')

)



fig.show()





print('Dataset train.csv analysis')

display(df_train.head())

display(df_train.describe())

display(df_train.isnull().sum(axis=0))
# df_train_man_clean = clean_text_process_manual(df_train)

# df_test_man_clean = clean_text_process_manual(df_test)
#df_train_man_clean.to_csv('df_train_man_clean.csv',index=False)

#df_test_man_clean.to_csv('df_test_man_clean.csv',index=False)



df_train_man_clean = pd.read_csv('../input/cleaned-tweets/df_train_man_clean.csv')

df_test_man_clean = pd.read_csv('../input/cleaned-tweets/df_test_man_clean.csv')
print('Cleaning and transforming train dataset')

df_train_cleaned = cleaning_text_process_auto(df_train_man_clean)

print('Finish!')
print('Cleaning and transforming test dataset')

df_test_cleaned = cleaning_text_process_auto(df_test_man_clean)

print('Finish!')
train_features = df_train_cleaned.copy()

test_features = df_test_cleaned.copy()



def create_nlp_features(df, text_col):

    

    df['n_char'] = df[text_col].apply(lambda x: sum([len(character) for character in x]))

    df['n_word'] = df[text_col].apply(lambda x: len(x))

    df['nunique_word'] = df[text_col].apply(lambda x: len(set(x)))

    df['len_word'] = df[text_col].apply(lambda x: round(np.mean([len(word) for word in x]),2))

    

    df['n_mention'] = df['mentions_l'].apply(lambda x: len(x))

    df['n_hashtag'] = df['hashtags_l'].apply(lambda x: len(x))



    

    return(df)







train_features = create_nlp_features(df_train_cleaned, 'text_tokenized')

train_features = create_nlp_features(df_train_cleaned, 'text_tokenized')



test_features = create_nlp_features(df_test_cleaned, 'text_tokenized_lem')

test_features = create_nlp_features(df_test_cleaned, 'text_tokenized_lem')



test_features = df_test_cleaned

# REMOVE TEXT WITH NULL TEXT CLEANED ONLY IN TRAIN

train_features = df_train_cleaned[df_train_cleaned.len_word.isnull() == False]

train_features.reset_index(drop = True, inplace = True)

test_features.reset_index(drop = True, inplace = True)
NLP_FEATURES = ['n_char','n_word','len_word','n_mention','n_hashtag','n_url','n_punt','n_stopw','nunique_word']

NLP_FEATURES_TITLES = tuple(['Distribution of feature: '+feature for feature in NLP_FEATURES])
fig = make_subplots(rows=5, cols=2, subplot_titles=NLP_FEATURES_TITLES)



train_t1 = train_features[train_features.target==1] # Disasters tweets

train_t0 = train_features[train_features.target==0] # Non Disasters tweets



i = 1

for index, nf in enumerate(NLP_FEATURES):

    

    if (index%2 == 0) & (index>=2):

        i+=1

    j = index%2+1

    

    trace_a = go.Histogram(x=train_t1[nf], histnorm='percent', name=nf+' - DT' ,marker_color=BAR_COLORS[1])

    trace_b = go.Histogram(x=train_t0[nf], histnorm='percent', name=nf+' - NDT' ,marker_color=BAR_COLORS[0])



    fig.append_trace(trace_a, i, j)

    fig.append_trace(trace_b, i, j)

    

    

fig.update_traces(opacity=0.75)



title = 'NLP Features Distribution Analysis between Targets'

height = 1600

dark_theme_layout(fig, title, height)

fig.update_layout(showlegend=False)

fig.show()

fig = make_subplots(rows=5, cols=2, subplot_titles=NLP_FEATURES_TITLES)



i = 1

for index, nf in enumerate(NLP_FEATURES):

    

    if (index%2 == 0) & (index>=2):

        i+=1

    j = index%2+1

    

    trace_a = go.Histogram(x=train_features[nf], histnorm='probability', name=nf+'-Disaster Tweets' ,marker_color=BAR_COLORS[1])

    trace_b = go.Histogram(x=test_features[nf], histnorm='probability', name=nf+'-Non Disaster Tweets' ,marker_color=BAR_COLORS[0])



    fig.append_trace(trace_a, i, j)

    fig.append_trace(trace_b, i, j)

    

    



    

fig.update_traces(opacity=0.75)



title = 'NLP Features distribution - Train vs Test Analysis'

height = 1600

dark_theme_layout(fig, title, height)

fig.update_layout(showlegend=False)

fig.show()

from nltk.util import ngrams # function for making ngrams

import collections

import itertools





train_ngrams = train_features[['id', 'text_tokenized','target']].copy()





train_ngrams_t1 = train_ngrams.loc[train_ngrams.target==1]

train_ngrams_t0 = train_ngrams.loc[train_ngrams.target==0]



# Corpus creation



corpus_t1 = list(itertools.chain(*train_ngrams_t1['text_tokenized'].values))

corpus_t0 = list(itertools.chain(*train_ngrams_t0['text_tokenized'].values))





t1_unigrams = collections.Counter(ngrams(corpus_t1, 1)).most_common(10)

t0_unigrams = collections.Counter(ngrams(corpus_t0, 1)).most_common(10)



t1_bigrams = collections.Counter(ngrams(corpus_t1, 2)).most_common(10)

t0_bigrams = collections.Counter(ngrams(corpus_t0, 2)).most_common(10)



t1_trigrams = collections.Counter(ngrams(corpus_t1, 3)).most_common(10)

t0_trigrams = collections.Counter(ngrams(corpus_t0, 3)).most_common(10)







t1_unigrams_df = pd.DataFrame.from_dict(dict(t1_unigrams), orient='index').reset_index()

t1_unigrams_df.columns = ['Ngram','Frequency']

t0_unigrams_df = pd.DataFrame.from_dict(dict(t0_unigrams), orient='index').reset_index()

t0_unigrams_df.columns = ['Ngram','Frequency']



t1_bigrams_df = pd.DataFrame.from_dict(dict(t1_bigrams), orient='index').reset_index()

t1_bigrams_df.columns = ['Ngram','Frequency']

t0_bigrams_df = pd.DataFrame.from_dict(dict(t0_bigrams), orient='index').reset_index()

t0_bigrams_df.columns = ['Ngram','Frequency']

                         

t1_trigrams_df = pd.DataFrame.from_dict(dict(t1_trigrams), orient='index').reset_index()

t1_trigrams_df.columns = ['Ngram','Frequency']

t0_trigrams_df = pd.DataFrame.from_dict(dict(t0_trigrams), orient='index').reset_index()

t0_trigrams_df.columns = ['Ngram','Frequency']





separator = '-'

t1_unigrams_df['Ngram'] = t1_unigrams_df['Ngram'].apply(lambda x: separator.join(list(x)))

t0_unigrams_df['Ngram'] = t0_unigrams_df['Ngram'].apply(lambda x: separator.join(list(x)))



t1_bigrams_df['Ngram'] = t1_bigrams_df['Ngram'].apply(lambda x: separator.join(list(x)))

t0_bigrams_df['Ngram'] = t0_bigrams_df['Ngram'].apply(lambda x: separator.join(list(x)))



t1_trigrams_df['Ngram'] = t1_trigrams_df['Ngram'].apply(lambda x: separator.join(list(x)))

t0_trigrams_df['Ngram'] = t0_trigrams_df['Ngram'].apply(lambda x: separator.join(list(x)))


title_unigrams = ['Unigrams - Disaster Tweets', 'Unigrams - Non Disaster Tweets']

fig = make_subplots(rows=1, cols=2, subplot_titles=title_unigrams)



trace_a = go.Bar(x=t1_unigrams_df['Ngram'],y=t1_unigrams_df['Frequency'], name='Unigrams - Disaster Tweets' ,marker_color=BAR_COLORS[1])

trace_b = go.Bar(x=t0_unigrams_df['Ngram'],y=t0_unigrams_df['Frequency'], name='Unigrams - Non Disaster Tweets' ,marker_color=BAR_COLORS[0])



fig.append_trace(trace_a, 1, 1)

fig.append_trace(trace_b, 1, 2)





title = 'Unigrams Analysis (After cleaning)'

height = 500

dark_theme_layout(fig, title, height)

fig.show()

title_bigrams = ['Bigrams - Disaster Tweets', 'Bigrams - Non Disaster Tweets']

fig = make_subplots(rows=1, cols=2, subplot_titles=title_unigrams)



trace_a = go.Bar(x=t1_bigrams_df['Ngram'],y=t1_bigrams_df['Frequency'], name='Bigrams - Disaster Tweets' ,marker_color=BAR_COLORS[1])

trace_b = go.Bar(x=t0_bigrams_df['Ngram'],y=t0_bigrams_df['Frequency'], name='Bigrams - Non Disaster Tweets' ,marker_color=BAR_COLORS[0])



fig.append_trace(trace_a, 1, 1)

fig.append_trace(trace_b, 1, 2)





title = 'Bigrams Analysis (After cleaning)'

height = 500

dark_theme_layout(fig, title, height)

fig.show()



title_bigrams = ['Trigrams - Disaster Tweets', 'Trigrams - Non Disaster Tweets']

fig = make_subplots(rows=1, cols=2, subplot_titles=title_unigrams)



trace_a = go.Bar(x=t1_trigrams_df['Ngram'],y=t1_trigrams_df['Frequency'], name='Trigrams - Disaster Tweets' ,marker_color=BAR_COLORS[1])

trace_b = go.Bar(x=t0_trigrams_df['Ngram'],y=t0_trigrams_df['Frequency'], name='Trigrams - Non Disaster Tweets' ,marker_color=BAR_COLORS[0])



fig.append_trace(trace_a, 1, 1)

fig.append_trace(trace_b, 1, 2)





title = 'Trigrams Analysis (After cleaning)'

height = 500

dark_theme_layout(fig, title, height)

fig.show()
glove_embeddings = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)



train_reduced = train_features.copy()

train_reduced = train_reduced[['id','text','text_tokenized','text_tokenized_lem','target']]



corpus_train_tokenized = list(itertools.chain(*train_features['text_tokenized'].values))

corpus_train_tokenized_lem = list(itertools.chain(*train_features['text_tokenized_lem'].values))



corpus_train_text_list = train_features['text'].apply(lambda s: s.split()).values 

corpus_train_text = list(itertools.chain(*corpus_train_text_list))
import operator

from tqdm import tqdm





def corpus_build(X):  

    vocab = {}

    for word in X:

        try:

            vocab[word] += 1

        except KeyError:

            vocab[word] = 1             

    return vocab



def check_embedding_coverage(corpus, embeddings):



    corpus = corpus_build(corpus)

    covered, oov = {}, {}

    n_covered, n_oov = 0,0

    not_cover = []

    for word in corpus:

        if word not in embeddings:

            not_cover.append(word)

        try:

            covered[word] = embeddings[word]

            n_covered += corpus[word]

        except:

            oov[word] = corpus[word]

            n_oov += corpus[word]

            

    # Get ratios

    voc_ratio = (len(covered)/len(corpus)*100)

    text_ratio = (n_covered/(n_covered + n_oov))*100

    print('Cover ', round(voc_ratio,2), '% of vocabulary and ',round(text_ratio,2), '% of the text')

    

    not_cover = list(set(not_cover))

    # Sort oov items

    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return (sorted_oov,not_cover)





print('Raw text')

corpus_text_oov, corpus_text_not_cover = check_embedding_coverage(corpus_train_text, glove_embeddings)       
print('Tokenized text')

corpus_tokenized_oov, corpus_tokenized_not_cover = check_embedding_coverage(corpus_train_tokenized, glove_embeddings)     
len(corpus_tokenized_not_cover)
print('Tokenized and lem text')

corpus_tokenized_lem_oov, corpus_tokenized_lem_not_cover = check_embedding_coverage(corpus_train_tokenized_lem, glove_embeddings)  
train_features['keyword'] = train_features['keyword'].fillna('no_keyword')

train_features['location'] = train_features['location'].fillna('no_location')

train_features['text_cleaned'] = train_features['text_tokenized_lem'].apply(lambda x: ' '.join(x))

test_features['text_cleaned'] = test_features['text_tokenized_lem'].apply(lambda x: ' '.join(x))
K = 10

skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)
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

        # -----------------

        # Train dataset

        # -----------------

        train_predictions = np.round(self.model.predict(self.X_train, verbose=0)) 

        

        # Get classification metrics

        train_precision = precision_score(self.y_train, train_predictions, average='macro')

        train_recall = recall_score(self.y_train, train_predictions, average='macro')

        train_f1 = f1_score(self.y_train, train_predictions, average='macro')

        

        # Storage classification metrics

        self.train_precision_scores.append(train_precision)        

        self.train_recall_scores.append(train_recall)

        self.train_f1_scores.append(train_f1)

        

        # -----------------

        # Validation dataset

        # -----------------

        val_predictions = np.round(self.model.predict(self.X_val, verbose=0))

        

        # Get classification metrics

        val_precision = precision_score(self.y_val, val_predictions, average='macro')

        val_recall = recall_score(self.y_val, val_predictions, average='macro')

        val_f1 = f1_score(self.y_val, val_predictions, average='macro')

        

        # Storage classification metrics

        self.val_precision_scores.append(val_precision)        

        self.val_recall_scores.append(val_recall)        

        self.val_f1_scores.append(val_f1)

        

    

        print('\nEpoch: {} - Training Precision: {:.6} - Training Recall: {:.6} - Training F1: {:.6}'.format(epoch + 1, train_precision, train_recall, train_f1))

        print('Epoch: {} - Validation Precision: {:.6} - Validation Recall: {:.6} - Validation F1: {:.6}'.format(epoch + 1, val_precision, val_recall, val_f1))  
%%time

bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)
from keras.utils.vis_utils import plot_model



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

        # Loop through the StratifiedKFold

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X['text_cleaned'], X['keyword'])):

  

            print('\nFold {}\n'.format(fold))

        

            # Train set

            # 1. Encode the cleaned text; storage encoded text and targets

            X_trn_encoded = self.encode(X.loc[trn_idx, 'text_cleaned'].str.lower())

            y_trn = X.loc[trn_idx, 'target']

            

            # Validation set

            # 1. Encode the cleaned text; storage encoded text and targets

            X_val_encoded = self.encode(X.loc[val_idx, 'text_cleaned'].str.lower())

            y_val = X.loc[val_idx, 'target']



            # Generate ClassificationReport

            

            # 1. Initialized the train_data and the validation_data for getting the metrics

            metrics = ClassificationReport(train_data=(X_trn_encoded, y_trn), validation_data=(X_val_encoded, y_val))

            

            # 2. Build the model

            model = self.build_model()        

            

            # 3. Fit the model

            model.fit(X_trn_encoded, y_trn, validation_data=(X_val_encoded, y_val), callbacks=[metrics], epochs=self.epochs, batch_size=self.batch_size)

            

            model_name = 'model_plot_'+str(fold)+'.png'

            plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

            

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
clf.train(train_features)
predictions_test = clf.predict(test_features)



model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

model_submission['target'] = np.round(predictions_test).astype('int')

model_submission.to_csv('model_submission_tokenized.csv', index=False)