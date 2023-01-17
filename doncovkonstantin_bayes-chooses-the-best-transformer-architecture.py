import pandas as pd

import numpy as np

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

import transformers as trfo 

import sklearn.model_selection as ms

import sklearn.metrics as m

from functools import partial

import hyperopt as ho

import pickle

import re

import string

import itertools

import time

import tqdm

import operator

import xgboost
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
train_df.loc[:5]
tokenizer = trfo.BertTokenizer.from_pretrained('bert-large-uncased')
tokenizer.encode('London!')
tokenizer.decode(101), tokenizer.decode(2414), tokenizer.decode(999), tokenizer.decode(102)
def build_vocab(sentences):

    vocab = {}

    for sentence in tqdm.tqdm(sentences):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
def check_coverage(vocab, embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm.tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of full text'.format(k / (k + i)))
vocab = build_vocab(train_df['text'].apply(lambda x: x.split()).values)

check_coverage(vocab, tokenizer.get_vocab())
train_df['text'] = train_df['text'].apply(lambda x: x.lower())



vocab = build_vocab(train_df['text'].apply(lambda x: x.split()).values)

check_coverage(vocab, tokenizer.get_vocab())
def remove_url(text):

    return re.sub(r'https?:\/\/t.co\/[A-Za-z0-9]+', '', text)
def remove_user(text):

    text = re.sub(r'\@[A-Za-z0-9]+', '', text)

    return text
train_df['text']  = train_df['text'].apply(lambda x: remove_url(x))

train_df['text']  = train_df['text'].apply(lambda x: remove_user(x))



vocab = build_vocab(train_df['text'].apply(lambda x: x.split()).values)

check_coverage(vocab, tokenizer.get_vocab())
abbreviations_mapping = {

    "$" : " dollar ",

    "€" : " euro ",

    "4ao" : "for adults only",

    "a.m" : "before midday",

    "a3" : "anytime anywhere anyplace",

    "aamof" : "as a matter of fact",

    "acct" : "account",

    "adih" : "another day in hell",

    "afaic" : "as far as i am concerned",

    "afaict" : "as far as i can tell",

    "afaik" : "as far as i know",

    "afair" : "as far as i remember",

    "afk" : "away from keyboard",

    "app" : "application",

    "approx" : "approximately",

    "apps" : "applications",

    "asap" : "as soon as possible",

    "asl" : "age, sex, location",

    "atk" : "at the keyboard",

    "ave." : "avenue",

    "aymm" : "are you my mother",

    "ayor" : "at your own risk", 

    "b&b" : "bed and breakfast",

    "b+b" : "bed and breakfast",

    "b.c" : "before christ",

    "b2b" : "business to business",

    "b2c" : "business to customer",

    "b4" : "before",

    "b4n" : "bye for now",

    "b@u" : "back at you",

    "bae" : "before anyone else",

    "bak" : "back at keyboard",

    "bbbg" : "bye bye be good",

    "bbc" : "british broadcasting corporation",

    "bbias" : "be back in a second",

    "bbl" : "be back later",

    "bbs" : "be back soon",

    "be4" : "before",

    "bfn" : "bye for now",

    "blvd" : "boulevard",

    "bout" : "about",

    "brb" : "be right back",

    "bros" : "brothers",

    "brt" : "be right there",

    "bsaaw" : "big smile and a wink",

    "btw" : "by the way",

    "bwl" : "bursting with laughter",

    "c/o" : "care of",

    "cet" : "central european time",

    "cf" : "compare",

    "cia" : "central intelligence agency",

    "csl" : "can not stop laughing",

    "cu" : "see you",

    "cul8r" : "see you later",

    "cv" : "curriculum vitae",

    "cwot" : "complete waste of time",

    "cya" : "see you",

    "cyt" : "see you tomorrow",

    "dae" : "does anyone else",

    "dbmib" : "do not bother me i am busy",

    "diy" : "do it yourself",

    "dm" : "direct message",

    "dwh" : "during work hours",

    "e123" : "easy as one two three",

    "eet" : "eastern european time",

    "eg" : "example",

    "embm" : "early morning business meeting",

    "encl" : "enclosed",

    "encl." : "enclosed",

    "etc" : "and so on",

    "faq" : "frequently asked questions",

    "fawc" : "for anyone who cares",

    "fb" : "facebook",

    "fc" : "fingers crossed",

    "fig" : "figure",

    "fimh" : "forever in my heart", 

    "ft." : "feet",

    "ft" : "featuring",

    "ftl" : "for the loss",

    "ftw" : "for the win",

    "fwiw" : "for what it is worth",

    "fyi" : "for your information",

    "g9" : "genius",

    "gahoy" : "get a hold of yourself",

    "gal" : "get a life",

    "gcse" : "general certificate of secondary education",

    "gfn" : "gone for now",

    "gg" : "good game",

    "gl" : "good luck",

    "glhf" : "good luck have fun",

    "gmt" : "greenwich mean time",

    "gmta" : "great minds think alike",

    "gn" : "good night",

    "g.o.a.t" : "greatest of all time",

    "goat" : "greatest of all time",

    "goi" : "get over it",

    "gps" : "global positioning system",

    "gr8" : "great",

    "gratz" : "congratulations",

    "gyal" : "girl",

    "h&c" : "hot and cold",

    "hp" : "horsepower",

    "hr" : "hour",

    "hrh" : "his royal highness",

    "ht" : "height",

    "ibrb" : "i will be right back",

    "ic" : "i see",

    "icq" : "i seek you",

    "icymi" : "in case you missed it",

    "idc" : "i do not care",

    "idgadf" : "i do not give a damn fuck",

    "idgaf" : "i do not give a fuck",

    "idk" : "i do not know",

    "ie" : "that is",

    "i.e" : "that is",

    "ifyp" : "i feel your pain",

    "IG" : "instagram",

    "iirc" : "if i remember correctly",

    "ilu" : "i love you",

    "ily" : "i love you",

    "imho" : "in my humble opinion",

    "imo" : "in my opinion",

    "imu" : "i miss you",

    "iow" : "in other words",

    "irl" : "in real life",

    "j4f" : "just for fun",

    "jic" : "just in case",

    "jk" : "just kidding",

    "jsyk" : "just so you know",

    "l8r" : "later",

    "lb" : "pound",

    "lbs" : "pounds",

    "ldr" : "long distance relationship",

    "lmao" : "laugh my ass off",

    "lmfao" : "laugh my fucking ass off",

    "lol" : "laughing out loud",

    "ltd" : "limited",

    "ltns" : "long time no see",

    "m8" : "mate",

    "mf" : "motherfucker",

    "mfs" : "motherfuckers",

    "mfw" : "my face when",

    "mofo" : "motherfucker",

    "mph" : "miles per hour",

    "mr" : "mister",

    "mrw" : "my reaction when",

    "ms" : "miss",

    "mte" : "my thoughts exactly",

    "nagi" : "not a good idea",

    "nbc" : "national broadcasting company",

    "nbd" : "not big deal",

    "nfs" : "not for sale",

    "ngl" : "not going to lie",

    "nhs" : "national health service",

    "nrn" : "no reply necessary",

    "nsfl" : "not safe for life",

    "nsfw" : "not safe for work",

    "nth" : "nice to have",

    "nvr" : "never",

    "nyc" : "new york city",

    "oc" : "original content",

    "og" : "original",

    "ohp" : "overhead projector",

    "oic" : "oh i see",

    "omdb" : "over my dead body",

    "omg" : "oh my god",

    "omw" : "on my way",

    "p.a" : "per annum",

    "p.m" : "after midday",

    "pm" : "prime minister",

    "poc" : "people of color",

    "pov" : "point of view",

    "pp" : "pages",

    "ppl" : "people",

    "prw" : "parents are watching",

    "ps" : "postscript",

    "pt" : "point",

    "ptb" : "please text back",

    "pto" : "please turn over",

    "qpsa" : "what happens", #"que pasa",

    "ratchet" : "rude",

    "rbtl" : "read between the lines",

    "rlrt" : "real life retweet", 

    "rofl" : "rolling on the floor laughing",

    "roflol" : "rolling on the floor laughing out loud",

    "rotflmao" : "rolling on the floor laughing my ass off",

    "rt" : "retweet",

    "ruok" : "are you ok",

    "sfw" : "safe for work",

    "sk8" : "skate",

    "smh" : "shake my head",

    "sq" : "square",

    "srsly" : "seriously", 

    "ssdd" : "same stuff different day",

    "tbh" : "to be honest",

    "tbs" : "tablespooful",

    "tbsp" : "tablespooful",

    "tfw" : "that feeling when",

    "thks" : "thank you",

    "tho" : "though",

    "thx" : "thank you",

    "tia" : "thanks in advance",

    "til" : "today i learned",

    "tl;dr" : "too long i did not read",

    "tldr" : "too long i did not read",

    "tmb" : "tweet me back",

    "tntl" : "trying not to laugh",

    "ttyl" : "talk to you later",

    "u" : "you",

    "u2" : "you too",

    "u4e" : "yours for ever",

    "utc" : "coordinated universal time",

    "w/" : "with",

    "w/o" : "without",

    "w8" : "wait",

    "wassup" : "what is up",

    "wb" : "welcome back",

    "wtf" : "what the fuck",

    "wtg" : "way to go",

    "wtpa" : "where the party at",

    "wuf" : "where are you from",

    "wuzup" : "what is up",

    "wywh" : "wish you were here",

    "yd" : "yard",

    "ygtr" : "you got that right",

    "ynk" : "you never know",

    "zzz" : "sleeping bored and tired"

}
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", 

                       "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 

                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",

                       "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 

                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 

                       "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",

                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 

                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 

                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 

                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 

                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 

                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", 

                       "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 

                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", 

                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" 

                      }

synonyms_mapping = {

    "&amp;": "&",

    "retweet": "response to",

    "wildfire": "flame",

    "reddit": "social network",

    "legionnaires": "disease",

    "thunderstorm": "storm",

    "sinkhole": "crater",

    "derailment": "runs off its rails",

    "windstorm": "storm",

    "twister": "tornado",

    "rescuers": "people who rescue",

    "whirlwind": "hurricane",

    "sandstorm": "storm",

    "mudslide": "avalanche",

    "curfew": "time when individuals are required to return to  their houses",

    "armageddon": "disaster"

}
def translate_with_mapping(text, dictionary):

    text = ' '.join([dictionary[t] if t in dictionary else t for t in text.split(' ')])

    return text
train_df['text']  = train_df['text'].apply(lambda x: translate_with_mapping(x, synonyms_mapping))

train_df['text']  = train_df['text'].apply(lambda x: translate_with_mapping(x, abbreviations_mapping))

train_df['text']  = train_df['text'].apply(lambda x: translate_with_mapping(x, contraction_mapping))



vocab = build_vocab(train_df['text'].apply(lambda x: x.split()).values)

check_coverage(vocab, tokenizer.get_vocab())
def remove_punct_dup(text):

    punc = set(string.punctuation) 



    newtext = []

    for k, g in itertools.groupby(text):

        if k in punc:

            newtext.append(k)

        else:

            newtext.extend(g)



    return ''.join(newtext)
train_df['text']  = train_df['text'].apply(lambda x: remove_punct_dup(x))



vocab = build_vocab(train_df['text'].apply(lambda x: x.split()).values)

check_coverage(vocab, tokenizer.get_vocab())
def init_tpu(tpu):

    tf.tpu.experimental.initialize_tpu_system(tpu)
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

init_tpu(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
def load_trials(name, remove_last=True):

    trials = pickle.load(open(name, 'rb'))

    if remove_last:

        trials = remove_last_trial(trials)

    return trials
def remove_last_trial(old_trials):



    trials = ho.Trials()

    

    for trial in old_trials.trials[:-1]:

        hyperopt_trial = ho.Trials().new_trial_docs(

                tids=[None],

                specs=[None],

                results=[None],

                miscs=[None])

        hyperopt_trial[0] = trial

        trials.insert_trial_docs(hyperopt_trial) 

        trials.refresh()

    return trials
def save_trials(trials):

    pickle.dump(trials, open(f'trials_{len(trials.trials)}.p', 'wb'))
def save_trials_and_call_objective(hparams, objective, trials, df, kf):

    print(f'save this len of trials: {len(trials.trials)}')

    save_trials(trials)

    loss = objective(df, kf, hparams)

    return loss
trials_file_name = '../input/ntrialsanneal17/anneal_trials_132.p'
try:

    trials = load_trials(trials_file_name, True)

    print('Found trials file')

except FileNotFoundError as e:

    trials = ho.Trials()

    print('Not found trials file')

len(trials.trials)
MAX_LENGTH = 200

BATCH_SIZE = 16

EPOCHS = 5
def encode_with_tokinizer(data, tokenizer, maximum_length) :

    input_ids = []

    attention_mask = []



    for i in range(len(data)):

        encoded = tokenizer.encode_plus(

            data[i],

            add_special_tokens=True,

            max_length=maximum_length,

            pad_to_max_length=True,

            return_attention_mask=True,

            return_token_type_ids=False

        )



        input_ids.append(encoded['input_ids'])

        attention_mask.append(encoded['attention_mask'])



    return np.array(input_ids), np.array(attention_mask)
def node_params(n_layers):

    params = {}

    

    params['pack_size'] = n_layers

    

    for n in range(n_layers):

        params['n_nodes_layer_{}'.format(n)] = ho.hp.quniform('n_nodes_{}_{}'.format(n_layers, n), 10, 2000, 25)

        params['dropout_layer_{}'.format(n)] = ho.hp.quniform('dropout_{}_{}'.format(n_layers, n), 0, 0.6, 0.05)



    return params
def create_model(transformer_model, hparams):

    input_ids = tf.keras.Input(shape=(MAX_LENGTH, ),dtype='int32')

    attention_mask = tf.keras.Input(shape=(MAX_LENGTH, ), dtype='int32')

    

    transformer = transformer_model([input_ids, attention_mask])    

    hidden_states = transformer[1]

    

    if hparams['hidden_states_size'] == 1:

        output = hidden_states[-1]

    else:

        hiddes_states_ind = list(range(-hparams['hidden_states_size'], 0, 1))

        output = tf.keras.layers.concatenate(tuple([hidden_states[i] for i in hiddes_states_ind]))



    if hparams['layers']['pack_size'] >= 1:

        output = tf.keras.layers.Dense(hparams['layers']['n_nodes_layer_0'], activation='relu')(output)

        output = tf.keras.layers.BatchNormalization()(output)

        output = tf.keras.layers.Dropout(hparams['layers']['dropout_layer_0'])(output)



    if hparams['layers']['pack_size'] >= 2:

        output = tf.keras.layers.Dense(hparams['layers']['n_nodes_layer_1'], activation='relu')(output)

        output = tf.keras.layers.BatchNormalization()(output)

        output = tf.keras.layers.Dropout(hparams['layers']['dropout_layer_1'])(output)

        

    if hparams['layers']['pack_size'] >= 3:

        output = tf.keras.layers.Dense(hparams['layers']['n_nodes_layer_2'], activation='relu')(output)

        output = tf.keras.layers.BatchNormalization()(output)

        output = tf.keras.layers.Dropout(hparams['layers']['dropout_layer_2'])(output) 

        

    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    model = tf.keras.models.Model(inputs = [input_ids, attention_mask], outputs = output)

    model.compile(tf.keras.optimizers.Adam(lr=hparams['lr_rate']), loss='binary_crossentropy', metrics=['accuracy'])

    return model
tokenizers = {

    'bert': trfo.BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True),

    'roberta': trfo.RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True),

    'distilbert': trfo.DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

}
models = {

    'bert': lambda : trfo.TFBertForSequenceClassification.from_pretrained('bert-large-uncased', output_hidden_states=True),

    'roberta': lambda : trfo.TFRobertaForSequenceClassification.from_pretrained('roberta-large', output_hidden_states=True),

    'distilbert': lambda : trfo.TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', output_hidden_states=True),

}
train_df = train_df.sample(frac=1).reset_index(drop=True)
kf = ms.KFold(n_splits=4, shuffle=False)
def cross_validate_transformer_hparams(df, kf, hparams):

        

    errors = np.zeros(0)

    input_ids, attention_mask = encode_with_tokinizer(df.text, tokenizers[hparams['model_name']], MAX_LENGTH)

            

    for train_index, val_index in kf.split(df):

        

        init_tpu(tpu)



        model = create_model(models[hparams['model_name']](), hparams)

  

        model.fit(

            x=[input_ids[train_index], attention_mask[train_index]], 

            y=df.iloc[train_index].target.values,

            validation_data=([input_ids[val_index], attention_mask[val_index]], df.iloc[val_index].target.values),

            verbose=2, 

            epochs=EPOCHS,

            batch_size=BATCH_SIZE

        )



        _, a = model.evaluate(x=[input_ids[val_index], attention_mask[val_index]], y=df.iloc[val_index].target.values, verbose=2)

        

        er = 1 - a

        

        errors = np.append(errors, er)

        del model

        

    mean_error = errors.mean()

    print(f'Hparams: {hparams}')

    print(f'Mean error: {mean_error}')

    print('--------------------------------------')

    

    del input_ids

    del attention_mask

    

    return mean_error
max_evals = 100
space = {

    'layers': ho.hp.choice('layers', [node_params(n) for n in [n for n in range(4)]]),

    'lr_rate': ho.hp.loguniform("lr_rate", np.log(0.00001), np.log(0.001)),

    'model_name': ho.hp.choice('model_name', ['bert', 'roberta', 'distilbert']),

    'hidden_states_size': ho.hp.choice('hidden_states_size', [n for n in range(1, 5)])

}



# with tpu_strategy.scope():

#     ho.fmin(fn=partial(save_trials_and_call_objective, objective=cross_validate_transformer_hparams, trials=trials, df=train_df, kf=kf), space=space, algo=ho.tpe.suggest, max_evals=max_evals, trials=trials)

# save_trials(trials)
max_evals = 200
# with tpu_strategy.scope():

#     ho.fmin(fn=partial(save_trials_and_call_objective, objective=cross_validate_transformer_hparams, trials=trials, df=train_df, kf=kf), space=space, algo=ho.anneal.suggest, max_evals=max_evals, trials=trials)

# save_trials(trials)
trials.trials
EPOCHS = 3
tr, hold, val = np.split(train_df, [int(.7*len(train_df)), int(.9*len(train_df))])

tr.shape, hold.shape, val.shape
def create_final_bert_model(train, val):

        

    input_ids = tf.keras.Input(shape=(MAX_LENGTH, ),dtype='int32')

    attention_mask = tf.keras.Input(shape=(MAX_LENGTH, ), dtype='int32')



    transformer = models['bert']()([input_ids, attention_mask])    

    hidden_states = transformer[1]



    output = hidden_states[-1]



    output = tf.keras.layers.Dense(175, activation='relu')(output)

    output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Dropout(0.5)(output)



    output = tf.keras.layers.Dense(1750, activation='relu')(output)

    output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Dropout(0.2)(output)



    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    final_model = tf.keras.models.Model(inputs = [input_ids, attention_mask], outputs = output)

    final_model.compile(tf.keras.optimizers.Adam(lr=1.0210068320376741e-05), loss='binary_crossentropy', metrics=['accuracy'])



    train_input_ids, train_attention_mask = encode_with_tokinizer(train.text.values, tokenizers['bert'], MAX_LENGTH)

    val_input_ids, val_attention_mask = encode_with_tokinizer(val.text.values, tokenizers['bert'], MAX_LENGTH)



    final_model.fit(

        x=[train_input_ids, train_attention_mask], 

        y=train.target.values,

        validation_data=([val_input_ids, val_attention_mask], val.target.values),

        verbose=1, 

        epochs=EPOCHS,

        batch_size=BATCH_SIZE

    )



    _, a = final_model.evaluate(x=[val_input_ids, val_attention_mask], y=val.target.values, verbose=2)



    er = 1 - a



    print(er)

    return final_model
def create_final_roberta_model(train, val):

        

    input_ids = tf.keras.Input(shape=(MAX_LENGTH, ),dtype='int32')

    attention_mask = tf.keras.Input(shape=(MAX_LENGTH, ), dtype='int32')



    transformer = models['roberta']()([input_ids, attention_mask])    

    hidden_states = transformer[1]



    hiddes_states_ind = list(range(-3, 0, 1))

    output = tf.keras.layers.concatenate(tuple([hidden_states[i] for i in hiddes_states_ind]))

    

    output = tf.keras.layers.Dense(1850, activation='relu')(output)

    output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Dropout(0.55)(output)



    output = tf.keras.layers.Dense(1975, activation='relu')(output)

    output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Dropout(0.6)(output)

    

    output = tf.keras.layers.Dense(50, activation='relu')(output)

    output = tf.keras.layers.BatchNormalization()(output)

    output = tf.keras.layers.Dropout(0.2)(output)





    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    final_model = tf.keras.models.Model(inputs = [input_ids, attention_mask], outputs = output)

    final_model.compile(tf.keras.optimizers.Adam(lr=1.0470485274874683e-05), loss='binary_crossentropy', metrics=['accuracy'])



    train_input_ids, train_attention_mask = encode_with_tokinizer(train.text.values, tokenizers['roberta'], MAX_LENGTH)

    val_input_ids, val_attention_mask = encode_with_tokinizer(val.text.values, tokenizers['roberta'], MAX_LENGTH)



    final_model.fit(

        x=[train_input_ids, train_attention_mask], 

        y=train.target.values,

        validation_data=([val_input_ids, val_attention_mask], val.target.values),

        verbose=1, 

        epochs=EPOCHS,

        batch_size=BATCH_SIZE

    )

    _, a = final_model.evaluate(x=[val_input_ids, val_attention_mask], y=val.target.values, verbose=2)



    er = 1 - a



    print(er)

    return final_model
def get_predicitons_with_ensemble(x, ensemble_of_classifiers, y=None):    

    bert_pred = get_prediciton_with_tokenizer(x, ensemble_of_classifiers['bert'], tokenizers['bert'])

    roberta_pred = get_prediciton_with_tokenizer(x, ensemble_of_classifiers['roberta'], tokenizers['roberta'])



    

    if y is not None:

        print(f'bert accuracy: {m.accuracy_score(y, np.round(bert_pred).astype(int))}')

        print(f'roberta accuracy: {m.accuracy_score(y, np.round(roberta_pred).astype(int))}')



    pred_results = np.array([bert_pred, roberta_pred])

    pred_results = np.swapaxes(pred_results,0,1)



    return pred_results
def get_prediciton_with_tokenizer(x, classifier, tokenizer):

    pred_results = pd.DataFrame()

    

    input_ids, attention_mask = encode_with_tokinizer(x, tokenizer, MAX_LENGTH)

    y_pred = classifier.predict([input_ids, attention_mask])[:, 0].reshape(-1)

    return y_pred

with tpu_strategy.scope():

    bert_model = create_final_bert_model(tr, hold)

    roberta_model = create_final_roberta_model(tr, hold)
ensemble_of_classifiers = {'bert': bert_model, 'roberta': roberta_model}
hold_ens_prediction = get_predicitons_with_ensemble(hold.text.values, ensemble_of_classifiers, hold.target.values)
X_stack_train = hold_ens_prediction

y_stack_train = hold.target.values
X_stack_train.shape, y_stack_train.shape
def cross_validate_xgb_hparams(hparams, x, y):



    estimator = xgboost.XGBClassifier(learning_rate=hparams['learning_rate'], max_depth=hparams['max_depth'], n_estimators=int(hparams['n_estimators']))

    cv_results = ms.cross_validate(estimator, x, y, cv=5, scoring='accuracy', n_jobs=3)



    mean_acc = np.mean(cv_results['test_score'])

    mean_error = 1 - mean_acc

    

    print(hparams)

    print(f'Mean error: {mean_error}')

    print('----------------------------------------')



    return mean_error
trials_xgb = ho.Trials()
max_evals_xgb = 50
learning_rate_xgb_arr = [0.1, 0.05, 0.0025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]

max_depth_xgb_arr = [2, 3, 4, 5, 6]
space_xgb = {

    'learning_rate': ho.hp.choice('learning_rate', learning_rate_xgb_arr),

    'max_depth': ho.hp.choice('max_depth', max_depth_xgb_arr),

    'n_estimators': ho.hp.quniform('n_estimators', 50, 2000, 5),

}



ho.fmin(fn=partial(cross_validate_xgb_hparams, x=X_stack_train, y=y_stack_train), space=space_xgb, algo=ho.tpe.suggest, max_evals=max_evals_xgb, trials=trials_xgb)
max_evals_xgb = 100
best_xgb = ho.fmin(fn=partial(cross_validate_xgb_hparams, x=X_stack_train, y=y_stack_train), space=space_xgb, algo=ho.anneal.suggest, max_evals=max_evals_xgb, trials=trials_xgb)
best_xgb
learning_rate_xgb = learning_rate_xgb_arr[best_xgb['learning_rate']]

max_depth_xgb = max_depth_xgb_arr[best_xgb['max_depth']]

n_estimators_xgb = int(best_xgb['n_estimators'])
learning_rate_xgb, max_depth_xgb, n_estimators_xgb
metalearner = xgboost.XGBClassifier(

    learning_rate=learning_rate_xgb, 

    max_depth=max_depth_xgb,

    n_estimators=n_estimators_xgb

)

metalearner.fit(X_stack_train, y_stack_train)
val_ens_predictions = get_predicitons_with_ensemble(val.text.values, ensemble_of_classifiers, val.target.values)
X_stack_val = val_ens_predictions



val_meta_predictions = metalearner.predict(X_stack_val)

X_stack_val.shape, val_meta_predictions.shape
m.accuracy_score(val_meta_predictions, val.target.values)
def get_test_predictions_using_metalearner(model, x, ids):

    prediction = model.predict(x)

    result = np.round(prediction).astype(int)

    output = pd.DataFrame({'id':ids,'target': result})

    return output
pred_df = pd.read_csv('../input/nlp-getting-started/test.csv')



pred_df['text'] = pred_df['text'].apply(lambda x: x.lower())



pred_df['text'] = pred_df['text'].apply(lambda x: remove_url(x))

pred_df['text'] = pred_df['text'].apply(lambda x: remove_user(x))



pred_df['text'] = pred_df['text'].apply(lambda x: translate_with_mapping(x, synonyms_mapping))

pred_df['text'] = pred_df['text'].apply(lambda x: translate_with_mapping(x, abbreviations_mapping))

pred_df['text'] = pred_df['text'].apply(lambda x: translate_with_mapping(x, contraction_mapping))



pred_df['text'] = pred_df['text'].apply(lambda x: remove_punct_dup(x))
pred_ens_predictions = get_predicitons_with_ensemble(pred_df.text.values, ensemble_of_classifiers)
get_test_predictions_using_metalearner(metalearner, pred_ens_predictions, pred_df.id).to_csv('submission.csv', index=False)