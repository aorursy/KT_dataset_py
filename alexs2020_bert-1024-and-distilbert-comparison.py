import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
## Had some problems here with unknown errors corresponding to DistilBERT as mentioned here: https://github.com/huggingface/transformers/issues/1829
# !pip install transformers
## ... so installed from source
!pip install git+https://github.com/huggingface/transformers

import transformers

# We will use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
    
import tokenization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import urllib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
train_data = pd.read_csv("/home/train.csv")
test_data = pd.read_csv("/home/test.csv")
sample_sub = pd.read_csv("/home/sample_submission.csv")

#train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
#test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
#sample_sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
print('Train data shape: {}'.format(train_data.shape))
print('Test data shape: {}'.format(test_data.shape))
print('Sample submission shape: {}'.format(sample_sub.shape))
train_data.head()
label = 'Disaster', 'Non-Disaster'
data = train_data.target.value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(data, labels=label,  startangle=90, autopct='%1.1f%%')
train_data.target.value_counts() #Label counts
train_data.isnull().sum()
## Adopted from https://www.kaggle.com/sagar7390/nlp-on-disaster-tweets-eda-glove-bert-using-tfhub/comments#data
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=train_data[train_data['target']==1]['text'].str.len()
ax1.hist(tweet_len, color='red')
ax1.set_title('disaster tweets')
tweet_len=train_data[train_data['target']==0]['text'].str.len()
ax2.hist(tweet_len,color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
abbreviations = {
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

def convert_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text
def clean_tweets(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    html=re.compile(r'<.*?>')
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    table=str.maketrans('','',string.punctuation)
    
    text = url.sub(r'',text)
    text = html.sub(r'',text)
    text = emoji_pattern.sub(r'', text)
    text = text.translate(table)

    return text
test_data.head
sample_sub.head
%%time
train_data['text']=train_data['text'].apply(lambda x : clean_tweets(x))
test_data['text']=test_data['text'].apply(lambda x : clean_tweets(x))

test_data['text']=test_data['text'].apply(lambda x : convert_abbrev_in_text(x))
train_data['text']=train_data['text'].apply(lambda x : convert_abbrev_in_text(x))
dropout_num=0
max_length=160
max_sequence_length=512
learning_rate_BERT=1e-6
val_split_BERT=0.2
epochs_BERT=1
batch_size_BERT=8
# Adopted with max_len placeholder for easy parameter manipulation from https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
def bert_encode(texts, tokenizer, max_len=max_sequence_length):
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
# Adopted with dropout_num, learning_rate and max_len placeholders for easy parameter manipulation from https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
def build_model(bert_layer, Dropout_num=dropout_num, max_len=max_sequence_length):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    
    if Dropout_num == 0:
        # Without Dropout
        out = Dense(1, activation='sigmoid')(clf_output)
    else:
        # With Dropout(Dropout_num), Dropout_num > 0
        x = Dropout(Dropout_num)(clf_output)
        out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=learning_rate_BERT), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
%%time
##module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# Adopted the max_len with placeholders from https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert?scriptVersionId=42583022
train_input = bert_encode(train_data.text.values, tokenizer, max_length)
test_input = bert_encode(test_data.text.values, tokenizer, max_length)
train_labels = train_data.target.values
model_BERT_1024 = build_model(bert_layer, max_len=max_length)
model_BERT_1024.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history_1024 = model_BERT_1024.fit(
    train_input, train_labels,
    validation_split=val_split_BERT,
    epochs=epochs_BERT,
    callbacks=[checkpoint],
    batch_size=batch_size_BERT
)
%%time
model_BERT_1024.load_weights('model.h5')
test_pred = model_BERT_1024.predict(test_input)
# Thanks to https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert?scriptVersionId=42583022
# Prediction by BERT model with my tuning for the training data - for the Confusion Matrix
%%time
train_pred_BERT = model_BERT_1024.predict(train_input)
train_pred_BERT_int = train_pred_BERT.round().astype('int')
sample_sub['target'] = test_pred.round().astype(int)
sample_sub.to_csv("./submission.csv", index=False, header=True)
check_sample = pd.read_csv("./submission.csv")
check_sample.head(20)
# For DistilBERT:
## https://www.kaggle.com/xhlulu/disaster-nlp-distilbert-in-tf

%%time
transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer_DistilBERT = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def build_model_DistilBERT(transformer, max_len=max_sequence_length):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=learning_rate_BERT), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
model_DistilBERT = build_model_DistilBERT(transformer_layer, max_len=max_length)
model_DistilBERT.summary()
train_input_DistilBERT = bert_encode(train_data.text.values, tokenizer_DistilBERT, max_len=160)
test_input_DistilBERT = bert_encode(test_data.text.values, tokenizer_DistilBERT, max_len=160)
train_labels_DistilBERT = train_data.target.values
train_history_DistilBERT = model_DistilBERT.fit(
    train_input, train_labels,
    validation_split=val_split_BERT,
    epochs=epochs_BERT,
    batch_size=batch_size_BERT
)
%%time
test_pred_DistilBERT = model_DistilBERT.predict(test_input_DistilBERT, verbose=1)
# Thanks to https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert?scriptVersionId=42583022
# Prediction by BERT model with my tuning for the training data - for the Confusion Matrix
%%time
train_pred_DistilBERT = model_DistilBERT.predict(train_input_DistilBERT)
train_pred_DistilBERT_int = train_pred_DistilBERT.round().astype('int')
sample_sub['target'] = test_pred_DistilBERT.round().astype(int)
sample_sub.to_csv('submission_DistilBERT.csv', index=False, header=True)

check_sample_DistilBERT = pd.read_csv("./submission_DistilBERT.csv")
check_sample_DistilBERT.head(20)
# Showing Confusion Matrix, thanks to https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert?scriptVersionId=42583022
def plot_cm(y_true, y_pred, title, figsize=(5,5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
# Showing Confusion Matrix for BERT model
plot_cm(train_pred_BERT_int, train_data['target'].values, 'Confusion matrix for BERT 1024 model', figsize=(7,7))
# Showing Confusion Matrix for BERT model
plot_cm(train_pred_DistilBERT_int, train_data['target'].values, 'Confusion matrix for DistilBERT model', figsize=(7,7))
