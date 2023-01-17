# Use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

!pip install sentencepiece
import tokenization

# text processing libraries
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# sklearn 
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
 
import warnings
warnings.filterwarnings('ignore')
train_aug = pd.read_csv('../input/augmented-data-bdc/train_aug.csv')
train_aug.drop('Unnamed: 0', axis=1, inplace=True)
train_aug.head()
#Training data
train = pd.read_excel('../input/fixed-bdc/Data BDC - Satria Data 2020/Data Latih/Data Latih BDC.xlsx')
print('Training data shape: ', train.shape)
train.head()
# Testing data 
test = pd.read_excel('../input/fixed-bdc/Data BDC - Satria Data 2020/Data Uji/Data Uji BDC.xlsx')
print('Testing data shape: ', test.shape)
test.head()
train['text'] = train['judul'] + ' ' + train['narasi']
test['text'] = test['judul'] + ' ' + test['narasi']
train = train.drop(columns=['ID', 'tanggal', 'judul', 'narasi', 'nama file gambar'])
test = test.drop(columns=['ID', 'tanggal', 'judul', 'narasi', 'nama file gambar'])

print(train.head())
print(test.head())
#Missing values in training set
train.isnull().sum()
#Missing values in test set
test.isnull().sum()
train['label'].value_counts()
# take copies of the data to leave the originals for BERT
train1 = train.copy()
test1 = test.copy()
# Applying a first round of text cleaning techniques

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower() # make text lower case
    text = re.sub('\[.*?\]', '', text) # remove text in square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text) # remove URLs
    text = re.sub('<.*?>+', '', text) # remove html tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub('\n', '', text) # remove words conatinaing numbers
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)

    return text
# emoji removal
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Applying the de=emojifying function to both test and training datasets
train1['text'] = train1['text'].apply(lambda x: remove_emoji(x))
test1['text'] = test1['text'].apply(lambda x: remove_emoji(x))
# text preprocessing function
def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer_reg = nltk.tokenize.RegexpTokenizer(r'\w+')
    
    nopunc = clean_text(text)
    tokenized_text = tokenizer_reg.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text

# Applying the cleaning function to both test and training datasets
train1['text'] = train1['text'].apply(lambda x: text_preprocessing(x))
test1['text'] = test1['text'].apply(lambda x: text_preprocessing(x))

# Let's take a look at the updated text
train1['text'].head()
train_full = pd.concat([train1, train_aug])
train_full.head()
#count_vectorizer = CountVectorizer()
count_vectorizer = CountVectorizer(ngram_range = (1,2), min_df = 1)
train_vectors = count_vectorizer.fit_transform(train1['text'])
test_vectors = count_vectorizer.transform(test1["text"])
train_aug_vectors = count_vectorizer.transform(train_full['text'])

## Keeping only non-zero elements to preserve space 
train_vectors.shape
train_vectors
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df = 2, max_df = 0.5)
train_tfidf = tfidf.fit_transform(train1['text'])
test_tfidf = tfidf.transform(test1["text"])
train_aug_tfidf = tfidf.fit_transform(train_full['text'])

train_tfidf.shape
# Fitting a simple Logistic Regression on BoW
logreg_bow = LogisticRegression(C=1.0)
logreg_bow.fit(train_vectors, train["label"])
scores = model_selection.cross_val_score(logreg_bow, train_vectors, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple Logistic Regression on TFIDF
logreg_tfidf = LogisticRegression(C=1.0)
scores = model_selection.cross_val_score(logreg_tfidf, train_tfidf, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple Naive Bayes on BoW
NB_bow = MultinomialNB()
NB_bow.fit(train_vectors, train["label"])
scores = model_selection.cross_val_score(NB_bow, train_vectors, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple Naive Bayes on TFIDF
NB_tfidf = MultinomialNB()
scores = model_selection.cross_val_score(NB_tfidf, train_tfidf, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple Random Forest on BoW
RF_bow = RandomForestClassifier()
RF_bow.fit(train_vectors, train["label"])
scores = model_selection.cross_val_score(RF_bow, train_vectors, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple Random Forest on TFIDF
RF_tfidf = RandomForestClassifier()
scores = model_selection.cross_val_score(RF_tfidf, train_tfidf, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple SVC on BoW
SVC_bow = SVC()
SVC_bow.fit(train_vectors, train["label"])
scores = model_selection.cross_val_score(SVC_bow, train_vectors, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple SVC on TFIDF
SVC_tfidf = SVC()
scores = model_selection.cross_val_score(SVC_tfidf, train_tfidf, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple XGB on BoW
XGB_bow = XGBClassifier()
XGB_bow.fit(train_vectors, train["label"])
scores = model_selection.cross_val_score(XGB_bow, train_vectors, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple XGB on TFIDF
XGB_tfidf = XGBClassifier()
scores = model_selection.cross_val_score(XGB_tfidf, train_tfidf, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple AdaBoost on BoW
ada_bow = AdaBoostClassifier()
ada_bow.fit(train_vectors, train["label"])
scores = model_selection.cross_val_score(ada_bow, train_vectors, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple AdaBoost on TFIDF
ada_tfidf = AdaBoostClassifier()
scores = model_selection.cross_val_score(ada_tfidf, train_tfidf, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple Extra on BoW
ext_bow = ExtraTreesClassifier()
ext_bow.fit(train_vectors, train["label"])
scores = model_selection.cross_val_score(ext_bow, train_vectors, train["label"], cv=5, scoring="f1")
scores.mean()
# Fitting a simple AdaBoost on TFIDF
ext_tfidf = ExtraTreesClassifier()
scores = model_selection.cross_val_score(ext_tfidf, train_tfidf, train["label"], cv=5, scoring="f1")
scores.mean()


import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets

import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

print('Using Tensorflow version:', tf.__version__)


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
             texts, 
             return_attention_masks=False, 
             return_token_type_ids=False,
             pad_to_max_length=True,
             max_length=maxlen)
    
    return np.array(enc_di['input_ids'])
def build_model(transformer, max_len=512):
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(2, activation='softmax')(cls_token) 
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Configuration
EPOCHS = 4
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MODEL = 'jplu/tf-xlm-roberta-large' # bert-base-multilingual-uncased
from tensorflow.keras.utils import to_categorical

# convert to one-hot-encoding-labels
train_full_labels = to_categorical(train_full['label'], num_classes=2)
train1_labels = to_categorical(train1['label'], num_classes=2)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train1['text'],
                                                  train1_labels,
                                                  stratify=train1_labels,
                                                  test_size=0.1,
                                                  random_state=2020)

X_train.shape, X_val.shape, y_train.shape, y_val.shape


# from sklearn.model_selection import StratifiedKFold

# tra_fold_df = []
# val_fold_df = []
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=88)
# for tra_idx, val_idx in skf.split(train1, train1[['label']]):
#     X_tra = train1.iloc[tra_idx]
#     X_val = train1.iloc[val_idx]
#     tra_fold_df.append(X_tra)
#     val_fold_df.append(X_val)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
MAX_LEN = 192

X_train = regular_encode(X_train.values, tokenizer, maxlen=MAX_LEN)

# X_train = regular_encode(X_train.values, tokenizer, maxlen=MAX_LEN)
X_val = regular_encode(X_val.values, tokenizer, maxlen=MAX_LEN)
X_test = regular_encode(test1['text'].values, tokenizer, maxlen=MAX_LEN)
# X_val = regular_encode(X_val.values, tokenizer, maxlen=MAX_LEN)
# X_test = regular_encode(test1['text'].values, tokenizer, maxlen=MAX_LEN)
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .batch(BATCH_SIZE)
)
%%time

with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
n_steps = X_train.shape[0] // BATCH_SIZE

train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=1
)
ext_test_pred = ext_bow.predict_proba(test_vectors)[:,1]
forest_test_pred = RF_bow.predict_proba(test_vectors)[:,1]
xgb_test_pred = XGB_bow.predict_proba(test_vectors)[:,1]
lg_test_pred = logreg_bow.predict_proba(test_vectors)[:,1]
ada_test_pred = ada_bow.predict_proba(test_vectors)[:,1]
roberta_test_pred = model.predict(test_dataset)[:,1]
X_train_predict = regular_encode(train1['text'].values, tokenizer, maxlen=MAX_LEN)


train_dataset_predict = (
    tf.data.Dataset
    .from_tensor_slices(X_train_predict)
    .batch(BATCH_SIZE)
)

# ext_train_pred = ext_bow.predict_proba(train_vectors)[:,1]
forest_train_pred = RF_bow.predict_proba(train_vectors)[:,1]
xgb_train_pred = XGB_bow.predict_proba(train_vectors)[:,1]
lg_train_pred = logreg_bow.predict_proba(train_vectors)[:,1]
ada_train_pred = ada_bow.predict_proba(train_vectors)[:,1]
roberta_train_pred = model.predict(train_dataset_predict)[:,1]

base_pred = pd.DataFrame({
    'ext':ext_train_pred.ravel(),
    'xgb':xgb_train_pred.ravel(), 
    'lg':lg_train_pred.ravel(),
    'ada':ada_train_pred.ravel(),
})

test_pred = pd.DataFrame({
    'ext':ext_test_pred.ravel(),
    'xgb':xgb_test_pred.ravel(), 
    'lg':lg_test_pred.ravel(),
    'ada':ada_test_pred.ravel(),
    
})
base_pred.head()
# Display numerical correlations between features on heatmap
sns.set(font_scale=1.1)
correlation_train = base_pred.corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.1f',
            cmap='coolwarm',
            square=True,
            mask=mask,
            linewidths=1)

plt.show()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
parameters = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 0.5, 1,10,100,1000], 
    'gamma': [1, 0.1, 0.001, 0.0001, 'auto'],
    'degree': [3, 4, 5]
}

final_model = GridSearchCV(SVC(), parameters, cv=5)
final_model.fit(base_pred, train1["label"])

scores = model_selection.cross_val_score(final_model, base_pred, train1["label"], cv=5, scoring="f1")
print('Cross Validation :', scores.mean())

# make prediction using our test data and model
y_pred = final_model.predict(base_pred)
print('')
print('###### SVC Classifier ######')

# evaluating the model
print("Testing Accuracy :", accuracy_score(train1["label"], y_pred))
print("Best Score :", final_model.best_score_)
print('F1-Score :', metrics.f1_score(train1["label"], y_pred, average = 'macro'))
print('Best Params :', final_model.best_params_)
print('Best Estimator', final_model.best_estimator_)
test_read = pd.read_excel('../input/fixed-bdc/Data BDC - Satria Data 2020/Data Uji/Data Uji BDC.xlsx')
final_pred = final_model.predict(test_pred)
submission_svc = pd.DataFrame()
submission_svc["ID"] = test_read["ID"]
submission_svc["prediksi"] = final_pred
submission_svc.to_csv("submission_stacking_proba_roberta.csv", index=False)
template = pd.read_csv('../input/groundtruth-bestlb-bdc/groundtruth.csv')
for aidi in template['ID']:
        template.loc[template['ID'] == aidi, 'prediksi'] = int(submission_svc.loc[submission_svc['ID'] == aidi]['prediksi'])
groundtruth = pd.read_csv('../input/groundtruth-bestlb-bdc/groundtruth.csv')


## CEK SCORE SUBMISSION
## MEMBANDINGKAN HASIL PREDIKSI DENGAN GROUNDTRUTH

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

y_true = groundtruth.prediksi
y_pred = template.prediksi

print(f'accuracy :',accuracy_score(y_true,y_pred))
print(f'f1_score_avg_none :',f1_score(y_true, y_pred, average=None))
print(f'f1_score_avg_macro :',f1_score(y_true, y_pred, average='macro'))
print(f'f1_score_avg_micro :',f1_score(y_true, y_pred, average='micro'))
print(f'f1_score_avg_weighted :',f1_score(y_true, y_pred, average='weighted'))


XGB_bow.fit(train_vectors, train["label"])
test_read = pd.read_excel('../input/fixed-bdc/Data BDC - Satria Data 2020/Data Uji/Data Uji BDC.xlsx')
sample_submission = pd.DataFrame()
sample_submission["ID"] = test_read["ID"]
sample_submission["prediksi"] = logreg_bow.predict(test_vectors) 
sample_submission.to_csv("submission_logreg.csv", index=False)
print(sample_submission.shape)
print(sample_submission.head())
# The Encoding function takes the text column from train or test dataframe, the tokenizer,
# and the maximum length of text string as input.

# Outputs:
# Tokens
# Pad masks - BERT learns by masking certain tokens in each sequence.
# Segment id

def bert_encode(texts, tokenizer, max_len = 512):
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
# Build and compile the model

def build_model(bert_layer, max_len = 512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
#Training data
train = pd.read_excel('../input/fixed-bdc/Data BDC - Satria Data 2020/Data Latih/Data Latih BDC.xlsx')
print('Training data shape: ', train.shape)
print(train.head())

# Testing data 
test = pd.read_excel('../input/fixed-bdc/Data BDC - Satria Data 2020/Data Uji/Data Uji BDC.xlsx')
print('Testing data shape: ', test.shape)
print(test.head())
train['text'] = train['judul'] + ' ' + train['narasi']
test['text'] = test['judul'] + ' ' + test['narasi']
train = train.drop(columns=['ID', 'tanggal', 'judul', 'narasi', 'nama file gambar'])
test = test.drop(columns=['ID', 'tanggal', 'judul', 'narasi', 'nama file gambar'])

print(train.head())
print(test.head())
# def decontracted(phrase):
#     # specific
#     phrase = re.sub(r"won\'t", "will not", phrase)
#     phrase = re.sub(r"can\'t", "can not", phrase)

#     # general
#     phrase = re.sub(r"n\'t", " not", phrase)
#     phrase = re.sub(r"\'re", " are", phrase)
#     phrase = re.sub(r"\'s", " is", phrase)
#     phrase = re.sub(r"\'d", " would", phrase)
#     phrase = re.sub(r"\'ll", " will", phrase)
#     phrase = re.sub(r"\'t", " not", phrase)
#     phrase = re.sub(r"\'ve", " have", phrase)
#     phrase = re.sub(r"\'m", " am", phrase)
#     return phrase
# import spacy
# import re
# nlp = spacy.load('en')
# def preprocessing(text):
#   text = text.replace('#','')
#   text = decontracted(text)
#   text = re.sub('\S*@\S*\s?','',text)
#   text = re.sub('http[s]?:(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)

#   token=[]
#   result=''
#   text = re.sub('[^A-z]', ' ',text.lower())
  
#   text = nlp(text)
#   for t in text:
#     if not t.is_stop and len(t)>2:  
#       token.append(t.lemma_)
#   result = ' '.join([i for i in token])

#   return result.strip()
# train.text = train.text.apply(lambda x : preprocessing(x))
# test.text = test.text.apply(lambda x : preprocessing(x))
#train.head()
# Download BERT architecture
# BERT-Large uncased: 24-layer, 1024-hidden-nodes, 16-attention-heads, 340M parameters

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.label.values
model = build_model(bert_layer, max_len=160)
model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=8
)
model.load_weights('model.h5')
test_pred = model.predict(test_input)
test_read = pd.read_excel('../input/fixed-bdc/Data BDC - Satria Data 2020/Data Uji/Data Uji BDC.xlsx')
submission = pd.DataFrame()
submission["ID"] = test_read["ID"]
submission['prediksi'] = test_pred.round().astype(int)
submission.to_csv("submission2.csv", index=False)