# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
import seaborn as sns
import re
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

import transformers #huggingface transformers library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from keras.layers.merge import add

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import unicodedata
import nltk
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
stopword_list = nltk.corpus.stopwords.words('english')
tokenizer = ToktokTokenizer()
nlp = spacy.load('en_core_web_sm', parse = False, tag=False, entity=False)
#from contractions import CONTRACTION_MAP
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
stopword_list = nltk.corpus.stopwords.words('english')
data = pd.read_json('/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json',lines=True)
# checking TPU
try:
    
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
# Data cleaning

# defining all functions
# Remove any emails 
def remove_emails(text):
    text = re.sub(r'\b[^\s]+@[^\s]+[.][^\s]+\b', ' ', text)
    return text

def remove_hyperlink(text):
    text=re.sub(r'(http|https)://[^\s]*',' ',text)
    return text

# Removing Digits
def remove_digits(text):
    #text= re.sub(r"\b\d+\b", "", text)
    text= re.sub(r"(\s\d+)", " ", text)
    return text
    

# Removing Special Characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z\s]', ' ', text)
    return text


# removing accented charactors
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

 # Removing Stopwords
def remove_stopwords(text,is_lower_case):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)   
    return filtered_text

# Lemmetization
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# Combine all the functions and creating a preprocessing pipeline
# # Text preprocessing
def text_preprocessing(corpus,isRemoveEmail,isRemoveDigits,isRemoveHyperLink, 
                     isRemoveSpecialCharac,isRemoveAccentChar,
                       text_lower_case,text_lemmatization, stopword_removal):
    
    normalized_corpus = []
    
    for doc in corpus:
        
        if text_lower_case:
            doc = doc.lower()
        
        if isRemoveEmail:
            doc = remove_emails(doc)
        
        if isRemoveHyperLink:
            doc=remove_hyperlink(doc)
             
        if isRemoveAccentChar:
            doc = remove_accented_chars(doc)
            
        if isRemoveDigits:
            doc = remove_digits(doc)
        
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        
        if text_lemmatization:
            doc = lemmatize_text(doc)
        
        if isRemoveSpecialCharac:
            doc = remove_special_characters(doc)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(doc,is_lower_case=text_lower_case)
                
        normalized_corpus.append(doc)
        
    return normalized_corpus

EMAIL_FLAG=True
DIGIT_FLAG=True
HYPER_LINK_FLAG=True
ALL_SPEC_CHAR_FLAG=True
ACCENT_CHAR_FLAG=True
LOWER_CASE_FLAG=True
LEMMETIZE_FLAG=False
STOPWORD_FLAG=True

clean_headline= text_preprocessing(data['headline'],EMAIL_FLAG,DIGIT_FLAG,HYPER_LINK_FLAG,
                   ALL_SPEC_CHAR_FLAG,ACCENT_CHAR_FLAG,
                  LOWER_CASE_FLAG,LEMMETIZE_FLAG,STOPWORD_FLAG)
clean_short_Desc = text_preprocessing(data['short_description'],EMAIL_FLAG,DIGIT_FLAG,HYPER_LINK_FLAG,
                   ALL_SPEC_CHAR_FLAG,ACCENT_CHAR_FLAG,
                  LOWER_CASE_FLAG,LEMMETIZE_FLAG,STOPWORD_FLAG)
data['clean_headline']=clean_headline
data['clean_short_Desc'] = clean_short_Desc

# Merging both the columns
data['MergedColumn'] = data[data.columns[6:8]].apply(
    lambda x: ' '.join(x.astype(str)),
    axis=1
)

df = data.copy()
del data
df.drop(columns=['headline', 'authors', 'link', 'short_description', 'date',
                   'clean_headline', 'clean_short_Desc'],axis=1,inplace=True)
print(f"The dataset contains { df.category.nunique() } unique categories")
#label encoding the categories. After this each category would be mapped to an integer.
encoder = LabelEncoder()
df['categoryEncoded'] = encoder.fit_transform(df['category'])
# Using hugging face tokenizer

def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])
#bert large uncased pretrained tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')
X_train,X_test ,y_train,y_test = train_test_split(df['MergedColumn'], df['categoryEncoded'], random_state = 2020, test_size = 0.3)
#tokenizing the news descriptions and converting the categories into one hot vectors using tf.keras.utils.to_categorical
Xtrain_encoded = regular_encode(X_train.astype('str'), tokenizer, maxlen=80)
ytrain_encoded = tf.keras.utils.to_categorical(y_train, num_classes=41,dtype = 'int32')
Xtest_encoded = regular_encode(X_test.astype('str'), tokenizer, maxlen=80)
ytest_encoded = tf.keras.utils.to_categorical(y_test, num_classes=41,dtype = 'int32')
def build_model(transformer, loss='categorical_crossentropy', max_len=512):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    #adding dropout layer
    x = tf.keras.layers.Dropout(0.3)(cls_token)
    #using a dense layer of 41 neurons as the number of unique categories is 41. 
    out = tf.keras.layers.Dense(41, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    #using categorical crossentropy as the loss as it is a multi-class classification problem
    model.compile(tf.keras.optimizers.Adam(lr=3e-5), loss=loss, metrics=['accuracy'])
    return model
#building the model on tpu
with strategy.scope():
    transformer_layer = transformers.TFAutoModel.from_pretrained('bert-large-uncased')
    model = build_model(transformer_layer, max_len=80)
model.summary()
#creating the training and testing dataset.
BATCH_SIZE = 32*strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE 
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((Xtrain_encoded, ytrain_encoded))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(Xtest_encoded)
    .batch(BATCH_SIZE)
)
#training for 10 epochs
n_steps = Xtrain_encoded.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    epochs=10
)
#making predictions
preds = model.predict(test_dataset,verbose = 1)
#converting the one hot vector output to a linear numpy array.
pred_classes = np.argmax(preds, axis = 1)
#extracting the classes from the label encoder
encoded_classes = encoder.classes_
#mapping the encoded output to actual categories
predicted_category = [encoded_classes[x] for x in pred_classes]
true_category = [encoded_classes[x] for x in y_test]
result_df = pd.DataFrame({'description':X_test,'true_category':true_category, 'predicted_category':predicted_category})
result_df.head()
print(f"Accuracy is {sklearn.metrics.accuracy_score(result_df['true_category'], result_df['predicted_category'])}")
result_df.to_csv('Predictions.csv', index = False)
result_df[result_df['true_category']!=result_df['predicted_category']]
confusion_mat = confusion_matrix(y_true = true_category, y_pred = predicted_category, labels=list(encoded_classes))
df_cm = pd.DataFrame(confusion_mat, index = list(encoded_classes),columns = list(encoded_classes))
plt.rcParams['figure.figsize'] = (20,20)
sns.heatmap(df_cm)
