import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
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
EPOCHS = 2    
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
# setting up datasets directory
basedir = '../input/shopee-code-league-20/_DS_Sentiment_Analysis_'
train_df = pd.read_csv(os.path.join(basedir,'modified_sentiments','train (add leak).csv'))
original_df = train_df.copy()

# Credits to Tony NG's Shopee website's scraping dataset to improve model training/fitting
added_df = pd.read_csv(os.path.join(basedir,'shopee_review_scraped','shopee_reviews.csv'))

test_df = pd.read_csv(os.path.join(basedir,'modified_sentiments','test.csv'))

# Preprocessing
train_df.drop('review_id', axis=1, inplace=True)

print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)
review_train = train_df['review'].tolist()
review_test = test_df['review'].tolist()

print(len(set(review_train).intersection(set(review_test))))

same_data_list = list(set(review_train).intersection(set(review_test)))
same_data_list[0:5]
added_df = added_df.rename(columns={'label': 'rating','text':'review'})

added_df.iloc[1431262]
# Drop the trash data
added_df = added_df.drop(1431262)
# Use this to use both shopee data and scraped data
train_df = train_df.append(added_df,ignore_index = True)
# change type from string `object` to integer
train_df['rating'] = train_df['rating'].astype(int)
train_df['rating'].value_counts()
review_train = added_df['review'].tolist()
review_test = test_df['review'].tolist()
#Inspect data leak (after adding scraped data)
matched_reviews = set(review_train).intersection(set(review_test))
print('Matched reviews from scraped data and the test set:', len(matched_reviews))
import emoji
def emoji_cleaning(text):
    
    # Change emoji to text
    text = emoji.demojize(text).replace(":", " ")
    
    # Delete repeated emoji
    tokenizer = text.split()
    repeated_list = []
    
    for word in tokenizer:
        if word not in repeated_list:
            repeated_list.append(word)
    
    text = ' '.join(text for text in repeated_list)
    text = text.replace("_", " ").replace("-", " ")
    return text
have_emoji_train_idx = []
have_emoji_test_idx = []

for idx, review in enumerate(train_df['review']):
    if any(char in emoji.UNICODE_EMOJI for char in review):
        have_emoji_train_idx.append(idx)
        
for idx, review in enumerate(test_df['review']):
    if any(char in emoji.UNICODE_EMOJI for char in review):
        have_emoji_test_idx.append(idx)
train_emoji_percentage = round(len(have_emoji_train_idx) / train_df.shape[0] * 100, 2)
print(f'Train data has {len(have_emoji_train_idx)} rows that used emoji, that means {train_emoji_percentage} percent of the total')

test_emoji_percentage = round(len(have_emoji_test_idx) / test_df.shape[0] * 100, 2)
print(f'Test data has {len(have_emoji_test_idx)} rows that used emoji, that means {test_emoji_percentage} percent of the total')
train_df_original = train_df.copy()
test_df_original = test_df.copy()

# emoji_cleaning
train_df.loc[have_emoji_train_idx, 'review'] = train_df.loc[have_emoji_train_idx, 'review'].apply(emoji_cleaning)
test_df.loc[have_emoji_test_idx, 'review'] = test_df.loc[have_emoji_test_idx, 'review'].apply(emoji_cleaning)
# before cleaning
train_df_original.loc[have_emoji_train_idx, 'review'].tail()
# after cleaning
train_df.loc[have_emoji_train_idx, 'review'].tail()
import string
string.punctuation
# Prints the distribution of the train set
original_df['rating'].value_counts(normalize = True)
for punc in string.punctuation:
    print(punc)
    print(original_df[original_df['review'].str.contains(punc,regex=False)].rating.value_counts(normalize = True))
    print('------------------------------------------------------------')
import re
def review_cleaning(text):
    
    # delete lowercase and newline
    text = text.lower()
    text = re.sub(r'\n', '', text)
    text = re.sub('([.,!?()])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    
    # change emoticon to text
    text = re.sub(r':\(', 'dislike', text)
    text = re.sub(r': \(\(', 'dislike', text)
    text = re.sub(r':, \(', 'dislike', text)
    text = re.sub(r':\)', 'smile', text)
    text = re.sub(r';\)', 'smile', text)
    text = re.sub(r':\)\)\)', 'smile', text)
    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)
    text = re.sub(r'=\)\)\)\)', 'smile', text)
    
    # We decide to include punctuation in the model so we comment this line out!
    # text = re.sub('[^a-z0-9! ]', ' ', text)
    
    tokenizer = text.split()
    
    return ' '.join([text for text in tokenizer])
train_df['review'] = train_df['review'].apply(review_cleaning)
test_df['review'] = test_df['review'].apply(review_cleaning)
repeated_rows_train = []
repeated_rows_test = []

for idx, review in enumerate(train_df['review']):
    if re.match(r'\w*(\w)\1+', review):
        repeated_rows_train.append(idx)
        
for idx, review in enumerate(test_df['review']):
    if re.match(r'\w*(\w)\1+', review):
        repeated_rows_test.append(idx)
def delete_repeated_char(text):
    
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    
    return text
train_df.loc[repeated_rows_train, 'review'] = train_df.loc[repeated_rows_train, 'review'].apply(delete_repeated_char)
test_df.loc[repeated_rows_test, 'review'] = test_df.loc[repeated_rows_test, 'review'].apply(delete_repeated_char)
print('Before: ', train_df_original.loc[92129, 'review'])
print('After: ', train_df.loc[92129, 'review'])

print('\nBefore: ', train_df_original.loc[56938, 'review'])
print('After: ', train_df.loc[56938, 'review'])

print('\nBefore: ', train_df_original.loc[72677, 'review'])
print('After: ', train_df.loc[72677, 'review'])

print('\nBefore: ', train_df_original.loc[36558, 'review'])
print('After: ', train_df.loc[36558, 'review'])
def recover_shortened_words(text):
    
    # put \b (boundary) for avoid the characters in the word to be replaced
    # I only make a few examples here, you can add if you're interested :)
    
    text = re.sub(r'\bapaa\b', 'apa', text)
    
    text = re.sub(r'\bbsk\b', 'besok', text)
    text = re.sub(r'\bbrngnya\b', 'barangnya', text)
    text = re.sub(r'\bbrp\b', 'berapa', text)
    text = re.sub(r'\bbgt\b', 'banget', text)
    text = re.sub(r'\bbngt\b', 'banget', text)
    text = re.sub(r'\bgini\b', 'begini', text)
    text = re.sub(r'\bbrg\b', 'barang', text)
    
    text = re.sub(r'\bdtg\b', 'datang', text)
    text = re.sub(r'\bd\b', 'di', text)
    text = re.sub(r'\bsdh\b', 'sudah', text)
    text = re.sub(r'\bdri\b', 'dari', text)
    text = re.sub(r'\bdsni\b', 'disini', text)
    
    text = re.sub(r'\bgk\b', 'gak', text)
    
    text = re.sub(r'\bhrs\b', 'harus', text)
    
    text = re.sub(r'\bjd\b', 'jadi', text)
    text = re.sub(r'\bjg\b', 'juga', text)
    text = re.sub(r'\bjgn\b', 'jangan', text)
    
    text = re.sub(r'\blg\b', 'lagi', text)
    text = re.sub(r'\blgi\b', 'lagi', text)
    text = re.sub(r'\blbh\b', 'lebih', text)
    text = re.sub(r'\blbih\b', 'lebih', text)
    
    text = re.sub(r'\bmksh\b', 'makasih', text)
    text = re.sub(r'\bmna\b', 'mana', text)
    
    text = re.sub(r'\borg\b', 'orang', text)
    
    text = re.sub(r'\bpjg\b', 'panjang', text)
    
    text = re.sub(r'\bka\b', 'kakak', text)
    text = re.sub(r'\bkk\b', 'kakak', text)
    text = re.sub(r'\bklo\b', 'kalau', text)
    text = re.sub(r'\bkmrn\b', 'kemarin', text)
    text = re.sub(r'\bkmrin\b', 'kemarin', text)
    text = re.sub(r'\bknp\b', 'kenapa', text)
    text = re.sub(r'\bkcil\b', 'kecil', text)
    
    text = re.sub(r'\bgmn\b', 'gimana', text)
    text = re.sub(r'\bgmna\b', 'gimana', text)
    
    text = re.sub(r'\btp\b', 'tapi', text)
    text = re.sub(r'\btq\b', 'thanks', text)
    text = re.sub(r'\btks\b', 'thanks', text)
    text = re.sub(r'\btlg\b', 'tolong', text)
    text = re.sub(r'\bgk\b', 'tidak', text)
    text = re.sub(r'\bgak\b', 'tidak', text)
    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)
    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)
    text = re.sub(r'\bga\b', 'tidak', text)
    text = re.sub(r'\btgl\b', 'tanggal', text)
    text = re.sub(r'\btggl\b', 'tanggal', text)
    text = re.sub(r'\bgamau\b', 'tidak mau', text)
    
    text = re.sub(r'\bsy\b', 'saya', text)
    text = re.sub(r'\bsis\b', 'sister', text)
    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)
    text = re.sub(r'\bmdh2n\b', 'semoga', text)
    text = re.sub(r'\bsmoga\b', 'semoga', text)
    text = re.sub(r'\bsmpai\b', 'sampai', text)
    text = re.sub(r'\bnympe\b', 'sampai', text)
    text = re.sub(r'\bdah\b', 'sudah', text)
    
    text = re.sub(r'\bberkali2\b', 'repeated', text)
    
    text = re.sub(r'\byg\b', 'yang', text)
    
    return text
%%time
train_df['review'] = train_df['review'].apply(recover_shortened_words)
rating_mapper_encode = {1: 0,
                        2: 1,
                        3: 2,
                        4: 3,
                        5: 4}

# convert back to original rating after prediction later(dont forget!!)
rating_mapper_decode = {0: 1,
                        1: 2,
                        2: 3,
                        3: 4,
                        4: 5}

train_df['rating'] = train_df['rating'].map(rating_mapper_encode)
train_df['rating'].value_counts()
from sklearn.utils import resample
df_majority = train_df[train_df.rating==4]
df_other = train_df[train_df.rating!=4]

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=500000,     # to match minority class
                                 random_state=123) # reproducible results

train_df = pd.concat([df_majority_downsampled, df_other])
# zero,one,two,three,four = np.bincount(train_df['rating'])
# total = zero + one + two + three + four


# weight_for_0 = (1 / zero)*(total)/5 
# weight_for_1 = (1 / one)*(total)/5
# weight_for_2 = (1 / two)*(total)/5
# weight_for_3 = (1 / three)*(total)/5
# weight_for_4 = (1 / four)*(total)/5

# class_weight = {0: weight_for_0, 1: weight_for_1, 2:weight_for_2,3:weight_for_3,4:weight_for_4}
# class_weight
# Dropping some duplicates
train_df = train_df.drop_duplicates(subset ="review")
from tensorflow.keras.utils import to_categorical

# convert to one-hot-encoding-labels
train_labels = to_categorical(train_df['rating'], num_classes=5)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df['review'],
                                                  train_labels,
                                                  stratify=train_labels,
                                                  test_size=0.1,
                                                  random_state=2020)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
MODEL = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TFAutoModel.from_pretrained('bert-base-uncased')
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(train_df['review'].str.len())
MAX_LEN = 320
plt.show()
X_train_encode = regular_encode(X_train.values, tokenizer, maxlen=MAX_LEN)
X_val_encode = regular_encode(X_val.values, tokenizer, maxlen=MAX_LEN)
X_test_encode = regular_encode(test_df['review'].values, tokenizer, maxlen=MAX_LEN)
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train_encode, y_train))
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val_encode, y_val))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test_encode)
    .batch(BATCH_SIZE)
)
def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    sequence_output = tf.keras.layers.Dropout(0.2)(sequence_output)   
    cls_token = sequence_output[:, 0, :]
    out = Dense(5, activation='softmax')(cls_token) # 5 ratings to predict
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
n_steps = X_train.shape[0] // BATCH_SIZE

train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Get training and test loss histories
training_loss = train_history.history['loss']
test_loss = train_history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
pred = model.predict(test_dataset, verbose=1)
# Check if this works
# saving the temporary model as pickle

import pickle
with open('pred_bert.pkl','wb') as f:
    pickle.dump(pred, f)
pred_sentiment = np.argmax(pred,axis = 1)
import pickle
import os
bert_cased_model = os.path.join(basedir, 'bert', 'bert-cased-500k.pkl')
with open(bert_cased_model,'rb') as f:
    pred_bert500_cased = pickle.load(f)
    print(pred_bert500_cased.shape)
bert_uncased_model = os.path.join(basedir, 'bert', 'bert-uncased-500k.pkl')
with open(bert_uncased_model,'rb') as f:
    pred_bert500_uncased = pickle.load(f)
    print(pred_bert500_uncased.shape)
bert_320_model = os.path.join(basedir, 'bert', 'bert-based-500k-320length.pkl')
with open(bert_320_model,'rb') as f:
    pred_bert500_cased_len320 = pickle.load(f)
    print(pred_bert500_cased_len320.shape)
pred_bert = (pred_bert500_cased + pred_bert500_uncased + pred_bert500_cased_len320)/3
gpt_500_model = os.path.join(basedir, 'bert', 'gpt2_pred_500k_len250.pkl')
with open(gpt_500_model,'rb') as f:
    pred_gpt500 = pickle.load(f)
    print(pred_gpt500.shape)
gpt_400_model = os.path.join(basedir, 'roberta', 'gpt2_pred_4epoch_400.pkl')
with open(gpt_400_model,'rb') as f:
    pred_gpt = pickle.load(f)
    print(pred_gpt.shape)
pred_gpt = (pred_gpt500 + pred_gpt)/2
roberta_model = os.path.join(basedir, 'roberta', 'ROBERTA_pred_4epoch_subscore66.pkl')
with open(roberta_model,'rb') as f:
    pred_roberta = pickle.load(f)
    print(pred_roberta.shape)
pred = (pred_bert + pred_gpt + pred_roberta ) / 3
final_pred = []
confident_3_index = []
for idx,p in enumerate(pred):
    if np.argmax(p) == 2 and p[2]>0.438: 
#         print(idx)
        confident_3_index.append(idx)
        final_pred.append(2)  #because it's 0-4
    else:
        p[2] = 0
        final_pred.append(np.argmax(p))
final_pred
submission = pd.DataFrame({'review_id':[i+1 for i in range(60427)],'rating':final_pred})
submission['rating'].value_counts(normalize = True)
final_pred = []
confident_4_index = []
for idx,p in enumerate(pred):
    if np.argmax(p) == 3 and p[3]>0.34: 
#         print(idx)
        confident_3_index.append(idx)
        final_pred.append(3)  #because it's 0-4
    else:
        p[3] = 0
        final_pred.append(np.argmax(p))
final_pred
submission = pd.DataFrame({'review_id':[i+1 for i in range(60427)],'rating':final_pred})
submission['rating'].value_counts(normalize = True)
# [0.11388, 0.02350, 0.06051, 0.39692, 0.40519]
final_pred = []
for idx,p in enumerate(pred):
    if np.argmax(p) == 4 and p[4]>0.2: 
#         print(idx)
        final_pred.append(4)  #because it's 0-4
    else:
        p[4] = 0
        if p[0] > 0.14:
            final_pred.append(0)
        else:
            final_pred.append(np.argmax(p))
final_pred
submission = pd.DataFrame({'review_id':[i+1 for i in range(60427)],'rating':final_pred})
submission['rating'].value_counts(normalize = True)
# [0.11388, 0.02350, 0.06051, 0.39692, 0.40519]
rating_mapper_decode = {0: 1,
                        1: 2,
                        2: 3,
                        3: 4,
                        4: 5}

submission['rating'] = submission['rating'].map(rating_mapper_decode)

# 3-models-ensemble
submission.to_csv('submission.csv', index=False)
!head submission.csv
# Public test set distribution  : [0.11388, 0.02350, 0.06051, 0.39692, 0.40519]
submission.rating.value_counts(normalize = True)