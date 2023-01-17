#### IMPORTANT
# Due to the time limit on Kaggle Kernels, I have reduced the model complexity and data size. 
# If you wish to run this locally, set COMPLETE_RUN = True
COMPLETE_RUN = False
from keras import backend
import numpy as np
import pandas as pd

import os
import random
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(10001)
random.seed(10002)
tf.set_random_seed(10003)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=5)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
max_size = 150000
max_features = 100000
maxlen = 500
embed_size = 300
embedding_file = '../input/donorschooseorg-preprocessed-data/embeddings_small.vec'

if COMPLETE_RUN == False:
    max_size = 150
    max_features = 100
    maxlen = 5
    embedding_file = '../input/donorschooseorg-preprocessed-data/embeddings_small.vec'
base = "../input/donorschoose-application-screening/"
train = pd.read_csv(base + "train.csv")
test = pd.read_csv(base + "test.csv")
resources = pd.read_csv(base + "resources.csv")
train = train.sort_values(by="project_submitted_datetime")
if COMPLETE_RUN == False:
    train = train[:200]
    test = test[:200]
    resources = resources[:100]
teachers_train = list(set(train.teacher_id.values))
teachers_test = list(set(test.teacher_id.values))
inter = set(teachers_train).intersection(teachers_test)
char_cols = ['project_title', 'project_essay_1', 'project_essay_2',
             'project_essay_3', 'project_essay_4', 'project_resource_summary']
resources['total_price'] = resources.quantity * resources.price

mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean()) 
sum_total_price = pd.DataFrame(resources.groupby('id').total_price.sum()) 
count_total_price = pd.DataFrame(resources.groupby('id').total_price.count())
mean_total_price['id'] = mean_total_price.index
sum_total_price['id'] = mean_total_price.index
count_total_price['id'] = mean_total_price.index
def create_features(df):
    df = pd.merge(df, mean_total_price, on='id')
    df = pd.merge(df, sum_total_price, on='id')
    df = pd.merge(df, count_total_price, on='id')
    df['year'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[0])
    df['month'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[1])
    for col in char_cols:
        df[col] = df[col].fillna("NA")
    df['text'] = df.apply(lambda x: " ".join(x[col] for col in char_cols), axis=1)
    return df
train = create_features(train)
test = create_features(test)
cat_features = ["teacher_prefix", "school_state", "year", "month", "project_grade_category", 
                "project_subject_categories", "project_subject_subcategories"]
num_features = ["teacher_number_of_previously_posted_projects", "total_price_x", 
                "total_price_y", "total_price"]
cat_features_hash = [col+"_hash" for col in cat_features]
def feature_hash(df, max_size=max_size):
    for col in cat_features:
        df[col+"_hash"] = df[col].apply(lambda x: hash(x)%max_size)
    return df
train = feature_hash(train)
test = feature_hash(test)
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import text, sequence
import re

scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[num_features])
X_test_num = scaler.transform(test[num_features])
X_train_cat = np.array(train[cat_features_hash], dtype=np.int)
X_test_cat = np.array(test[cat_features_hash], dtype=np.int)
tokenizer = text.Tokenizer(num_words=max_features)
def preprocess(string):
    '''
    :param string:
    :return:
    '''
    string = re.sub(r'(\")', ' ', string)
    string = re.sub(r'(\r)', ' ', string)
    string = re.sub(r'(\n)', ' ', string)
    string = re.sub(r'(\r\n)', ' ', string)
    string = re.sub(r'(\\)', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub(r'\:', ' ', string)
    string = re.sub(r'\"\"\"\"', ' ', string)
    string = re.sub(r'_', ' ', string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub(r'\=', ' ', string)

    return string

train["text"]=train["text"].apply(preprocess)
test["text"]=test["text"].apply(preprocess)
tokenizer.fit_on_texts(train["text"].tolist()+test["text"].tolist())
list_tokenized_train = tokenizer.texts_to_sequences(train["text"].tolist())
list_tokenized_test = tokenizer.texts_to_sequences(test["text"].tolist())
X_train_words = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test_words = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

X_train_target = train.project_is_approved
embeddings_index = {}
with open(embedding_file, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, CuDNNGRU, Bidirectional, GRU, BatchNormalization
from keras.models import Model
from keras import optimizers
recurrent_units = 96
convolution_filters = 128
dense_units = [256, 128, 64]
dropout_rate = 0.3
learning_rate = 5e-3

if COMPLETE_RUN == False:
    recurrent_units = 8
    convolution_filters = 8
    dense_units = [8, 4, 4]
def get_model():
    input_cat = Input((len(cat_features_hash), ))
    input_num = Input((len(num_features), ))
    input_words = Input((maxlen, ))
    
    x_cat = Embedding(max_size, 20)(input_cat)
    x_cat = SpatialDropout1D(dropout_rate)(x_cat)
    x_cat = Flatten()(x_cat)
    
    x_words = Embedding(max_features, 300,
                            weights=[embedding_matrix],
                            trainable=False)(input_words)
    x_words = SpatialDropout1D(dropout_rate)(x_words)
    
    x_words1 = Bidirectional(GRU(recurrent_units, return_sequences=True))(x_words)
    x_words1 = Convolution1D(convolution_filters, 3, activation="relu")(x_words1)
    x_words1_1 = GlobalMaxPool1D()(x_words1)
    x_words1_2 = GlobalAveragePooling1D()(x_words1)
    
    x_words2 = Convolution1D(convolution_filters, 2, activation="relu")(x_words)
    x_words2 = Convolution1D(convolution_filters, 2, activation="relu")(x_words2)
    x_words2_1 = GlobalMaxPool1D()(x_words2)
    x_words2_2 = GlobalAveragePooling1D()(x_words2)
    
    x_num = input_num

    x = concatenate([x_words1_1, x_words1_2, x_words2_1, x_words2_2, x_cat, x_num])
    x = BatchNormalization()(x)
    x = Dense(dense_units[0], activation="relu")(x)
    x = Dense(dense_units[1], activation="relu")(x)
    
    x = concatenate([x, x_num])
    x = Dense(dense_units[2], activation="relu")(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(learning_rate, decay=1e-6),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model
model = get_model()
from keras.callbacks import *
from sklearn.metrics import roc_auc_score
file_path = 'best.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1,
                               epsilon=1e-4, mode='min')

callbacks_list = [checkpoint, early, lr_reduced]

history = model.fit([X_train_cat, X_train_num, X_train_words], X_train_target, validation_split=0.1,
                    verbose=1, callbacks=callbacks_list, epochs=5, batch_size=256)
model.load_weights(file_path)
pred_test = model.predict([X_test_cat, X_test_num, X_test_words], batch_size=1024, verbose=1)

test["project_is_approved"] = pred_test
if COMPLETE_RUN:
    test[['id', 'project_is_approved']].to_csv("submission.csv", index=False)
from scipy.special import expit, logit

LABELS = ["project_is_approved"]

base = "../input/the-all-in-one-model-prediction-files/"
predict_list = []
for j in range(10):
    predict_list.append(pd.read_csv(base + "submission_%d.csv"%j)[LABELS])
    
predcitions = np.zeros_like(predict_list[0])
for predict in predict_list:
    predcitions = np.add(predcitions, logit(predict)) 
predcitions /= len(predict_list)
predcitions = expit(predcitions)

submission = pd.read_csv('../input/donorschoose-application-screening/sample_submission.csv')
submission[LABELS] = predcitions
submission.to_csv('submission.csv', index=False)
import os
os.remove('best.h5')
