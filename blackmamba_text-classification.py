import pandas as pd
import numpy as np
df = pd.read_csv("/kaggle/input/multiclass-text-classification/Merilytics_Clean.csv")
df.head(10)
def reduce_mem_usage(df: pd.DataFrame,
                     verbose: bool = True) -> pd.DataFrame:
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)

    return df

df_reduced = reduce_mem_usage(df)
#df_reduced.to_csv('Merilytics_reduced.csv')
print("\n {} rows of data available\n\n".format(len(df_reduced)))
print("\n {} No of unique reviwers \n\n".format(len(df_reduced['review_id'].unique())))
print("\n No of null's in all columns \n\n",df_reduced.isnull().sum())
print("\n Data per class of review rating \n\n",df_reduced['stars'].value_counts())
#Data Preperation
df_clean = df_reduced[df_reduced['review_id'].notnull() & df_reduced['text'].notnull()]
df_clean.isnull().sum()
reviews = np.array(df_clean['text'])
ratings = np.array(df_clean['stars'])

# build train and test datasets

train_len = int(0.7*len(df_clean))
print(f'Training Data {train_len} Testing Data {len(reviews)-train_len}')
train_reviews = reviews[:train_len]
train_rating = ratings[:train_len]
test_reviews = reviews[train_len:]
test_rating = ratings[train_len:]
del df
del df_clean
del df_reduced
import gensim
import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import ToktokTokenizer
from sklearn.preprocessing import LabelEncoder
tokenizer = ToktokTokenizer()
le = LabelEncoder()
num_classes=5
# tokenize train reviews & encode train labels
tokenized_train = [tokenizer.tokenize(text)
                   for text in train_reviews]
y_tr = le.fit_transform(train_rating)
y_train = keras.utils.to_categorical(y_tr, num_classes)
# tokenize test reviews & encode test labels
tokenized_test = [tokenizer.tokenize(text)
                   for text in test_reviews]
y_ts = le.fit_transform(test_rating)
y_test = keras.utils.to_categorical(y_ts, num_classes)
# print class label encoding map and encoded labels
print('Rating class label map:', dict(zip(le.classes_, le.transform(le.classes_))))
print('Sample test label transformation:\n'+'-'*35,
      '\nActual Labels:', test_rating[:3], '\nEncoded Labels:', y_ts[:3], 
      '\nOne hot encoded Labels:\n', y_test[:3])
w2v_num_features = 50
w2v_model = gensim.models.Word2Vec(tokenized_train, size=w2v_num_features, window=10,
                                   min_count=10, sample=1e-3)
def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)
# generate averaged word vector features from word2vec model
avg_wv_train_features = averaged_word2vec_vectorizer(corpus=tokenized_train, model=w2v_model,
                                                     num_features=50)
avg_wv_test_features = averaged_word2vec_vectorizer(corpus=tokenized_test, model=w2v_model,
                                                    num_features=50)
print('Word2Vec model:> Train features shape:', avg_wv_train_features.shape, ' Test features shape:', avg_wv_test_features.shape)
#print('GloVe model:> Train features shape:', train_glove_features.shape, ' Test features shape:', test_glove_features.shape)
def construct_deepnn_architecture(num_input_features):
    dnn_model = Sequential()
    dnn_model.add(Dense(512, activation='relu', input_shape=(num_input_features,)))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(512, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(512, activation='relu'))
    dnn_model.add(Dropout(0.2))
    dnn_model.add(Dense(5))
    dnn_model.add(Activation('softmax'))

    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',                 
                      metrics=['accuracy'])
    return dnn_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

w2v_dnn = construct_deepnn_architecture(num_input_features=50)
glove_dnn = construct_deepnn_architecture(num_input_features=50)

SVG(model_to_dot(w2v_dnn, show_shapes=True, show_layer_names=False, 
                 rankdir='TB').create(prog='dot', format='svg'))
from sklearn.metrics import classification_report

batch_size = 150
w2v_dnn.fit(avg_wv_train_features, y_train, epochs=5, batch_size=batch_size, 
            shuffle=True, validation_split=0.1, verbose=1)
y_pred = w2v_dnn.predict_classes(avg_wv_test_features)
predictions = le.inverse_transform(y_pred)

print(classification_report(test_rating,predictions))
#print(predictions)