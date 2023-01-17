#load modules
import nltk #for tokenizing
import tqdm
#https://github.com/tqdm/tqdm 第三方進度條模組
import numpy as np
import pandas as pd
#PunktSentenceTokenizer
nltk.download("punkt")
import zipfile
import gc
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input/glove6b300dtxt"))
os.listdir("../input/donorschoose-application-screening")
#load data
df_train = pd.read_csv('../input/donorschoose-application-screening/train.csv')
df_test = pd.read_csv('../input/donorschoose-application-screening/test.csv')
df_resources = pd.read_csv('../input/donorschoose-application-screening/resources.csv')
#load word embedded vectors
import os
embedding_path = "../input/glove6b300dtxt/glove.6B.300d.txt"

def read_embedding_vec(embedding_file_path):
    embedding_list = []
    embedding_word_dict = dict()
    
    f = open(embedding_file_path)

    for index, line in enumerate(f):
        #the line contains target word and its word vector, so split it first
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        #save the word vec into a list
        embedding_list.append(coefs)
        #build a connection between word and its word id (0, 1, 2, ......)
        embedding_word_dict[word] = len(embedding_word_dict)
    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict
#load embedding word vectors
embedding_list, embedding_word_dict = read_embedding_vec(embedding_path)
#embedding_list[:5], embedding_word_dict['the']
#set word_vector for UNKNOWN_WORD and _NAN_
try:
    embedding_word_dict['nanword']
except:
    embedding_word_dict['nanword'] = len(embedding_word_dict)
    embedding_list = np.append(embedding_list, [np.zeros_like(embedding_list[0,:])], axis = 0)
try:
    embedding_word_dict['unknownword']
except:
    embedding_word_dict['unknownword'] = len(embedding_word_dict)
    embedding_list = np.append(embedding_list, [np.zeros_like(embedding_list[0,:])], axis = 0)
#(embedding_word_dict['nanword'], embedding_word_dict['unknownword'], len(embedding_word_dict)), embedding_list[-3:]
import re
def preprocess(string):
    '''
    :param string:
    :return:
    '''
    string = string.lower()
    string = re.sub(r'(\\")', '', string)
    string = re.sub(r'(\\r\\n)', ' ', string)
    string = re.sub(r'(\\r)', ' ', string)
    string = re.sub(r'(\\n)', ' ', string)
    string = re.sub(r'(\\)', ' ', string)
    string = re.sub(r'\\t', ' ', string)
    string = re.sub(r'\:', ' ', string)
    string = re.sub(r'\"\"\"\"', ' ', string)
    string = re.sub(r'_', ' ', string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub(r'\=', ' ', string)
    string = re.sub(r'\-', ' ', string)
    #sep numstring to "num string"
    string = " ".join(re.split(r'(\d+)', string))
    string = string.strip()
    return string
words_dict = dict()
#for sentences checking
#for i in range(20,50):
#    print(df_train['project_title'][i])
#tokenize_sentences(df_train['project_title'], words_dict, early_stopping = False)
#combine the sentences
def combine_text(df):
    try:
        if len(df['text']) == len(df): return df
    except: pass
    df['text'] = ""
    text_col  = ['project_title', 'project_essay_1', 
                 'project_essay_2', 'project_essay_3', 
                 'project_essay_4', 'project_resource_summary']
        
    #find nan values and apply fillna
    df[text_col] = df[text_col].fillna('NANWORD')
    df['text'] = df.apply(lambda x: " ".join([str(x[col]).strip() for col in text_col]), axis=1)
    #df.drop(['project_title', 'project_essay_1', 
    #         'project_essay_2', 'project_essay_3', 
    #         'project_essay_4', 'project_resource_summary'], axis = 1, inplace = True)
    return df
#df_train.drop('text', axis = 1, inplace = True)
df_train = combine_text(df_train)
df_train.head()
#print(df_train.iloc[0, -3])
#print(df_train.iloc[0, -2])
def tokenize_sentences(sentences, words_dict, early_stopping = False):
    '''
    read sentences from the dataset and return tokenized_sentences and local words_dict
    early_stopping is set to run small dataset to check how the tokenization goes
    '''
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        #check attribute first
        if hasattr(sentence, "decode"): 
            sentence = sentence.decode("utf-8")
        #run preprocessing
        sentence = preprocess(sentence)
        #start tokenizing
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
        if early_stopping:
            print(sentence)
            if len(tokenized_sentences) == 50:
                return tokenized_sentences, words_dict
    return tokenized_sentences, words_dict
#start tokenization 
df_train['tokenized_text'], words_dict = tokenize_sentences(df_train['text'], words_dict)
#count the sentence length for each project
df_train["tokenized_text_length"] = df_train['tokenized_text'].apply(lambda x: len(x))
plt.hist(df_train[df_train["project_is_approved"] == True]["tokenized_text_length"], 
         histtype = 'bar', bins=50, range = (150, 650),
         label = "Approved",
         color = 'red', stacked = True)
plt.hist(df_train[df_train["project_is_approved"] == False]["tokenized_text_length"], 
         histtype = 'bar', bins=50, range = (150, 650),
         label = "Not approved",
         color = 'green', stacked = True)
plt.legend()
len(df_train[(df_train["project_is_approved"] == True) & (df_train["tokenized_text_length"]>= 630)])
def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    '''
    The dataset might not use all the pre-trained word vectors, we only need the words included in the dataset.
    '''
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict
words_dict['nanword']
def convert_tokens_to_embedded(tokenized_sentences, words_to_ids, embedding_word_dict, 
                          sentences_length = 630, padding = embedding_word_dict['nanword'], 
                               early_stopping = False):
    '''
    tokenized_sentences and embedding_word_dict have different word ids, this function would try to look them up
    Input: local tokenized sentence
    Output: Lookup embedded tokenized sentence
    '''
    words_train = []
    #loop all the sentences of the local dataset
    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_to_ids[word_index]
            #why -1? -1 to get index and len(embedding_word_dict) = 400001, 
            #cleared_embedding_word_dict['unknownword'], cleared_embedding_word_dict['nanword']
            #look up to embedding_word_dict to find the word index
            word_id = embedding_word_dict.get(word, embedding_word_dict['unknownword'])
            current_words.append(word_id)
        #set maximum for sentence length
        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            #tokenized_sentences length is less than the setup sentence length => add NANWORD
            current_words += [padding] * (sentences_length - len(current_words))
        if early_stopping and len(words_train) >= 10:
            break
        words_train.append(current_words)
    return words_train
words_to_ids = dict()
for key, value in words_dict.items():
    words_to_ids[value] = key
train_x_words = np.array(convert_tokens_to_embedded(df_train['tokenized_text'], 
                words_to_ids, embedding_word_dict, 
                sentences_length = 630))
#print(' '.join(words_to_ids[word] for word in df_train['tokenized_text'][0]))
#df_train.drop('lookup_tokenzied_sentence', axis = 1, inplace = True)
df_train['tokenized_text'].head()
#check code
#" ".join([embedding_ids_dict[word_idx] for word_idx in convert_tokens_to_ids(df_train['tokenized_text'], 
#                                      words_to_ids, embedding_word_dict, 
#                                      early_stopping = True)[0]])
#Looks pretty good!
#df_train.head()
#drop some unused columns to release memory
df_train.drop(['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
              'project_resource_summary', 'text'], axis = 1, inplace = True)
gc.collect()
#note that price is only for one quantity of the resource
df_resources['total_price'] = df_resources['quantity'] * df_resources['price']
sum_total_price = pd.DataFrame(df_resources.groupby('id').total_price.sum()).reset_index()
#sum_total_price.head()
#append resources items to train and test
df_train = pd.merge(df_train, sum_total_price, on='id')
df_test = pd.merge(df_test, sum_total_price, on = 'id')
train_x_num = np.array(df_train[['total_price', 'teacher_number_of_previously_posted_projects', 'tokenized_text_length']])
#import scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_x_num = scaler.fit_transform(train_x_num)
#import modules from keras
from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
                         GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, CuDNNGRU, Dropout,\
                         Bidirectional, GRU, BatchNormalization, AveragePooling1D, MaxPooling1D
from keras.models import Model, Sequential
from keras import optimizers
def get_model(embedding_matrix = embedding_list, num_features = 3,
              maxlen = 630, dropout_rate = 0.4, learning_rate = 5e-3,
              recurrent_units = 64):
    '''
    embedding_matrix needs to be an array
    '''
    input_words = Input((maxlen, ))
    
    input_nums = Input((num_features, ))
    #set dim for embedded word
    embedding_dim = 300
    max_features = len(embedding_matrix)
    
    x_words = Embedding(max_features, embedding_dim,
                        weights = [embedding_matrix], 
                        input_length = maxlen,
                        trainable=False)(input_words)
    x_words = SpatialDropout1D(dropout_rate)(x_words)
    
    x_words1 = Bidirectional(GRU(recurrent_units, return_sequences=True))(x_words)
    #set filters to be 128, kernel_size to be 5
    x_words1 = Convolution1D(64, 5, activation="relu")(x_words1)
    x_words1_1 = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x_words1)
    x_words1_1 = GlobalAveragePooling1D()(x_words1)
    
    #Do we really need the CNN model to make the prediction better?
    #x_words2 = Convolution1D(32, 5, activation="relu")(x_words)
    #x_words2 = MaxPooling1D(2,padding='same')(x_words2)
    #x_words2 = Convolution1D(64, 5, activation="relu")(x_words2)
    #x_words2_1 = MaxPooling1D(2,padding='same')(x_words2)
    #x_words2_1 = GlobalAveragePooling1D()(x_words2_1)
    
    #x = concatenate([x_words1_1, x_words2_1])
    #print(x.shape)
    
    #try if only GRU can output a great score
    x = BatchNormalization()(x_words1_1)
    
    dense_units = [64, 16] #Ver8 & Ver9
    #dense_units = [32, 8] #Ver11
    #x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units[0], activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units[1], activation="relu")(x)
    
    #join x_nums
    x_nums = input_nums
    #since x might have some gradient vanishing situations already, I would like to do batchnorm again before concatenate with x_nums
    x = BatchNormalization()(x)
    x = concatenate([x, x_nums])
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_words, input_nums], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(learning_rate, decay=1e-6),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model
#del model
#del get_model
#gc.collect()
model = get_model()
from keras.callbacks import *
from sklearn.metrics import roc_auc_score
file_path = 'best.h5'
#test = np.array(train_x_words)
#test.shape
#train_x_words.shape
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1,
                               epsilon=1e-4, mode='min')

callbacks_list = [checkpoint, early, lr_reduced]

history = model.fit([train_x_words, train_x_num], 
                    df_train['project_is_approved'], validation_split=0.03,
                    verbose=1, callbacks=callbacks_list, epochs=10, batch_size=256)
#del data only for training to release memory
del df_train
#del cleared_embedding_word_dict
#del cleared_embedding_list
del train_x_words
gc.collect()
# test data preprocessing
df_test = combine_text(df_test)
# drop other text columns directly
df_test.drop(['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
             'project_resource_summary'], axis = 1, inplace = True)
gc.collect()
# test text tokenization
df_test['tokenized_text'], words_dict = tokenize_sentences(df_test['text'], words_dict)
df_test["tokenized_text_length"] = df_test['tokenized_text'].apply(lambda x: len(x))
# test_x_num
test_x_num = np.array(df_test[['total_price', 'teacher_number_of_previously_posted_projects', 'tokenized_text_length']])
test_x_num = scaler.transform(test_x_num)
#words_to_ids = dict()
#renew words_to_ids
for key, value in words_dict.items():
    words_to_ids[value] = key
#test tokenized words vector lookup
test_x_words = np.array(convert_tokens_to_embedded(df_test['tokenized_text'], 
                words_to_ids, embedding_word_dict, 
                padding = embedding_word_dict['nanword'],
                sentences_length = 630))
#run prediction
model.load_weights(file_path)
pred_test = model.predict([test_x_words, test_x_num], batch_size=1024, verbose=1)

df_test["project_is_approved"] = pred_test
#submit our prediction!
df_test[['id', 'project_is_approved']].to_csv("submission.csv", index=False)
