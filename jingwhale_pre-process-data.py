# regression



import numpy as np

import pandas as pd

import re

import sys

from tqdm import tqdm

from keras.preprocessing.sequence import pad_sequences

import time

import itertools

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

import joblib

# Any results you write to the current directory are saved as output.



def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def config_GPU():

    GPU_list = get_available_gpus()

    if len(GPU_list)>0:

        from keras import backend as K

        import tensorflow as tf

        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()  

        config.gpu_options.allow_growth = True  

        config.gpu_options.per_process_gpu_memory_fraction = 0.99

        set_session(tf.Session(config=config)) 

        print('GPU configed')

    else:

        print('Do not found GPU')







class process_description(object):

    tqdm.pandas(desc="")

    def __init__(self,vec_dim = 128):

        self.max_len = 0

        self.vec_dim = vec_dim

    def process_description_file(self,file,max_len =  0.99):

        print('\nProcessing text data...')

        file = file.fillna('').str.lower()

        #file = file.progress_apply(emoji.demojize)  

        file = file.progress_apply(lambda x: self.process_string(x))

        text_len = file.apply(len)

        self.max_len = int(np.ceil(text_len.quantile(0.99)))

        print('Percentage of quantile of sentence lenghts: 50%: {} 75%: {} 90%: {} 95%: {} 99%: {}'.format(text_len.quantile(0.5),text_len.quantile(0.75),

                                                                     text_len.quantile(0.90),text_len.quantile(0.95),

                                                                     text_len.quantile(0.99)))

        print('The length of sentences: ',self.max_len)

        print('Data processed sucessfully.')

        return file

    def process_string(self,s):

        s = re.sub(r"(\W)", r" \1 ", s)

        s = re.sub('[0-9][0-9a-z]*', '<number>', s)

        s = re.sub(r'[" "]+', " ", s)

        s = s.rstrip().strip()

        s = s.split(' ')

        return s



    def padding_file(self,file,MAX_SEQUENCE_LENGTH = None):

        if MAX_SEQUENCE_LENGTH:

            max_len = MAX_SEQUENCE_LENGTH

        else:

            max_len = self.max_len

        file = pad_sequences(file,

            maxlen = max_len,

            dtype=int,

            padding = 'post',

            truncating = 'post',

            value = 0)

        return file.astype(np.int32)



    def token2vec(self,token):

        if token in self.word2index:

            return self.word2index[token]

        else:

            return 1

    def sent2vec(self,sentence):

        return np.array(list(map(self.token2vec,sentence)))

    def text_to_vect(self,file,dictionary):

        print('\nConverting sentences to vectors')

        self.word2index = dictionary

        res = file.copy()

        start_time = time.time()

        for i in tqdm(range(len(file))):

            res[i] = self.sent2vec(file[i])

        #sent_vector = np.array(sent_vector)

        print('Converted sucessfully. Time used:',time.time()-start_time)

        return res



def explore_string(unprocess_data,ind,desc_parser):

    start_time = time.time()

    sentence_length = []

    upper_counts =[]

    upper_pct = []

    puncts_count = []

    puncts_pct = []

    num_exclamation = []

    num_question = []

    num_token = []

    num_unk =[]

    pct_unk = []

    #desc_parser = process_description()

    for s in tqdm(unprocess_data):

        len_s = len(s)

        sentence_length.append(len_s)

        upper_counts.append(len([ c for c in s if c.isupper()]))

        upper_pct.append(len([ c for c in s if c.isupper()])/len_s)

        puncts_count.append(len(re.findall("\W",s)))

        puncts_pct.append(len(re.findall("\W",s))/len_s)

        num_exclamation.append(s.count('!'))

        num_question.append(s.count('?'))

        

        s_tokens = desc_parser.process_string(s)

        num_token.append(len(s_tokens))

        num_unk.append(len([t for t in s_tokens if t not in ind]))

        pct_unk.append(len([t for t in s_tokens if t not in ind])/len(s_tokens))

    print('mining text using: {} min'.format((time.time() - start_time)/60))

    numerical_data = pd.DataFrame({'string_length':sentence_length,

                        'upper_number':upper_counts,

                        'upper_percentage':upper_pct,

                        'puncts_number':puncts_count,

                        'puncts_percentage':puncts_pct,

                        '"!"_number':num_exclamation,

                        '"?"_number':num_question,

                        'words_number':num_token,

                        'outside_dictionary_number':num_unk,

                        'outside_dictionary_percentage':pct_unk})

    for col in numerical_data:

        if numerical_data[col].dtypes == np.int64:

            numerical_data[col] = numerical_data[col].astype(np.int32)

        elif numerical_data[col].dtypes == np.float64:

            numerical_data[col] = numerical_data[col].astype(np.float32)



    return numerical_data



def creating_embedding_weights(glove_embedding_index,ind):

    #Creating word index

    print(len(ind))

    # if the word not in the embedding, using <unk> instead. The vector of <unk> is the mean of all vectors, the <pad> keeps 0

    word_index = {'<pad>':0,'<unk>':1}

    for i,word in enumerate(ind.keys()):

        word_index[word] = i+2

    print("Number of unique tokens: ",len(word_index))

    

    #creating matrix

    EMBEDDING_DIM = glove_embedding_index['i'].shape[0]

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():

        embedding_vector = glove_embedding_index.get(word)

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector



    embedding_matrix[1] = embedding_matrix.mean(axis = 0)

    print(embedding_matrix.shape)

    return word_index,embedding_matrix.astype(np.float32)



def extract_useful_glove(tokens,glove_embedding_index):

    total_tokens = list(itertools.chain.from_iterable(tokens))

    unique_tokens = set(total_tokens)

    len_unique_tokens = len(unique_tokens)

    len_glove = len(glove_embedding_index)

    len_tokens = len(total_tokens)

    ood = {}

    ind = {}

    within = 0

    for token in tqdm(total_tokens):

        if token in  glove_embedding_index:

            within += 1

            if token in ind:

                ind[token] += 1

            else:

                ind[token] = 1



        else:

            if token in ood:

                ood[token] += 1

            else:

                ood[token] =1

    len_ood = len(ood)

    len_ind = len(ind)

    useless_glove = set(glove_embedding_index.keys()).difference(set(ind.keys()))

    print("{} of unique tokens were embedded.\n{} of text were embeded\n{} of Glove keys were usesless".format(

        len_ind/(len_unique_tokens),  within/len_tokens, len(useless_glove)/len_glove))

    

    ood = {k: v for k, v in sorted(ood.items(), key=lambda x: x[1],reverse = True)}

    return ood,ind



def orgnize_embedding_weights(file_path,text):

    print('Loading GloVe...')

    glove_embedding_index = {}

    f = open(file_path)

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        glove_embedding_index[word] = coefs

    f.close()

    print('Found %s word vectors.' % len(glove_embedding_index))

    

    ood,ind = extract_useful_glove(text,glove_embedding_index)

    word_index,embedding_matrix = creating_embedding_weights(glove_embedding_index,ind)

    joblib.dump(embedding_matrix.astype(np.float32),"embedding_matrix.joblib")

    return word_index,ind





def load_data(train_path,test_path,nrows):

    usecols = ['id','comment_text','target','male','female','homosexual_gay_or_lesbian','christian','jewish',

              'muslim','black','white','psychiatric_or_mental_illness']

    dtype = {'id':np.int32,'target':np.float32,'male':np.float32,'female':np.float32,

             'homosexual_gay_or_lesbian':np.float32,'christian':np.float32,'jewish':np.float32,

              'muslim':np.float32,'black':np.float32,'white':np.float32,'psychiatric_or_mental_illness':np.float32}

    train = pd.read_csv(train_path,usecols = usecols,nrows = nrows,dtype = dtype)

    test = pd.read_csv(test_path,usecols = ['id','comment_text'],nrows = nrows,dtype = {'id':np.int32})

    print("Train data shape: {}\tTest data shape: {}".format(train.shape,test.shape))

    train_0 = train[train.target == 0]

    train_non_0 = train[train.target != 0]

    del train

    # clean duplicates

    print('zero train data length: ',len(train_0))

    train_0 = train_0.drop_duplicates(subset = ['comment_text'])

    print('zero train data length after drop-duplicates: ',len(train_0))

    #train_0 = train_0.sample(frac=0.5, random_state=37)

    print('non-zero train data length: ',len(train_non_0))

    train_non_0 = train_non_0.drop_duplicates(subset = ['comment_text','target'])

    print('non-zero train data length after drop-duplicates: ',len(train_non_0))

    # merge text

    total_text = pd.concat([train_0,train_non_0,test],

                           ignore_index = True,

                          sort = False)

    print('train & test combine data shape: ',total_text.shape)

    return total_text

def split_data(df):

    numerical_cols = ['string_length','upper_number','upper_percentage','puncts_number',

                        'puncts_percentage', '"!"_number','"?"_number','words_number',

                        'outside_dictionary_number','outside_dictionary_percentage']

    train_total = df[~df.target.isna()]

    print('saving train data...')

    train_total.to_csv('train_processed.csv',index = False)

    print('train data saved.')

    train_total.info()

    print(train_total.shape)

    del train_total

    

    test_total = df[df.target.isna()][['id','comment_text'] + numerical_cols]

    print('saving test data...')

    test_total.to_csv('test_processed.csv',index = False)

    print('test data saved.')

    test_total.info()

    #submission = pd.DataFrame({'id': test_total['id'].values, 'prediction': 0})

    print(test_total.shape)

    #X_train_text = np.array(list(train_total['comment_text']))

    #X_test_text = np.array(list(test_total['comment_text']))

    

    #X_train_num = train_total[numerical_cols].values

    #X_test_num = test_total[numerical_cols].values

    

    #print('X_train_text shape: {}\tX_test_text shape: {}'.format(X_train_text.shape,X_test_text.shape))

    #print('X_train_num shape: {}\tX_test_num shape: {}'.format(X_train_num.shape,X_test_num.shape))

    

    #regression

    #y_train = train_total['target']

        #y_train = (train_total['target'] >= 0.5).astype(int)



    #print("y_train.shape: ",y_train.shape)

        #y_stat = y_train.value_counts()

        #y_stat = y_stat/y_stat.sum()

        #print('y percentage: ',dict(y_stat))

    

        #class_weight = dict(1-y_stat)

    

    #save data

    #joblib.dump(X_train_num,"X_train_num.joblib")

    #joblib.dump(X_test_num,"X_test_num.joblib")

    

    #joblib.dump(X_train_text,"X_train_text.joblib")

    #joblib.dump(X_test_text,"X_test_text.joblib") 

    

    #joblib.dump(y_train,"y_train.joblib")

    #return X_train.astype(np.float32),y_train.astype(np.float32),X_test.astype(np.float32),None,submission
def main():

    DEBUG = False

    if DEBUG:

        NROWS = 1000

        EPOCHS = 3

        MAX_SEQUENCE_LENGTH = 10

        glove_path = "../input/glove-twitter/glove.twitter.27B.25d.txt"

    else:

        NROWS = None

        EPOCHS = 10

        MAX_SEQUENCE_LENGTH = 100

        glove_path = "../input/glove-twitter-100d/glove.twitter.27B.100d.txt"

    KFOLD = 5

    EMBEDDING_DIM = 157

    time_dict = {}

    time_dict['Start_time'] = time.time()





    #loading data

    print('\n**Loading data...**')

    train_path = "../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv"

    test_path = "../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv"

    total_text = load_data(train_path,test_path,NROWS)

    #print(total_text.tail())

    time_dict['load_data'] = time.time()



    #creating weights

    print('\n**Creating embedding weights...**')

    desc_parser = process_description()

    total_text['processed_text'] = list(desc_parser.process_description_file(total_text['comment_text']))

    word_index,ind = orgnize_embedding_weights(glove_path ,total_text['processed_text'])

    time_dict['emb_weight'] = time.time()

    #print('matrix shape',embedding_matrix.shape)



    #numerical_df

    numerical_df = explore_string(total_text['comment_text'],ind,desc_parser)

    del ind





    #converting data

    print('\n**Converting data...**')

    total_text['processed_text'] = list(desc_parser.text_to_vect(total_text['processed_text'],word_index))

    del word_index

    total_text['processed_text'] = list(desc_parser.padding_file(total_text['processed_text'],MAX_SEQUENCE_LENGTH ))

    del desc_parser



    #concat text and numerical

    total_text = pd.concat([total_text,numerical_df],axis = 1, sort = False)

    del numerical_df

    #print(total_text.head())

    #X_train,y_train,X_test,class_weight,submission = split_data(total_text)

    split_data(total_text) 

    time_dict['convert_data'] = time.time()

    del total_text

    '''

    X_train_num = joblib.load("X_train_num.joblib")

    X_test_num = joblib.load("X_test_num.joblib")



    X_train_text = joblib.load("X_train_text.joblib")

    X_test_text = joblib.load("X_test_text.joblib") 

    y_train = joblib.load("y_train.joblib")



    train_total = pd.read_csv('train_processed.csv')

    test_total = pd.read_csv('test_processed.csv')

    '''

if __name__ == "__main__":

    main()