# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#!/usr/bin/env python

# -*- coding: utf-8 -*-



"""

Catégorisation par apprentissage en Traitement Automatique des Langues (TALN)



notes things to change/add: 

-add model : multi channel cnn

_get the best units for lstm

_tunning



"""

#imports : 

import pandas as pd 

import numpy as np

from nltk.tokenize import word_tokenize

from tqdm import tqdm

import re

import sys

from gensim.models import FastText

import tensorflow as tf

from keras import backend as K

from sklearn.model_selection import train_test_split

from keras.activations import relu

from keras.layers import concatenate,  merge

from keras.models import Model

from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D

from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D

from keras.layers import GRU, LSTM, Bidirectional, Activation

from keras.layers import Activation, Flatten, Lambda

from keras.layers import GlobalAveragePooling1D, MaxPooling1D,GlobalMaxPool1D

from keras.preprocessing import text, sequence

from keras.callbacks import Callback

from keras import optimizers

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, callbacks

from keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import AveragePooling1D







#initializing

max_features = 2000 # 5gram = 5*len(dataset)

emb_size = 300

#create dict of {word:vect}

fastText_FILE = '../input/cc.fr.300.vec'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

embeddings_coeff= dict(get_coefs(*o.rstrip().rsplit(' ')) 

                for o in tqdm(open(fastText_FILE, encoding="utf8"),  " vect word"))





class NlpCorpus(object): 

    """

    Preprocessing the data

    """

    

    def __init__(self, csv_fname):

        self._fname = csv_fname

         #load the dataset into a dataframe and add column names(header)

        self._dataframe = pd.read_csv(self._fname, sep =";", names = ["Labels", "Corpus" , "Pivot"])

        

    #*****************************************************************

    def create_ngrams(self, number):

        #exception for param number : if number = 1 : 1-gram => dataframe["Pivot"]

        try:

            assert number > 2

            

            #add a column for ngrams :

            self._dataframe['%s_grams_pivot'%number] = 0

            #rows to drop:

            drop_list = []

            length_first = len(self._dataframe)

            

            #split the corpus into words and enumerate to keep the index of the pivot :

            for idx, sentence in tqdm(enumerate(self._dataframe["Corpus"]), "Read Corpus"):  # for each sentence

                #strip :

                #phrases = re.sub("[\(\)\{\}\[\]\:\-]", ' ', sentence)

                list_tokens = word_tokenize(sentence)  #either use this or split found that word_tokenize is better

                # find the index of pivot word

                for index, word in enumerate(list_tokens):

                    #check if pivot is one word ou multiple words

                    if word == self._dataframe["Pivot"][idx] and len(

                        (self._dataframe["Pivot"][idx]).split()) == 1 and list_tokens[

                            index+1] == ']':

                        index_pivot = index

                        case = "Normal"

                        length_pivot = 1

                    else:    

                        if word == self._dataframe["Pivot"][idx].split()[0] and list_tokens[

                            index-1] == '[': 

                            length_pivot = len((self._dataframe["Pivot"][idx]).split())

                            need = number-length_pivot 

                            

                            if  length_pivot > number: 

                                sys.stderr.write("""Length of Pivot > ngram """)

                                drop_list.append(idx)

                                

                            if length_pivot == number:

                                case = "Gram is Pivot"

                            else: 

                                index_pivot = index

                                case = "Compound"

                

                #create list gram depending on the wereabouts of pivot(first/last)                

                if index_pivot == 0 :

                    list_final = list_tokens[index_pivot:number]

                    

                if (len(list_tokens)-1) == index_pivot + length_pivot -1 :

                    list_final = list_tokens[len(list_tokens)-number:len(list_tokens)] 

 

                if case == 'Normal' and index_pivot != 0 and (len(list_tokens)-1) != index_pivot:

                    if (number-1)%2 == 0:

                        need_each = int((number-1)/2)

                        list_final = list_tokens[(index_pivot-need_each) : (index_pivot +need_each)+1]

                        

                        if len(list_final) != number: 

                            #check  list

                            if len(list_tokens[index_pivot:])-1 < need_each :

                                complete = need_each - (len(list_tokens[index_pivot:])-1)

                                list_final = list_tokens[index_pivot-need_each-complete :]

                                

                            else : 

                                complete = need_each - (len(list_tokens[:index_pivot])-1)-1

                                list_final = list_tokens[:index_pivot +need_each +1+complete]

                        

                    if (number-1)%2 == 1:

                        need = number-1

                        #check if i can take just from one side

                        if len(list_tokens[index_pivot:]) >= number:

                            list_final = list_tokens[index_pivot: index_pivot+ number] 

                            

                        else:

                            list_final = list_tokens[index_pivot-(need): index_pivot+1]

                            

                if case == 'Compound' and index_pivot != 0 and (len(list_tokens)-1) != index_pivot + length_pivot -1:

                    if need%2 == 1 :

                        if len(list_tokens[index_pivot+length_pivot:]) >= need:

                            list_final = list_tokens[index_pivot : index_pivot +(length_pivot)+need]

                            

                        else : 

                            list_final = list_tokens[index_pivot-1:]

                            

                    if need%2 == 0:

                        need_each = int(need/2)

                        list_final = list_tokens[index_pivot-need_each : index_pivot +(length_pivot)+need_each]

                        

                        if len(list_final) != number : 

                            if len(list_tokens[index_pivot+length_pivot:]) < need_each:

                                complete = need_each -len(list_tokens[index_pivot+length_pivot:])

                                list_final = list_tokens[index_pivot-(length_pivot)-complete+1:]

                                

                            else : 

                                complete = need_each -len(list_tokens[:index_pivot])

                                list_final = list_tokens[:index_pivot+length_pivot+complete+need_each]

       

                if case == "Gram is Pivot"  and index_pivot != 0 and (len(list_tokens)-1) != index_pivot +length_pivot-1:

                    list_final = self._dataframe["Pivot"][idx].split()

                

                #exception :

                if len(list_final) != number:

                    sys.stderr.write(""" error in creating n-grams :  for sentence number %s  length : %d instead of %d """ % (idx, 

                                        len(list_final) ,

                                        number))

                    break

                #put the data in the right column

                self._dataframe['%s_grams_pivot'%number][idx] = list_final

                

                

            #cleaning rows:

            if len(drop_list) != 0:

                self._dataframe = self._dataframe.drop(drop_list, axis=0)

                print("axis dropped are =", drop_list)

            

        except ValueError:

            print("should be int")

            

        except AssertionError:

            print("number should be > 2")

        

        print("len dataframe now", len(self._dataframe))

        print("len dataframe at first", length_first)

        return self

         

    #**********************************************************************************

    def matrix_emb(self,number):

        for idx, list_vect in enumerate(self._dataframe['%s_word_vector'%number].values):

            if idx == 0:

                matrix = list_vect

            else :

                matrix = np.vstack((matrix,list_vect)) #shape(567,100)

            

                    

        return matrix

                

    #***********************/*********************************************************    

    def create_nvectors(self,number=1):

        if number == 1 :

            name = 'Pivot'

            self._dataframe['%s_word_vector'%number]=0 

            self._dataframe['%s_word_vector'%number] =self._dataframe['%s_word_vector'%number].astype(object)

          

            #the use of fastext

            sequence = [[word] for word in self._dataframe[name]]

            fastext_model= FastText(sequence , min_count=1)

            self._dataframe['%s_word_vector'%number] = self._dataframe[name].apply(lambda x : fastext_model.wv.get_vector(x) )

                

        else : 

            name = '%s_grams_pivot'%number

            self._dataframe['%s_word_vector'%number]=0 

            self._dataframe['%s_word_vector'%number] =self._dataframe['%s_word_vector'%number].astype(object)

            #the use of fastext

            for index,list_gram in enumerate(self._dataframe[name]):

                

                sequence = [[word] for word in list_gram]

                fastext_model= FastText(sequence , min_count=1)

                word_vect = np.empty((0))

                for idx,word in enumerate(list_gram):

                    word_vect = np.append(word_vect,fastext_model.wv.get_vector(word))

                self._dataframe['%s_word_vector'%number][index] = word_vect

                

        self.matrix = self.matrix_emb(number)   

        return self

        

    #***************************************************************************************************    

        

    def create_nvectors2(self,number=1):

        list_drop = []

        if number == 1 :

            name = 'Pivot'

            self._dataframe['%s_word_vector'%number]=0 

            self._dataframe['%s_word_vector'%number] =self._dataframe['%s_word_vector'%number].astype(object)

          

            #the use of fastext

            self._dataframe['%s_word_vector'%number] = self._dataframe[name].apply(lambda x : embeddings_coeff.get(x) )

                

        else : 

            name = '%s_grams_pivot'%number

            self._dataframe['%s_word_vector'%number]=0 

            self._dataframe['%s_word_vector'%number] =self._dataframe['%s_word_vector'%number].astype(object)

            #the use of fastext

            for index,list_gram in enumerate(self._dataframe[name]):

                word_vect = np.empty((0))

                for word in (list_gram):

                    vector = embeddings_coeff.get(word)

                    if vector is None:

                        list_word = word.split("'")

                        print("list",list_word)

                        word_1 = max(list_word[0],list_word[1])

                        vector_1 = embeddings_coeff.get(word_1)

                        if vector_1 is None:

                            word_2 = min(list_word[0],list_word[1])

                            vector_2 = embeddings_coeff.get(word_2)

                            if vector_2 is None:

                                print("can't seem to create a vect with this word =", word)

                                list_drop.append(index)

                            else:

                                word_vect = np.append(word_vect,vector_2)

                        else:

                            word_vect = np.append(word_vect,vector_1)

                    else:

                        word_vect = np.append(word_vect,vector)

                    

                self._dataframe['%s_word_vector'%number][index]= word_vect

                

            #cleaning rows:

            if len(list_drop) != 0:

                self._dataframe = self._dataframe.drop(list_drop, axis=0)

                print("axis dropped are =", list_drop)

            

        self.matrix = self.matrix_emb(number)

        return self

############################################################################################     

class RNN(object):  

    """

    RNN : GRU /

            LSTM

            for sentences classification.

    inspired from https://github.com/yhuangbl/toxic_comments

    

    """

    def __init__(self, embed_size ,embedding_matrix):

        self.embed_size = embed_size

        self.embedding_matrix = embedding_matrix 



    #***********************************************************************************

    def build_model_GRU(self, maxlen, model_num, tunning = False):

        if model_num == 1 :

            train = False

        else : 

            train = True

            

        comment = Input(shape=(maxlen,))

        emb_comment = Embedding(max_features,self.embed_size, weights=[self.embedding_matrix]

                                        , trainable=False)

        emb_comment.build((None,)) 

        embedded = emb_comment(comment)

        layer = GRU(30)(embedded)

        outp = Dense(2, activation="sigmoid")(layer)

        model = Model(inputs=comment, outputs=outp)

        model.summary()

        if tunning :

            model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(),metrics=["accuracy"])

        return model

    #*************************************************************************************************

    def build_model_LSTM(self,maxlen, model_num, tunning= False):

        if model_num == 1:

            comment = Input(shape=(maxlen,))

            emb_comment = Embedding(max_features,self.embed_size, weights=[self.embedding_matrix]

                                        , trainable=True)

            emb_comment.build((None,)) 

            embedded = emb_comment(comment)

            layer = LSTM(300)(embedded)

            layer = Dense(256)(layer)

            layer = Activation('relu')(layer)

            layer = Dropout(0.5)(layer)

            layer = Dense(2, activation="sigmoid")(layer)

            model = Model(inputs=comment,outputs=layer)

        else : 

            model = Sequential()

            model.add(Embedding(max_features, self.embed_size, input_length=maxlen , trainable = False))

            model.add(LSTM(30, recurrent_dropout=0.3))

            model.add(Dense(2, activation='sigmoid'))

        model.summary()

        if tunning :

            model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(),metrics=["accuracy"])

        return model



############################################################################################     

        

class DPCNN(object):  

    """

    DPCNN for sentences classification.

    inspired from https://www.kaggle.com/michaelsnell/conv1d-dpcnn-in-keras#L55

    """

    def __init__(self, embed_size ,embedding_matrix):

        self.embed_size = embed_size

        self.embedding_matrix = embedding_matrix

    #*********************************************************************************    

    def build_model(self, maxlen, model_num, tunning=False):

        filter_nr = 64

        filter_size = 3

        kernel_size = 3

        max_pool_size = 3

        max_pool_strides = 2

        dense_nr = 256

        train_embed = False

        spatial_dropout = 0.2

        dropout=0.2

        conv_kern_reg = regularizers.l2(0.00001)

        conv_bias_reg = regularizers.l2(0.00001)

        

        if model_num == 1: 

            comment = Input(shape=(maxlen,))

            emb_comment = Embedding(max_features,self.embed_size, weights=[self.embedding_matrix]

                                    , trainable=train_embed)

            emb_comment.build((None,)) 

            embedded = emb_comment(comment)

            emb_comment = SpatialDropout1D(spatial_dropout)(embedded)

            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 

                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)( emb_comment)

            block1 = BatchNormalization()(block1)

            block1 = PReLU()(block1)

            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 

                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)

            block1 = BatchNormalization()(block1)

            block1 = PReLU()(block1)

            #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output

            #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output

            resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', 

                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)( emb_comment)

            resize_emb = PReLU()(resize_emb)

            block1_output = add([block1, resize_emb])

            block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

            block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 

                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)

            block2 = BatchNormalization()(block2)

            block2 = PReLU()(block2)

            block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 

                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)

            block2 = BatchNormalization()(block2)

            block2 = PReLU()(block2)

            block2_output = add([block2, block1_output])

            output = GlobalMaxPooling1D()(block2_output)

            output = Dense(dense_nr, activation='linear')(block1_output)

            output = BatchNormalization()(output)

            output = PReLU()(output)

            output = Flatten()(output)

            output = Dense(2, activation='sigmoid')(output)

            model = Model(comment, output)

            

        if model_num == 2:

            comment = Input(shape=(maxlen,))

            emb_comment = Embedding(max_features,self.embed_size, weights=[self.embedding_matrix])

            emb_comment.build((None,)) 

            embedded = emb_comment(comment)

            emb_comment = SpatialDropout1D(spatial_dropout)(embedded)

            embedding = PReLU()(emb_comment)  # pre activation

            block1 = Conv1D(filter_nr, kernel_size, padding='same',

                            kernel_regularizer=conv_kern_reg)(embedding)

            block1 = PReLU()(block1)

            block1 = Conv1D(filter_nr, kernel_size, padding='same',

                            kernel_regularizer=conv_kern_reg)(block1)

            # reshape layer if needed

            conc1 = None

            if filter_nr != self.embed_size:

                embedding_resize = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',

                                        kernel_regularizer=conv_kern_reg)(embedding)

                block1 = Lambda(relu)(block1)

                conc1 = add([embedding_resize, block1])

            else:

                conc1 = add([embedding, block1])

            # block 2 & block 3 are dpcnn repeating blocks

            downsample1 = MaxPooling1D(pool_size=3, strides=2)(conc1)

            downsample1 = PReLU()(downsample1)  # pre activation

            block2 = Conv1D(filter_nr, kernel_size, padding='same',

                            kernel_regularizer=conv_kern_reg)(downsample1)

            block2 = SpatialDropout1D(dropout)(block2)

            block2 = PReLU()(block2)

            block2 = Conv1D(filter_nr, kernel_size, padding='same',

                            kernel_regularizer=conv_kern_reg)(block2)

            block2 = SpatialDropout1D(dropout)(block2)

            conc2 = add([downsample1, block2])

            after_pool = Flatten()(conc2)

            outp = Dense(2, activation="sigmoid")(after_pool)

            model = Model(inputs=comment, outputs=outp)

            

        model.summary()

        if tunning :

            model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(),metrics=["accuracy"])

       

        return model



############################################################################################     



class CNN(object):

    def __init__(self, embed_size ,embedding_matrix):

        self.embed_size = embed_size

        self.embedding_matrix = embedding_matrix 

    

    def build_model(self, maxlen, model_num, tunning= False):

        if model_num == 2 :

            y_dim=2

            num_filters=5

            filter_sizes = [1,1,1]

            pooling = 'max'

            dropout=0.5

            pool_padding = 'valid'

            embed_input = Input(shape=(maxlen,))

            

            comment = Input(shape=(maxlen,))

            emb_comment = Embedding(max_features,self.embed_size, weights=[self.embedding_matrix])

            emb_comment.build((None,)) # if you don't do this, the next step won't work

            

            embedded = emb_comment(embed_input)

            x= SpatialDropout1D(0.2)(embedded) 

            ## concat

            pooled_outputs = []

            for i in (filter_sizes):

                conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)

                if pooling=='max':

                    conv = MaxPooling1D(pool_size=maxlen-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)

                else:

                    conv = AveragePooling1D(pool_size=maxlen-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)            

                pooled_outputs.append(conv)

            merge = concatenate(pooled_outputs)



            x = Flatten()(merge)

            x = Dropout(dropout)(x)

            predictions = Dense(y_dim, activation = 'sigmoid')(x)



            model = Model(inputs=embed_input,outputs=predictions)



        if model_num == 1 :

            print("en attente")

        if tunning:

            model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(),metrics=["accuracy"])

        return model

        

        





###########################################################################################

class Neural_Networks(object):

    

    def __init__(self, dataframe=None, model_name=None, learning_rate= 0.0001, epoch=20, batch_size=100,

                number_grams=1, ratio=0.95  , emb_matrix=None ,validation = None, matrix_val=None,

                model_number= 1) :

        self.validation = validation

        self.matrix_val = matrix_val

        self.model_name = model_name

        self._dataframe = dataframe

        self.epoch = epoch

        self.lr = learning_rate

        self.batch_size = batch_size

        self.number = number_grams

        self.emb_matrix= emb_matrix

        self.model_number = model_number

        #split train/test

        self.build_train_test_validation(ratio)

        print("check train")

        #initialize the model

        if self.model_name == 'DPCNN':

            self.model = DPCNN(emb_size, self.emb_matrix).build_model(self.number,model_number)

        if self.model_name == 'GRU':

            self.model = RNN(emb_size, self.emb_matrix).build_model_GRU(self.number, model_number)

        if self.model_name == 'LSTM':

            self.model = RNN(emb_size, self.emb_matrix).build_model_LSTM(self.number, model_number)

        if self.model_name == 'CNN':

            self.model = CNN(emb_size, self.emb_matrix).build_model(self.number, model_number)

        

        

    #*********************************************************************************

    

    #create train /test

    def build_train_test_validation(self,ratio):

        #build train/test:

        self.x_train,self.x_test,self.y_train, self.y_test = train_test_split(

            sequence.pad_sequences(self.emb_matrix, maxlen=self.number, padding='post'),

            self._dataframe['Labels'].values, train_size=ratio,

            random_state=233)

        # create categories O and 1

        self.y_train= np_utils.to_categorical(self.y_train, 2)

        self.y_test = np_utils.to_categorical( self.y_test, 2)

        #build validation:    

        self.x_val = sequence.pad_sequences(self.matrix_val, maxlen=self.number, padding='post')

        self.y_val = self.validation['Labels'].values

        self.y_val= np_utils.to_categorical(self.y_val, 2)

        return self

    #*********************************************************************************

    

    def compileModel(self,loss_function='binary_crossentropy',

            optimizer_function = optimizers.Adam(),

            metrics_function = ['accuracy']):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

        #cross-entropy loss function to calculate the error between the predicted and actual word.

        self.model.compile(loss=loss_function, optimizer=optimizer_function,

                            metrics=metrics_function)

        return self

    #*********************************************************************************

    

    def train(self):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

        self.model.fit(self.x_train, self.y_train, 

                batch_size = self.batch_size, 

                epochs = self.epoch, 

                validation_data=(self.x_val, self.y_val),

                callbacks =[EarlyStopping(monitor='val_loss',min_delta=self.lr)] ,

                verbose = 1)

        # evaluate the model

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1) 

        print('Accuracy: %f' % (accuracy*100))

        

        return self

    #*********************************************************************************

    

    def save(self):

        self.fname = self.model_name

        self.model.save(self.fname)

        return "model saved in ", self.fname

    #*********************************************************************************

    

    def load(self):

        if self.fname:

            print(">> Load model from disc", self.fname)

            self.model = load_model(self.fname)

        return self

    #*********************************************************************************

    def tunning(self,dictionnary = dict(batch_size=[10, 20, 40, 60, 80, 100],  

                                        epochs=[10, 20, 50])): 

        np.random.seed(7)

        if self.model_name == 'DPCNN':

            model = DPCNN(emb_size, self.emb_matrix).build_model

        if self.model_name == 'GRU':

            model = RNN(emb_size, self.emb_matrix).build_model_GRU

        if self.model_name == 'LSTM':

            model = RNN(emb_size, self.emb_matrix).build_model_LSTM

        if self.model_name == 'CNN':

            model = CNN(emb_size, self.emb_matrix).build_model

        

        

            

        classifier = KerasClassifier(build_fn=model, verbose=1)

        param = {'tunning' : [True],

                        'model_num': [self.model_number],

                        'maxlen': [self.number]}

        param.update(dictionnary)

        grid = GridSearchCV(estimator=classifier, param_grid=param,

                                     n_jobs=1, cv=3)

        grid_result = grid.fit(self.x_train, self.y_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        return self

    #*********************************************************************************



#Test Part :

if __name__ == "__main__":

    c = NlpCorpus("../input/corpus_apprentissage_fevrier.csv")

    c.create_ngrams(5)

    c.create_nvectors2(5)

    

    v = NlpCorpus("../input/corpus_validation_fevrier.csv")

    v.create_ngrams(5)

    v.create_nvectors2(5)

    

    print(">> nvectors created")

   

    #should be efficient but doesn't work here 

    #n = Neural_Networks(dataframe=c._dataframe, model_name=input(

    #                    'choose model betwwen (LSTM/GRU/DPCNN/LSTM_BI): '),

    #                    emb_matrix =c.matrix, number_grams=int(input('n_grams: ')),validation=v._dataframe,

    #                    matrix_val=v.matrix, model_number = int(input('1: not trainable vect word and 2 : trainable vect word: ')))

    n = Neural_Networks(dataframe=c._dataframe, model_name='LSTM',

                        emb_matrix =c.matrix, number_grams=5,validation=v._dataframe,

                        matrix_val=v.matrix, model_number = 1)

    n.compileModel()

    n.train()

    #Accuracy DPCNN model 1/2 57.142860

    #Accuracy: LSTM model 1/2 57.142860

   



   
from keras.layers import Embedding

from keras.models import Sequential, Model

from keras.layers import Dense, Activation

from keras.layers import Flatten, Conv1D, SpatialDropout1D, MaxPooling1D,AveragePooling1D, merge, concatenate, Input, Dropout



# ONE LAYER

def model( max_length=3, y_dim=2, num_filters=5, filter_sizes = [2,3,5], pooling = 'max', pool_padding = 'valid', dropout = 0.2):

    # Input Layer

#     embed_input = Input(shape=(max_length,output_dim))

#    embed_input = Input(shape=(max_length,))

 #   x =embed_input

  #  x = SpatialDropout1D(0.2)(x)



    max_length=3

    y_dim=2

    filter_sizes = [1,1,1]

    pooling = 'max'

    dropout=0.5

    embed_input = Input(shape=(max_length,))

    emb_comment = Embedding(2000,100, weights=[c.matrix_emb]

                                    , trainable=False)

    emb_comment.build((None,)) # if you don't do this, the next step won't work

    embedded = emb_comment(embed_input)

    x= SpatialDropout1D(0.2)(embedded)

    ## concat

    pooled_outputs = []

    for i in (filter_sizes):

        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)

        if pooling=='max':

            conv = MaxPooling1D(pool_size=max_length-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)

        else:

            conv = AveragePooling1D(pool_size=max_length-filter_sizes[i]+1, strides=1, padding = pool_padding)(conv)            

        pooled_outputs.append(conv)

    merge = concatenate(pooled_outputs)

        

    x = Flatten()(merge)

    x = Dropout(dropout)(x)

    predictions = Dense(y_dim, activation = 'sigmoid')(x)

    

    model = Model(inputs=embed_input,outputs=predictions)



    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])

    print(model.summary())

    

    from keras.utils import plot_model

    plot_model(model, to_file='shared_input_layer.png')

    

    return model





model = model( max_length=3,y_dim=2,filter_sizes = [2,3,5],pooling = 'max',dropout=0.5)

n = Neural_Networks(dataframe=c._dataframe, model_name='CNN',

                        emb_matrix =c.matrix, number_grams=5,validation=v._dataframe,

                        matrix_val=v.matrix, model_number = 2)

n.compileModel()

n.train()
n.tunning()