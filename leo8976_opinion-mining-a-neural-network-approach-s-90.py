import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
#read the data
data = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
#pick at the first 5 rows of the data
data.head()
data.info()
data.sentiment.value_counts()
#importing and installing of packages
!pip install stanza
import os,re,nltk,stanza
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
#for lets prepare all the functions and tools, in the end will apply them all in sequntioal order on the dataframe
def data_cleaning(rev:str)->str:
    """
    lower-casing, decontraction  and cleaning of html tags for each review

    Parameters
    ----------
    rev : str
        the review that needs to be preprocessed.

    Returns
    -------
    rev : str
        the cleaned review .

    """
    rev = re.sub(re.compile('<.*?>'), "", rev.lower())
    rev = re.sub("'s", " is", rev)
    rev = re.sub("'ve", " have", rev)
    rev = re.sub("n't", " not", rev)
    rev = re.sub("cannot", " can not", rev)
    rev = re.sub("'re", " are", rev)
    rev = re.sub("'d", " would", rev)
    rev = re.sub("'ll", " will", rev)
    rev = re.sub("won\'t", "will not", rev)
    rev = re.sub("can\'t", "can not", rev)
    rev = re.sub("\'t", " not", rev)
    rev = re.sub("\'ve", " have", rev)
    rev = re.sub("\'m", " am", rev)
    rev = re.sub("[^a-z ]+", '', rev.replace('.',' ').replace(',',' '))
    return rev
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not', 'nor','no'}
def remove_bad_words(rev:str)->str:
    """
     stop words(except no,not and nor) and single character word removal
        
     Parameters
     ----------
     rev : str
         review to be cleaned.

     Returns
     -------
     rev : str
         review after the cleaning process.

     """
    temp=""
    for word in rev.split():
        if word not in stop_words and len(set(word))>1:
            temp+=word+ " "
    return temp
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors="tokenize, mwt,pos,lemma",tokenize_no_ssplit=True)
def lematize(rev:str)->list:
    """
    lemattization of the review. 
        
    Parameters
    ----------
    rev : str
        the review that we want to lemmatize.

    Returns
    -------
    rev : list
        the review after tokenization and lematization.

    """
    return [ word.lemma for sent in nlp(rev).sentences for word in sent.words]
def split_into_train_test_dev():
    global data
    temp = {}
    temp['train'], temp['test'] = train_test_split(data, test_size=0.15, random_state=42)
    temp['train'] , temp['dev'] = train_test_split(temp['train'], test_size=0.15, random_state=42)
    data = temp
class Dictionary:
    """
    a class that represents the dictionary of the text
    the class gives a unique index to every word
    """
    def __init__(self):
        self.word2idx = {"PAD":0}
        self.idx2word = {0:"PAD"}
        self.word2freq = {"PAD":1}
        self.idx = 1
    
    def add_words(self,input):
        """
        Input: word or words 
        InputType: list,tuple or str
        a method to add a word(s),
        the method checks if the word(s) is allredy in the dictinoary
        if not, add it.
        othewith adds one to its freq
        """
        def add():
            if word not in self.word2idx.keys():
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.word2freq[word] = 1
                self.idx+=1
            else:
                self.word2freq[word] += 1
       
        def addSeq():
            nonlocal word
            for word in input:
                add()
        inputType = type(input)
        word = []
        if inputType != list and inputType != str and inputType != tuple:
            raise TypeError ("dict at add_word :the type of the input is not allowed")
        if inputType == "str":
            input = [input]
        addSeq()
            
            
    def __getitem__(self,words):
        try:
            if type(words) == str:
                return self.word2idx[words]
            return [self.word2idx[word] for word in words]

        except KeyError:
            print("The word does not exists in the dict")
            return None
    
    def __setitem__(self,word,id):
        """
        add (word,id) to the dict iff the word does not appear in the dict
        """
        if word not in self.word2idx.keys():
            self.word2idx[word] = id 
            self.word2idx[id] = word 
            self.idx+=1

            
    def __len__(self):
        """
        returns the size of the dictionary
        """
        return len(self.word2idx)
dictionary =Dictionary()
def fit_on_train(rev:list)->list:
    """
    building dictionary based on the words in the train set
    the final result of this method is a dictionary that holds 
    the following information:
    for each word:
        1. it's uniqe id (for future translation of the words in case of embedding layer)
        2. the freq of each word in the set (for optional filtering of un-freqent words)
 

    Parameters
    ----------
    rev : list
        list of words.

    Returns
    -------
    rev : list
        list of words.

    """
    dictionary.add_words(rev)
    return rev
def transform(rev:list)->list:
    """
    function to transform a review into sequence 

    Parameters
    ----------
    rev : list
        list of tokens.
 
    Returns
    -------
    list
        returns the transformed review.

    """
    return [dictionary[word] for word in rev if dictionary[word]]
def build_weight_matrix():
    """
    a function that creates weight_matrix out of FastText pre-trained vectors
    for future use as a weights init of an embedding layer
    if there is no vector for the word that we inizlize it with random vector
        
    Returns
    -------
    weight_matrix : list
    """
    def init_fast_text():
        nonlocal fastText,dim
        print("Loading FastText pre-trained vectors")
        with open('../input/fasttext-wikinews/wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n',errors='ignore') as f:
            _, dim = map(int, f.readline().split())
            fastText = {}
            for line in f:
                tokens = line.rstrip().split(' ')
                fastText[tokens[0]] = np.asarray(tokens[1:], "float32")
    
    def build_matrix():
        nonlocal weights_matrix
        print("starting to build weight matrix for embedding encoding,  based on FastText pre-trained vectors")
        maching_words = 0
        dataset_size = len(dictionary)+1
        weights_matrix = np.zeros(shape=(dataset_size,dim))
        for i,word in enumerate(dictionary.word2idx.keys(),1):
            try:
                save = fastText[word]
                maching_words += 1
            except KeyError:
                save = np.random.uniform(size=(dim,))
            weights_matrix[i] = save
                     
        print("pre-treind words: {} randomaly initilaized: {}".format(maching_words,dataset_size))     
    fastText,dim,weights_matrix=[],0,[]
    init_fast_text()
    build_matrix()
    data['matrix'] = weights_matrix
data.review = data.review.apply(data_cleaning)
data.review = data.review.apply(remove_bad_words)
data.review = data.review.apply(lematize)
data.sentiment = data.sentiment.map(lambda sent:{'positive':1,'negative':0}[sent])
split_into_train_test_dev()
data["train"].review.apply(fit_on_train)
max_len = data["train"].review.apply(len).max()
for set in ["train","test","dev"]:     
    data[set].review = data[set].review.apply(transform)
    data[set].review = data[set].review.apply(lambda rev:rev+[0]
                                                      *(max_len-len(rev)) 
                                   if len(rev)<max_len else rev[:max_len])
    data[set] ={"X":np.array(data[set].review.to_list()),"Y":np.array(data[set].sentiment.to_list())}   
build_weight_matrix()
#for future use of the preprocessed data , so we will not do the preprocess again and agin on the draft , saving!
import pickle
data['dict'] = dictionary
with open("/kaggle/working/data.pkl","wb") as f:
     pickle.dump(data,f)
#load the  trained data from the last draft session
import pickle
with open('../input/pretrained-data-from-previous-draft-session/data.pkl',"rb") as f:
     data = pickle.load(f)
#importing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Flatten 
from tensorflow.keras.callbacks import ModelCheckpoint
#callback: we will monitor val_acc (because it is a classification problem) and save the best model.
def checkpoint(name:str):
    return ModelCheckpoint(name, monitor='val_binary_accuracy', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto', save_freq="epoch")
#baseline
model = Sequential([Embedding(len(data['dict'])+1, 300,weights=[data['matrix']],input_length=data['train']['X'].shape[-1],
                              mask_zero=True,name="words_latent_space"),
                    Flatten(name="flat"),
                    Dense(1, activation='sigmoid',name='classifyer')],name="Baseline")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.summary()
hist = model.fit(data['train']['X'], data['train']['Y'], batch_size=64, epochs=8,
                                  validation_data=(data['dev']['X'],data['dev']['Y']),
                 callbacks=[checkpoint('/kaggle/working/fc_baseline.h5')])
#configure ploting function for future use
import matplotlib.pyplot as plt
def plot(hist:dict,y_title:str,plot_title:str):
    plt.figure(figsize=(6,2),dpi=140,facecolor="w")
    plt.grid(c='black',linestyle="-",linewidth=2)
    plt.ylabel(y_title)
    plt.xlabel("epochs")
    plt.title(plot_title)
    plt.plot(hist)
plot(hist.history["binary_accuracy"],"acc","train");plot(hist.history["val_binary_accuracy"],"val_acc","dev")
model = Sequential([Embedding(len(data['dict'])+1,200,input_length=data['train']['X'].shape[-1],
                              mask_zero=True,name="words_latent_space"),
                    Flatten(name="flat"),
                    Dense(1, activation='sigmoid',name='classifyer')],name="200_emd")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.summary()
hist = model.fit(data['train']['X'], data['train']['Y'], batch_size=64, epochs=8,
                                  validation_data=(data['dev']['X'],data['dev']['Y']),
                 callbacks=[checkpoint('/kaggle/working/fc_baseline_200_emd.h5')])
plot(hist.history["binary_accuracy"],"acc","train");plot(hist.history["val_binary_accuracy"],"val_acc","dev")
from tensorflow.keras import regularizers

model = Sequential([Embedding(len(data['dict'])+1,200,input_length=data['train']['X'].shape[-1],
                              embeddings_regularizer=regularizers.l2(1e-6),
                              mask_zero=True,name="words_latent_space"),
                    Flatten(name="flat"),
                    Dense(1, activation='sigmoid',name='classifyer')],name="200_emb_d")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.summary()
hist = model.fit(data['train']['X'], data['train']['Y'], batch_size=64, epochs=5,
                                  validation_data=(data['dev']['X'],data['dev']['Y']),
                 callbacks=[checkpoint('/kaggle/working/fc_baseline_200_emb_r.h5')])
plot(hist.history["binary_accuracy"],"acc","train");plot(hist.history["val_binary_accuracy"],"val_acc","dev")
model = Sequential([Embedding(len(data['dict'])+1,128,input_length=data['train']['X'].shape[-1],
                              embeddings_regularizer=regularizers.l2(1e-6),
                              mask_zero=True,name="words_latent_space"),
                    Flatten(name="flat"),
                    Dense(1, activation='sigmoid',name='classifyer')],name="128_emd")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.summary()
hist = model.fit(data['train']['X'], data['train']['Y'], batch_size=64, epochs=5,
                                  validation_data=(data['dev']['X'],data['dev']['Y']),
                 callbacks=[checkpoint('/kaggle/working/128_emd.h5')])
plot(hist.history["binary_accuracy"],"acc","train");plot(hist.history["val_binary_accuracy"],"val_acc","dev")
#baseline
from tensorflow.keras.layers import Bidirectional,LSTM
model = Sequential([Embedding(len(data['dict'])+1, 300,weights=[data['matrix']], mask_zero=True),
                    Bidirectional(LSTM(300),name="bi-lstm"),
                    Dense(1, activation='sigmoid',name='classifyer')],name="lstm_base")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.summary()
hist = model.fit(data['train']['X'], data['train']['Y'], batch_size=64, epochs=5,
                                  validation_data=(data['dev']['X'],data['dev']['Y']),
                 callbacks=[checkpoint('/kaggle/working/lstm_base.h5')])
plot(hist.history["binary_accuracy"],"acc","train");plot(hist.history["val_binary_accuracy"],"val_acc","dev")
model = Sequential([Embedding(len(data['dict'])+1, 128,mask_zero=True),
                    Bidirectional(LSTM(128),name="bi-lstm"),
                    Dense(1, activation='sigmoid',name='classifyer')],name="lstm_128")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
model.summary()
hist = model.fit(data['train']['X'], data['train']['Y'], batch_size=64, epochs=5,
                                  validation_data=(data['dev']['X'],data['dev']['Y']),
                 callbacks=[checkpoint('/kaggle/working/lstm_128.h5')])
plot(hist.history["binary_accuracy"],"acc","train");plot(hist.history["val_binary_accuracy"],"val_acc","dev")
from tensorflow.keras.models import load_model
best_model= load_model('/kaggle/working/lstm_128.h5')
eval=best_model.predict_classes(data['test']['X'])
from sklearn.metrics import classification_report
print(classification_report(eval, data['test']['Y']))                        