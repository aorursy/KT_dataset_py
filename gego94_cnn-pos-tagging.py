import numpy as np

import warnings 

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.display import Image, display

warnings.filterwarnings("ignore")

import gensim 

from gensim.models import Word2Vec 

import tensorflow

import keras

from keras.models import Sequential,Input,Model

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.advanced_activations import LeakyReLU

from keras.utils import plot_model

from sklearn.preprocessing import OneHotEncoder

from keras.callbacks import ModelCheckpoint

import math
for e in open('/kaggle/input/tagssignificato/tags.txt', 'r', encoding='windows-1252'):

    print(e)
input_train =  open('/kaggle/input/traintest/train.txt', 'r', encoding='utf-8-sig')

def print_first_lines(file):

    ind = 0

    for i in file:

        print(i)

        ind += 1

        if ind == 10: break

print_first_lines(input_train)
def dividi(text):

    count=0

    words = []

    tags = []

    for e in open(text, 'r', encoding='utf-8-sig'):

        words.append([])

        tags.append([])

        for p in e.split(" ") :

            spl = p.split("_")

            if len(spl) == 2:

                words[-1].append(spl[0].strip().rstrip("\n").lower())

                tags[-1].append(spl[1].strip().rstrip("\n"))

                count+=1

    return words, tags

# divido il train in parole e tag

words_train, tags_train = dividi('/kaggle/input/traintest/train.txt')

# testo la rete su dati di test esterni al train

words_test, tags_test = dividi('/kaggle/input/traintest/conlltest.txt')

print('Frasi\n\n')

print_first_lines(words_train)

print('\n\nTags\n\n')

print_first_lines(tags_train)
tgs = {'ADJ':0,'NOUN':1,'ADP':2,'DET':3,'PROPN':4,'PUNCT':5,'AUX':6,'VERB':7,'PRON':8,'CCONJ':9,'NUM':10,'ADV':11,'INTJ':12,'SCONJ':13,'X':14,'SYM':15,'PART':16}

inv_tgs = {0:'ADJ',1:'NOUN',2:'ADP',3:'DET',4:'PROPN',5:'PUNCT',6:'AUX',7:'VERB',8:'PRON',9:'CCONJ',10:'NUM',11:'ADV',12:'INTJ',13:'SCONJ',14:'X',15:'SYM',16:'PART'}
class w2v:



    # metodo per addestrare il modello di word2vec

    # carica un modello preaddestrato su un corpus enorme

    # addestra un modello per i singoli caratteri, in modo da riuscire a gestire anche le parole non presenti nel dizionario

    def train(self, words, dim=300, window=10):

        self.dim = dim

        self.window = window

        self.model = gensim.models.Word2Vec.load('/kaggle/input/modelliw2v/wiki_iter5_algorithmskipgram_window10_size300_neg-samples10.m')

        self.model_char = Word2Vec([[c for c in parola]for frase in words_train for parola in frase], min_count = 1, size =  300, window = 5)

    

    # per ottenere la rappresentazione della parola dal modello di word2vec

    # la rappresentazione per le parole sconosciute è data dalla somma delle rappresentazioni dei singoli caratteri

    # per simboli sconosciuti, viene ritornato un vettore nullo 

    def get(self, word):

        try:

            return np.array(self.model.wv[word])

        except KeyError:

            try:

                return np.array(sum([np.array(self.model_char.wv[w]) for w in word]))/len(word)

            except:

                return np.array(np.zeros(self.dim))

class adatta_dati:

    

    # converte in vettori tutte le parole del corpus

    def dividi_parole(self, words):

        return np.array([el for frase in words for el in self.traduci_singola_frase(frase)])

    

    # traduce con OneHotEncoding i tag

    def traduci_tags(self, tags):

        enc = OneHotEncoder()

        return enc.fit_transform(np.array(tags).reshape(-1, 1 ))



    # traduce tutti i dati in modo da renderli validi per la rete definita

    def traduci(self, x_train, y_train, x_test, y_test, window=5, dimension=300):

        self.window=window

        self.dimension=dimension      

        self.w2v = w2v()

        self.w2v.train(x_train, self.dimension, self.window)

        self.train = self.dividi_parole(x_train)

        self.train_tags = self.traduci_tags([ tgs[t] for frase in y_train for t in frase])

        self.words_test = self.dividi_parole(x_test)

        self.tags_test = self.traduci_tags([ tgs[t] for frase in y_test for t in frase])

        print('Fine traduzione')

        print("Self.train.shape = {}".format(self.train.shape))

    

    # ritorna la traduzione con Word2Vec delle parole di una singola frase, in particolare creando delle finestre

    # di dimensione self.window per ogni parola

    # una finestra è data da 5 parole [pi-2, pi-1, pi, pi+1, pi+2] tradotte in vettori di self.dimension elementi

    # le parole che sforano la frase, sono vettori nulli

    def traduci_singola_frase(self, frase):

        return np.array([np.array([

            np.array(np.zeros(self.w2v.dim)) 

            if parola-math.floor(self.window/2)+i<0 or parola-math.floor(self.window/2)+i>=len(frase) 

            else self.w2v.get(frase[parola-math.floor(self.window/2)+i].lower()) 

            for i in range(self.window)

        ] 

        ).reshape(self.window, self.dimension, 1)

    for parola in range(len(frase))])

    

    def get(self):

        return self.train, self.train_tags, self.words_test, self.tags_test

    

r = adatta_dati()

r.traduci(words_train, tags_train ,words_test, tags_test)
class valuta(tensorflow.keras.callbacks.Callback):

    def __init__(self, tagger):

        self.tagger = tagger

    def on_epoch_end(self, epoch, logs={}): 

        self.tagger.res.append(self.tagger.net.evaluate(self.tagger.words_test, self.tagger.tags_test, batch_size=128))
class tagger:



    # percorso dove salvare il modello e il disegno della rete

    def __init__(self, path):

        self.path = path



    # assegna i parametri a valori dell'oggetto e chiama i metodi per addestrare la rete

    # per comodità passo come parametro la classe adatta_dati precedentemente creata così da velocizzare la classe tagger

    def addestra(self, adatta_dati, epoche=20, window=5, dimension=300):

        self.window=window

        self.dimension=dimension      

        self.adatta_dati = adatta_dati  

        self.epoche=epoche

        self.res = []

        self.w2v = self.adatta_dati.w2v

        self.train, self.train_tags,self.words_test, self.tags_test = self.adatta_dati.get() #prendo i dati dalla classe adatta_dati

        self.addestra_rete()



    # definisce la rete 

    def define_net(self):

        net = Sequential()

        net.add(Conv2D(32, kernel_size=(3, 3), input_shape=(self.window, self.dimension, 1)))

        net.add(LeakyReLU(alpha=0.1))

        net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        net.add(Flatten())

        net.add(Dense(256, activation='linear'))

        net.add(LeakyReLU(alpha=0.1))                  

        net.add(Dense(17, activation='softmax'))

        return net



    # definisce la rete poi la carica se già presente, altrimenti compila la rete e la ritorna

    def get_net(self):

        net = self.define_net()

        self.load_net(self.path + "model.hdf5", net)

        net.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

        return net



    # metodo per far partire l'addestramento della rete, visualizza inoltre la rete e salva il modello ad ogni iterazione, se essa ha migliorato la precedente accuracy 

    def addestra_rete(self):

        self.net = self.get_net()

        self.net.summary()

        plot_model(self.net, show_shapes=True, show_layer_names=True, to_file=self.path + 'model.png')

        display(Image(retina=True, filename=self.path + 'model.png'))

        print(self.train.shape)

        history = self.net.fit(self.train, self.train_tags, validation_data=(self.train, self.train_tags), epochs=self.epoche,batch_size=1024,shuffle=True, verbose=2, 

                               callbacks=[

                                          ModelCheckpoint(self.path + "model.hdf5", monitor='acc', verbose=2, save_best_only=True, mode='max'), 

                                          valuta(self)

                                          ])

    

    # cerca di caricare i pesi di una rete dal path passato

    def load_net(self, path, net):

        print('Carico il modello della rete')

        try:

            net.load_weights(path)

            print('Modello caricato')

        except:

            print('Modello non trovato')

        return net



    # metodo per predire un esempio

    def predict(self, x):

        y = self.net.predict(x)

        return [inv_tgs[np.where(el == np.amax(el))[0][0]] for el in y]

        

t = tagger("/kaggle/working/")

t.addestra(r)
plt.plot([i for i in range(1,21)], np.array(t.res)[:,1]*100)

plt.xlabel('Epoche')

plt.ylabel('Accuracy %')
frase = 'Per analisi grammaticale si intende il procedimento per identificare la categoria lessicale di ogni parola nel contesto nel quale è usata .'.split(" ")

ris = t.predict( np.array(r.traduci_singola_frase(frase)))

for i in range(len(frase)):

    print('Parola : {} , tag : {}'.format(frase[i], ris[i]))