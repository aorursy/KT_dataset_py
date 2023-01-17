import numpy as np

import pandas as pd

import re

from gensim.models import KeyedVectors

from gensim.models import Word2Vec

from gensim.test.utils import common_texts, get_tmpfile





from nltk.corpus import stopwords
df = pd.read_csv('sarcasm.v2',delimiter='\t', header=None)

df = df[[1,0]]

df.columns = ("v1","v2")

df.head()
sns.countplot(df.v1)

plt.xlabel('Label')

plt.title('Number of sarcastic and not sarcastic comments')
df['v3'] = df2['v2'].str.len()

df.describe()

##falta agregar el analisis de reddit
def prep_data(line):

    sentence = re.sub('[^a-zA-Z]',' ',line)

    words = sentence.lower().split()

    stops = set(stopwords.words("english")) 

    words = [w for w in words if not w in stops]

    

    return (words)

sarc_sentences = []

sarc_label = []

sarc_data = open('sarcasm.v2') ##CAMBIAR PARA QUE FUNCIONE EN EL DATASET DE ARRIBA

for line in sarc_data:

    X,y = line.split('\t')

    sarc_sentences.append(prep_data(X))

    if y == 'sarc':

        sarc_label.append(0)

    else:

        sarc_label.append(1)



sarc_sentences[:2]

    

    

    
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

model.doesnt_match("man woman dog child kitchen".split()) #for test
def create_feature_vec(sentence,model, num_features):

    feat_vec = np.zeros(num_features,dtype="float32")

    nwords = 0

    

    index2word_set = set(model.index2word)

    

    for word in sentence:

        if word in index2word_set:

            nwords = nwords + 1

            feat_vec = np.add(feat_vec,model[word])

    feat_vec = np.divide(feat_vec,nwords)

    return feat_vec



def get_avg_vec(sentences, model, num_features):



    feat_vec = np.zeros((len(sentences),num_features),dtype="float32")

    for counter,sentence in enumerate(sentences):

        feat_vec[counter] = create_feature_vec(sentence,model,num_features)

    return feat_vec



sarc_vecs = get_avg_vec(sarc_sentences,model,300)
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences



max_sen = 0

for vec in sarc_vecs:

    if len(vec) > max_sen:

        max_sen = len(vec)

print(max_sen)

X_train, X_test, y_train, y_test = train_test_split(sarc_vecs, sarc_label, test_size=0.20, random_state=42)

X_train = pad_sequences(X_train, padding='post', maxlen=max_sen)

X_test = pad_sequences(X_test, padding='post', maxlen=max_sen)



print(len(sarc_vecs))
from keras.models import Sequential

from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding



def CNN (data_size,input_len,max_sen):

    nn = Sequential()

    nn.add(Embedding(data_size, input_len, input_length=max_sen,trainable=False))

    nn.add(Conv1D(128, 5, activation='relu'))

    nn.add(GlobalMaxPooling1D())

    nn.add(Dense(10, activation='relu'))

    nn.add(Dense(1, activation='sigmoid'))

    nn.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



    nn.summary()

    return nn



nn = CNN(4692,300,max_sen)
nn.fit(X_train, y_train,

                     epochs=50,

                     verbose=False,

                     validation_data=(X_test, y_test),

                     batch_size=10)

loss, accuracy = nn.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = nn.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))
from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

def RNN(max_words,max_len):

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

    model.summary()

    return model
rnn = RNN(max_sen)

rnn.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,

          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#Source: https://github.com/jjrob13/sklearn_cotraining

from copy import copy

class CoTraining(object):

    def __init__(self,clf, clf2 = None, n_pos=-1,n_neg=-1,k_iter=30,pool=75):

        self.clf1_ = clf

        if clf2 == None:

            self.clf2_ = copy(clf)

        else:

            self.clf2_ = clf2

        if (n_pos == -1 and n_neg != -1) or (n_pos != -1 and n_neg == -1):

            raise ValueError('Current implementation supports either both p and n being specified, or neither')

        self.n_pos_ = n_pos

        self.n_neg_ = n_neg

        self.k_iter_ = k_iter

        self.pool_ = pool

        

        random.seed()

    def fit(self,X1,X2,y):

        

        y = np.asarray(y)

        

        if self.n_pos_ == -1 and self.n_neg_ == -1:

            num_pos = sum(1 for y_i in y if y_i == 1)

            num_neg = sum(1 for y_i in y if y_i == 0)

            n_p_ratio = num_neg / float(num_pos)

            if n_p_ratio > 1:

                self.n_pos_ = 1

                self.n_neg_ = round(self.n_pos_*n_p_ratio)

            else:

                self.n_neg_ = 1

                self.p_pos_ = round(self.n_neg_/n_p_ratio)

        assert(self.p_pos_ > 0 and self.n_neg_ > 0 and self.k_iter_ > 0 and self.pool_ > 0)

        

        U = [i for i, y_i in enumerate(y) if y_i == -1]

        random.shuffle(U)

        U_ = U[-min(len(U), self.pool_):]

        L = [i for i, y_i in enumerate(y) if y_i != -1]

        U = U[:-len(U_)]

        it = 0

        

        while it != self.k_iter_ and U:

            it += 1

            self.clf1_.fit(X1[L], y[L])

            self.clf2_.fit(X2[L], y[L])

  

            y1 = self.clf1_.predict(X1[U_])

            y2 = self.clf2_.predict(X2[U_])

            

            n, p = [], []

            

            for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):

                if len(p) == 2 * self.p_ and len(n) == 2 * self.n_:

                    break

                if y1_i == y2_i == 1 and len(p) < self.p_:

                    p.append(i)

                if y2_i == y1_i == 0 and len(n) < self.n_:

                    n.append(i)

                y[[U_[x] for x in p]] = 1

                y[[U_[x] for x in n]] = 0

                

                L.extend([U_[x] for x in p])

                L.extend([U_[x] for x in n])

                

                for i in p: U_.pop(i)

                for i in n: U_.pop(i)

                    

                add_counter = 0 

                num_to_add = len(p) + len(n)

                while add_counter != num_to_add and U:

                    add_counter += 1

                    U_.append(U.pop())

        self.clf1_.fit(X1[L], y[L])

        self.clf2_.fit(X2[L], y[L])

        return self

    def predict(self, X1, X2):

        

        y1 = self.clf1_.predict(X1)

        y2 = self.clf2_.predict(X2)

        

        y_pred = np.asarray([-1] * X1.shape[0])

        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):

            if y1_i == y2_i:

                y_pred[i] = y1_i

            else:

                y_pred[i] = random.randint(0, 1)

        assert not (-1 in y_pred)

        return y_pred
#Esto se puede mover pa arriba



reddit = pd.read_csv('reddit_comments.csv')



text_reddit = reddit['text'].values



reddit_sentences = []

for comment in text_reddit:

    reddit_sentences.append(prep_data(comment))



reddit_vecs = get_avg_vec(reddit_sentences,model,300)

reddit_label = [-1 for i in range(len(reddit_sentences)) ]



#mix ambos datasets.. X1 (MITAD REDDIT MITAD SARCASM pero con la mitad de los features) Y X2(la otra mitad) y (son las etiquetas de todo 0 sarc, 1 no sarc, -1 nada)

#calcular cuantos pos y neg por X1



random.Random(4).shuffle(sarc_vec)

random.Random(4).shuffle(sarc_label)



sarc_len = len(sarc_vec)

reddit_len = len(reddit_vecs)

X = sarc_vec + reddit_vecs



y = sarc_label + reddit_label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)



X1_train = X_train[:,:max_sen//2]

X2_train = X_train[:,max_sen//2:]



X1_test = X_test[:,:max_sen//2]

X2_test = X_test[:,max_sen//2:]



from sklearn.datasets import make_classification

cotraining = CoTraining(CNN(len(X1_train,max_sen,max_sen)), RNN())

cotraining.fit(X1_train,X2_train,y_train)

y_pred = cotraining.predict(X1_test,X2_test)

print (classification_report(y_test, y_pred))

 