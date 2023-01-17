



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder





from sklearn.metrics import confusion_matrix

from wordcloud import WordCloud



from sklearn.metrics import roc_auc_score







!pip install contractions

import contractions



from nltk.stem import WordNetLemmatizer 

from nltk.corpus import wordnet as wn

from nltk import pos_tag

from collections import defaultdict





from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/train.txt",sep=';',names=['text','emotion'])

df_val = pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/val.txt",sep=';',names=['text','emotion'])

df_test = pd.read_csv("/kaggle/input/emotions-dataset-for-nlp/test.txt",sep=';',names=['text','emotion'])
df_train.emotion.value_counts()
data_total = [df_train,df_val,df_test]



    


fig,ax = plt.subplots(1,3,figsize=(20,5))

for i,data in enumerate(data_total):

    sns.countplot(data.emotion,ax=ax[i])
def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)


tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV

stop_words = set(stopwords.words('english')) 

stem = WordNetLemmatizer ()

def clean_text(text):

    text = text.lower()

    text = word_tokenize(text)

    text = [contractions.fix(word) for word in text]

    text = [stem.lemmatize(w,tag_map[tag[0]]) for w, tag in pos_tag(text) if w not in stop_words]

    return " ".join(text)

    

    
for i,data in enumerate(data_total):

    data.text = data.text.apply(clean_text)
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 

).generate(str(data))

    return wordcloud



categories = df_train["emotion"].unique()







fig, axes = plt.subplots(ncols=2, nrows=3,figsize=(30,25))

plt.axis('off')

for category, ax in zip(categories, axes.flat):

    wordcloud = show_wordcloud(df_train[df_train["emotion"]==category]['text'])

    ax.imshow(wordcloud)

    ax.title.set_text(category)

    ax.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.01)



laberEncoder = LabelEncoder()

laberEncoder.fit(df_train.emotion)

for i,data in enumerate(data_total):

    data.emotion = laberEncoder.transform(data.emotion)
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1, 2))

vectors = vectorizer.fit_transform(df_train.text)

feature_names = vectorizer.get_feature_names()

dense = vectors.todense()

denselist = dense.tolist()

df = pd.DataFrame(denselist, columns=feature_names)
vectors = vectorizer.transform(df_val.text)

dense = vectors.todense()

denselist = dense.tolist()

df_valfeature = pd.DataFrame(denselist, columns=feature_names)



class EnsembleHelper(object):

    def __init__(self,models,seed=42,params=None,cv=5):

        self.seed = seed

        self.params = params

        

        

    def gridSearchCV(self,estimator,params,X_train,Y_train):

        if not params :

            params = {}

        grid = GridSearchCV(estimator=estimator, param_grid=params, cv=5)

        grid.fit(X_train,Y_train)

        rf_best = grid.best_estimator_

        print(estimator,grid.best_params_,grid.best_score_,sep="|")

        return rf_best

        

    

    def fit(self, X_train,Y_train):

        ensemblemodel = []

        for key , estimator in models:

            Dict = {search_key.split("__")[1]:val for search_key, val in self.params.items() if search_key.startswith(key)}

            bestmodel = self.gridSearchCV(estimator,Dict,X_train,Y_train)

            final = (key , bestmodel)

            ensemblemodel.append(final)

        

        self.clf = VotingClassifier(estimators=ensemblemodel,voting='hard')

        self.clf.fit(X_train,Y_train)

        

        

    def predict(self,X_test):

        predictions = self.clf.predict(X_test)

        return predictions

        

    def score(self,X_test,Y_test):

        return self.clf.score(X_test,Y_test)    

        

        
models = [

    ('lr' , LogisticRegression(random_state=0,max_iter=10000)),

    ('sgd', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, tol=None)),

    ('mnb', MultinomialNB()),

    ('nb',BernoulliNB()),

    ('dt',DecisionTreeClassifier())

]



params = {'lr__C': [0.1,1.0,10.0],

          'mnb__alpha': np.linspace(0.0, 1.0, 5),

          'nb__alpha': np.linspace(0.0, 1.0, 5),

          'dt__criterion' : ['gini', 'entropy']}

ensemble = EnsembleHelper(models,cv=5,params=params)

ensemble.fit(df, df_train.emotion)

predict = ensemble.predict(df_valfeature)

print("Ensemble score : ",ensemble.score(df_valfeature,df_val.emotion))
from gensim.models import word2vec

import tqdm
num_features = 5000  

min_word_count = 2 

num_workers = 4    

context = 10        

downsampling = 1e-3 





word2VecModel = word2vec.Word2Vec(df_train.text,\

                          workers=num_workers,\

                          size=num_features,\

                          min_count=min_word_count,\

                          window=context,

                          sample=downsampling)





word2VecModel.init_sims(replace=True)





#word2VecModel_name = "emotionsword2Vec"

#word2VecModel.save(word2VecModel_name)





def document_vector(doc):

    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""

    doc = [word for word in doc if word in word2VecModel.wv.vocab]

    return np.mean(word2VecModel[doc], axis=0)



train_vec = df_train.text.apply(document_vector)

val_vec = df_val.text.apply(document_vector)
len(train_vec[0])
models = [

    ('lr' , LogisticRegression(random_state=0,max_iter=10000)),

    ('sgd', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, tol=None))

]



params = {'lr__C': [0.1,1.0,10.0],

          'dt__criterion' : ['gini', 'entropy']}
ensemble = EnsembleHelper(models,cv=5,params=params)

ensemble.fit(list(train_vec), df_train.emotion)

predict = ensemble.predict(list(val_vec))

print("Ensemble score : ",ensemble.score(list(val_vec),df_val.emotion))
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn import utils
def tokenize_text(text):

    tokens = []

    for word in word_tokenize(text):

        if len(word) < 2:

            continue

        tokens.append(word.lower())

    return tokens



def vector_for_learning(model, input_docs):

    sents = input_docs

    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])

    return targets, feature_vectors
train_documents = [TaggedDocument(words=tokenize_text(x), tags=[y] ) for x, y in zip(df_train['text'], df_train['emotion'])]

val_documents = [TaggedDocument(words=tokenize_text(x), tags=[y] ) for x, y in zip(df_val['text'], df_val['emotion'])]
model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=4, alpha=0.025, min_alpha=0.001)

model_dbow.build_vocab([x for x in train_documents])

train_documents  = utils.shuffle(train_documents)

model_dbow.train(train_documents,total_examples=len(train_documents), epochs=30)



y_train, X_train = vector_for_learning(model_dbow, train_documents)

y_test, X_test = vector_for_learning(model_dbow, val_documents)
models = [

    ('lr' , LogisticRegression(random_state=0,max_iter=10000)),

    ('sgd', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, tol=None)),

    ('dt',DecisionTreeClassifier())

]



params = {'lr__C': [0.1,1.0,10.0],

          'dt__criterion' : ['gini', 'entropy']}
ensemble = EnsembleHelper(models,cv=5,params=params)

ensemble.fit(X_train, y_train)

predict = ensemble.predict(X_test)

print("Ensemble score : ",ensemble.score(X_test,y_test))
from numpy import array

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.vis_utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Embedding

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.merge import concatenate

from keras import regularizers
class DLModel(object):

    def __init__(self,X_train,Y_train):

        self.X_train = X_train

        self.Y_train = Y_train

        self.output = len(Y_train.unique())

        self.length = max([len(s.split()) for s in self.X_train])

        self.tokenizer = self.create_tokenizer(X_train)

        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.model = self.define_model(self.length, self.vocab_size)

        

     

    def preprocessing(self,X_train):

        tokenizer = self.create_tokenizer(X_train)

        encoded = tokenizer.texts_to_sequences(X_train)

        padded = pad_sequences(encoded, maxlen=self.length, padding='post')

        return padded

        

    def create_tokenizer(self,sentences):

        tokenizer = Tokenizer()

        tokenizer.fit_on_texts(sentences)

        return tokenizer



    

    # define the model

    def define_model(self,length, vocab_size):

        # channel 1

        inputs1 = Input(shape=(length,))

        embedding1 = Embedding(vocab_size, 100)(inputs1)

        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)

        drop1 = Dropout(0.5)(conv1)

        pool1 = MaxPooling1D(pool_size=2)(drop1)

        flat1 = Flatten()(pool1)

        # channel 2

        inputs2 = Input(shape=(length,))

        embedding2 = Embedding(vocab_size, 100)(inputs2)

        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)

        drop2 = Dropout(0.5)(conv2)

        pool2 = MaxPooling1D(pool_size=2)(drop2)

        flat2 = Flatten()(pool2)

        # channel 3

        inputs3 = Input(shape=(length,))

        embedding3 = Embedding(vocab_size, 100)(inputs3)

        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)

        drop3 = Dropout(0.5)(conv3)

        pool3 = MaxPooling1D(pool_size=2)(drop3)

        flat3 = Flatten()(pool3)

        # merge

        merged = concatenate([flat1, flat2, flat3])

        # interpretation

        dense1 = Dense(20,kernel_regularizer=regularizers.l2(0.001), activation='relu')(merged)

        drop4 = Dropout(0.5)(dense1)

        dense2 = Dense(10,kernel_regularizer=regularizers.l2(0.001), activation='relu')(drop4)

        drop5 = Dropout(0.5)(dense2)

        outputs = Dense(self.output, activation='sigmoid')(drop5)

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        # compile

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # summarize

        print(model.summary())

        plot_model(model, show_shapes=True, to_file='multichannel.png')

        return model

    

    def fit(self,X_train,Y_train,X_val,Y_val):

        trainX = self.preprocessing(X_train)

        valX = self.preprocessing(X_val)

        self.model.fit([trainX,trainX,trainX], pd.get_dummies(Y_train),validation_data=([valX,valX,valX], pd.get_dummies(Y_val)), epochs=20, batch_size=16)

        

    def evaluate(self,X_test,Y_test):

        testX = self.preprocessing(X_test)

        loss, acc = self.model.evaluate([testX,testX,testX], pd.get_dummies(Y_test), verbose=0)

        return loss, acc

    

    def predict(self,X_test):

        X_test = self.preprocessing(X_test)

        ynew = self.model.predict(X_test)

        return ynew

        
model = DLModel(df_train.text,df_train.emotion)

model.fit(df_train.text,df_train.emotion,df_val.text,df_val.emotion)

loss, acc = model.evaluate(df_test.text,df_test.emotion)

print(loss, acc)