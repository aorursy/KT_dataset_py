import pandas as pd #database manipulation

import numpy as np #math library

import matplotlib.pyplot as plt

import seaborn as sns
import codecs

from tqdm import tqdm

raw_df=pd.read_csv('../input/intention/intent.csv',encoding='utf-8',delimiter=',')
raw_df[raw_df['temp'].isnull()]
import re

regexcsv=re.compile("\n,")
with open('../input/intention/intent.csv','r') as f:

    raw=f.read()

fixed=regexcsv.sub(',',raw)

with open('intent_fixed.csv','w') as f:

    f.write(fixed)
fixed_df=pd.read_csv('../input/intention/intent_fixed.csv',encoding='utf-8',delimiter=',')
fixed_df[fixed_df['temp'].isnull()]
fixed_df.isnull().sum().plot.bar(title='Empty Values')

fixed_df=fixed_df.fillna(0)

fixed_df.isnull().sum()
sub=fixed_df[fixed_df.columns[1:]].sum()

print(sub)

sub.plot.bar(title='Class Distribution')
def plot_doc_lengths(dataframe):

    max_seq_len = np.round(dataframe.doc_len.mean() + dataframe.doc_len.std()).astype(int)

    sns.distplot(tuple(dataframe.doc_len), hist=True, kde=True, label='Document lengths')

    plt.axvline(x=max_seq_len, color='k', linestyle='--', label=f'Sequence length mean:{max_seq_len}')

    plt.title('Document lengths')

    plt.legend()

    plt.show()

    print(f" the bigger document contain {dataframe['doc_len'].max()} words  and the smaller {dataframe['doc_len'].min()} words")
fixed_df['doc_len'] = fixed_df.motivos.apply(lambda words: len(words.split()))

plot_doc_lengths(fixed_df)
fixed_df[fixed_df.doc_len<3].head(10)
fixed_df[fixed_df.doc_len>50].head(10)
fixed_df=fixed_df[fixed_df.doc_len>2]
plot_doc_lengths(fixed_df)
sub=fixed_df[fixed_df.columns[1:-1]].sum()

print(sub)

sub.plot.bar(title='Class Distribution')
!python -m spacy download es
import re

import spacy

import string

#this are our cleaning rules

cleaningOptions = {

    '[A-Za-z0-9_-]{18,}':' ',#remove long words

    '[0-9]+':' ',#remove numbers

    #delete emails

    "[A-Za-z0-9_-]*@[A-Za-z0-9._-]*\s?":' ',

    #delete links

    "https?://[A-Za-z0-9./-]+":' ',

}



co_SentenceLevel = {

    '[A-Za-z0-9_-]{18,}':' ',#remove long words

    '[0-9]+':' ',#remove numbers

    #Separate simbols from words

    '(':' ( ',

    '/':' / ',

    ')':' ) ',

    '?':' ? ',

    '¿':' ¿ ',

    ']':' ] ',

    '[':' [ ',

    '}':' } ',

    '{':' { ',

    '<':' < ',

    '"':' " ',

    '>':' > ',

    ',':' , ',

    '!':' ! ',

    '.':' . ',

    ':':' : ',

    #delete some useless simbols

    '-':' ',

    '$':' ',

    '%':' ',

    #delete double space, and sequences of "-,*,^,."

    '\s{2,}|\?{2,}|\!{2,}|#{2,}|={2,}|-{2,}|_{2,}|\.{2,}|\*{2,}|\^{2,}':' ',

    #delete emails

    "[A-Za-z0-9_-]*@[A-Za-z0-9._-]*\s?":"",

    #delete links

    "https?://[A-Za-z0-9./-]+":"",

}





def escapePattern(pattern):

    """Helper function to build our regex"""

    if len(pattern)==1:

        pattern=re.escape(pattern)

    return pattern



def compileCleanerRegex(cleaningOptions):

    """Given a dictionary of rules this contruct the regular expresion to detect the patterns """

    return re.compile("(%s)" % "|".join(map(escapePattern,cleaningOptions.keys())))
def clean_text(text,cleaningOptions,cleaningRegex,removePunct=None):

    """Cleaning function for text

       Given a text this function applies the cleaning rules defined

       in a dictionary using a regex to detect the patterns.

   Args:

       text (str): The text we want to clean.

       cleaningRegex(regex): Regular expression to detect

                                    the patterns defined in the cleaning options

                                    compiled using the compileCleanerRegex(cleaningOptions) function.



    Returns:

        The cleaned text applying the cleaning options.

    """



    #""" REMOVING PUNCTUATIONS ALREADY PERFORMED by KERAS TOKENIZER 

    #removePunct=str.maketrans('','',string.punctuation)

    if removePunct:

        text=text.translate(removePunct)

    return cleaningRegex.sub(lambda mo:cleaningOptions.get(mo.group(1),), text).lower()



def lemmatize(text,spacy_model):

    """Uses a Spacy model to lemmatize the given text"""

    return ' '.join(tok.lemma_ for tok in spacy_model(text))
cleaning_regex=compileCleanerRegex(cleaningOptions)

cregex_sentenceLevel=compileCleanerRegex(co_SentenceLevel)

spacy_nlp =spacy.load('es')
from tqdm.auto import tqdm,trange

def preprocessing(documents,cleaningOptions,cleaningRegex,lemmatizer=None,removePunct=False):

    cleaned_docs=[]

    rmPunc =str.maketrans(string.punctuation,' '*len(string.punctuation))if removePunct else None

    for doc in tqdm(documents,desc='Cleaning Documents'):

        doc=clean_text(doc,cleaningOptions,cleaningRegex,removePunct=rmPunc)

        if lemmatizer:

            doc=lemmatize(doc,spacy_nlp)

        cleaned_docs.append(doc)

    return cleaned_docs
fixed_df.insert(1,'p_motivos',preprocessing(fixed_df.motivos,cleaningOptions,cleaning_regex,lemmatizer=None,removePunct=True))
fixed_df.insert(1,'p_motivos_lemmatized',preprocessing(fixed_df.motivos,cleaningOptions,cleaning_regex,lemmatizer=spacy_nlp,removePunct=True))
fixed_df.insert(1,'p_motivos_sentence_level',preprocessing(fixed_df.motivos,co_SentenceLevel,cregex_sentenceLevel,lemmatizer=None,removePunct=False))
sample=fixed_df.sample()

sample[sample.columns[1:4]]
labels=fixed_df[fixed_df.columns[4:-1]].as_matrix()

labels.shape
texts=fixed_df[fixed_df.columns[0:4]].as_matrix()

texts.shape
from skmultilearn.model_selection import iterative_train_test_split

#I will use this method since we are working with multilabel classification

X_train, y_train, X_validation, y_validation = iterative_train_test_split(texts,labels, test_size = 0.25)

label_names=fixed_df.columns[4:-1]
import matplotlib.pyplot as plt

fig, (ax1, ax2,ax3) =  plt.subplots(1, 3,figsize=(20,6))

fig.suptitle('Class Distributions')

ax1.bar(label_names,labels.sum(axis=0),color='purple')

ax1.set_title("Original Set")

ax2.bar(label_names,y_train.sum(axis=0))

ax2.set_title("Training Set")

ax3.bar(label_names,y_validation.sum(axis=0),color='red')

ax3.set_title("Validation Set")
def exploreVocab(tokenizer,topk=10,lessk=None,delete=False):

    if not lessk:

        lessk=topk

    print('Found {} unique tokens.\n'.format(len(tokenizer.word_index)))

    print("Show the most frequent word index:")

    for i, word in enumerate(sorted(tokenizer.word_counts, key=tokenizer.word_counts.get, reverse=True)):

        print('   {} ({}) --> {}'.format(word, tokenizer.word_counts[word], tokenizer.word_index[word]))

        if delete:

            del tokenizer.index_word[tokenizer.word_index[word]]

            del tokenizer.index_docs[tokenizer.word_index[word]]

            del tokenizer.word_index[word]

            del tokenizer.word_docs[word]

            del tokenizer.word_counts[word]

        if i == topk: 

            print('')

            break

    print("Show the less frequent word index:")

    for i, word in enumerate(sorted(tokenizer.word_counts, key=tokenizer.word_counts.get, reverse=False)):

        print('   {} ({}) --> {}'.format(word, tokenizer.word_counts[word], tokenizer.word_index[word]))

        if delete:

            del tokenizer.index_word[tokenizer.word_index[word]]

            del tokenizer.index_docs[tokenizer.word_index[word]]

            del tokenizer.word_index[word]

            del tokenizer.word_docs[word]

            del tokenizer.word_counts[word]

        if i == lessk: 

            print('')

            break
import tensorflow as tf

tokenizer_lemmatized = tf.keras.preprocessing.text.Tokenizer()

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer_lemmatized.fit_on_texts(X_train[:,2])#Train the tokenizer with the lemmatized docs
exploreVocab(tokenizer_lemmatized,topk=4,lessk=13,delete=True)
print('New Vocab')

exploreVocab(tokenizer_lemmatized,topk=4,lessk=4,delete=False)
X_train_lemmatized=tokenizer_lemmatized.texts_to_sequences(X_train[:,2])

X_val_lemmatized=tokenizer_lemmatized.texts_to_sequences(X_validation[:,2])
fixed_df.columns
tokenizer.fit_on_texts(X_train[:,3])
exploreVocab(tokenizer,topk=8,lessk=16,delete=True)
print("New Vocab")

exploreVocab(tokenizer,topk=8,lessk=10,delete=False)
X_train_preprocessed=tokenizer.texts_to_sequences(X_train[:,3])

X_val_preprocessed=tokenizer.texts_to_sequences(X_validation[:,3])
!pip install laserembeddings

!python -m laserembeddings download-models
from laserembeddings import Laser

laser = Laser()
X_train_sentence= laser.embed_sentences(X_train[:,1],lang='es')

X_val_sentence= laser.embed_sentences(X_validation[:,1],lang='es')
from sklearn.decomposition import PCA

pca = PCA(n_components=4, whiten=True)  # TSNE(n_components=2, n_iter=3000, verbose=2)

pca.fit(np.vstack([X_train_sentence,X_val_sentence]))

print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())
def plot_similar_word(Y,sentences,pca):

    # find tsne coords for 2 dimensions

    Y = pca.transform(Y)

    x_coords = Y[:, 0]

    y_coords = Y[:, 1]



    # display scatter plot

    plt.figure(figsize=(10, 10), dpi=80)

    plt.scatter(x_coords, y_coords, marker='x')



    for k, (sentence,x, y) in enumerate(zip(sentences,x_coords, y_coords)):

        plt.annotate(f"{sentence[:50]}...",xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=8,

                     color='blue' if k <sentences.shape[0]//2 else 'red', weight='bold')

    plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)

    plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)

    plt.title('Visualization of the multilingual word embedding space')

    plt.show()
example_spanish=[X_train_sentence[-5:],X_train[:,0][-5:]]

english=['TO GO SEE MY FAMILY WHICH IS IN PEROTE VERACRUZ ',

            'I request the loan to start a business since I need it to finish putting together the product investment',

            'Hello, good afternoon, I have a snack bar I want to apply for a loan because I want to enlarge my snack bar and because I lack merchandise and I wanted to apply for a loan because we also have a mechanical workshop but the workshop is from my dad',

            'for my business to stock it better to keep moving forward',

            'I have the opportunity to make an investment for my own family business']

englishEm=laser.embed_sentences(english,lang='en')



plot_similar_word(np.vstack([X_train_sentence[-5:],englishEm]),np.concatenate([X_train[:,0][-5:],english],axis=0),pca)
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

parameters = [

    {

        'classifier': [MultinomialNB()],

    },

    {

        'classifier': [LogisticRegression()],

    },

    {

        'classifier': [RandomForestClassifier()],

    },

    {

        'classifier': [MLPClassifier()],

    },

    {

        'classifier': [SGDClassifier()],

        'classifier__penalty':["l2"]

    },

]
from sklearn.metrics import hamming_loss,make_scorer

score_func=make_scorer(hamming_loss,greater_is_better=False)
def norm_data(data):

    return (data - data.min(0)) / data.ptp(0)
from skmultilearn.problem_transform import BinaryRelevance

clf = GridSearchCV(BinaryRelevance(), parameters, scoring=score_func, n_jobs=-1)

clf.fit(norm_data(X_train_sentence),y_train)

print (clf.best_params_, clf.best_score_)
"""

----Citation-----

Taken from kaggle.com/grfiv4/displaying-the-results-of-a-grid-search

"""

def GridSearch_table_plot(grid_clf, param_name, num_results=15, negative=True, graph=True, display_all_params=True):

    '''Display grid search results



    Arguments

    ---------

    grid_clf           the estimator resulting from a grid search

                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display

                       Default: 15

    negative           boolean: should the sign of the score be reversed?

                       scoring = 'neg_log_loss', for instance

                       Default: True

    graph              boolean: should a graph be produced?

                       non-numeric parameters (True/False, None) don't graph well

                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?

                       Default: True

    Usage

    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

    '''

    from matplotlib      import pyplot as plt

    from IPython.display import display

    import pandas as pd



    clf = grid_clf.best_estimator_

    clf_params = grid_clf.best_params_

    if negative:

        clf_score = -grid_clf.best_score_

    else:

        clf_score = grid_clf.best_score_

    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]

    cv_results = grid_clf.cv_results_

    if display_all_params:

        import pprint

        pprint.pprint(clf.get_params())



    # pick out the best results

    # =========================

    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')



    best_row = scores_df.iloc[0, :]

    if negative:

        best_mean = -best_row['mean_test_score']

    else:

        best_mean = best_row['mean_test_score']

    best_stdev = best_row['std_test_score']

    best_param = best_row['param_' + param_name]



    # display the top 'num_results' results

    # =====================================

    display(pd.DataFrame(cv_results) \

            .sort_values(by='rank_test_score').head(num_results))

    

    print("best parameters: {}".format(clf_params))

    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
GridSearch_table_plot(clf,'classifier',  graph=False, display_all_params=False,negative=True)
y_predict_sentence=clf.best_estimator_.predict(norm_data(X_val_sentence))
from sklearn.metrics import classification_report

print(classification_report(y_validation, y_predict_sentence.toarray(),target_names=label_names))
from tensorflow.keras.preprocessing import sequence

max_length=60

X_train_lemmatized=sequence.pad_sequences(X_train_preprocessed,padding='post',maxlen=max_length)

X_train_preprocessed=sequence.pad_sequences(X_train_preprocessed,padding='post',maxlen=max_length)

X_val_lemmatized=sequence.pad_sequences(X_val_lemmatized,padding='post',maxlen=max_length)

X_val_preprocessed=sequence.pad_sequences(X_val_preprocessed,padding='post',maxlen=max_length)
!wget "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec"
def read(file=None,embed_dim=300,threshold=None, vocabulary=None):

    embedding_matrix= np.zeros((max(vocabulary.index_word.keys())+1, embed_dim)) if threshold is None else np.zeros((threshold, embed_dim))

    words_not_found=[]

    matching=[]

    f = codecs.open(file, encoding='utf-8')

    for line in tqdm(f):

        vec = line.rstrip().rsplit(' ')

        word=vec[0].lower()

        if word in vocabulary.word_index:

            try:

                matching.append(word)

                embedding_matrix[vocabulary.word_index[word]]= np.asarray(vec[1:], dtype='float32')

            except:

                words_not_found.append(word)  

        else:

            words_not_found.append(word)      

    f.close()

    return embedding_matrix,words_not_found,matching
embedding_matrix,words_not_found,match= read("wiki.es.vec",vocabulary=tokenizer)

embedding_matrix_lemmatized,words_not_found_lem,match_lem= read("wiki.es.vec",vocabulary=tokenizer_lemmatized)
print(f"Preprocessed Vocabulary:{len(words_not_found)} words missing in the embedding space, {len(match)} words in the embedding space")

print(f"Lemmatized Vocabulary:{len(words_not_found_lem)} words missing in the embedding space, {len(match_lem)} words in the embedding space")
import keras

from keras.layers import *

from keras import Sequential,optimizers,regularizers



class IntentionCNN(Sequential):

    """

    This class extends  keras.sequencial in order to build our 

    model according to the designed architecture

    """

    def __init__(self,max_length,number_of_classes,embedding_matrix=None,vocab_size=None,tokenizer=None):

        #creating the model heritance from Keras.sequencial

        super().__init__()

        self.__weight_decay = 1e-4

        #optimizers

        self.__adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        #params for the embedding layer

        self.__embedding_dim=100 if embedding_matrix is None else embedding_matrix.shape[1]

        #self.__vocab_size=vocab_size if tokenizer is None else tokenizer.word_index.__len__()+1

        self.__vocab_size=vocab_size if tokenizer is None else max(tokenizer.index_word.keys())+1

        try:

            self.__max_length=max_length

            self.__number_of_classes=number_of_classes 

        except NameError as error:

            print("Error ",error," must be defined.")

            

        #defining layers

        #This layer will learn an embedding the vocab_size is the vocabulary learn from our tokenizer

        #the embedding dimension is defined by our selfs in this case we choose a dimension of 100

        #the input length is the maximum length of the documents we will use

        if embedding_matrix is None:

            self.add(Embedding(self.__vocab_size,

                               self.__embedding_dim,

                               input_length=self.__max_length,trainable=True))

        else:

            self.add(Embedding(embedding_matrix.shape[0],

                               embedding_matrix.shape[1],

                               weights=[embedding_matrix],

                               input_length=self.__max_length,

                               trainable=False))

        self.add(BatchNormalization())

        self.add(Conv1D( 8, 3,activation='relu',padding='same'))#output_dim,kernel_size,stride

        self.add(MaxPooling1D(2))

        self.add(BatchNormalization())

        self.add(Conv1D(16, 3,activation='relu',padding='same'))#output_dim,kernel_size,stride

        self.add(MaxPooling1D(2))

        

        self.add(BatchNormalization())

        self.add(Conv1D(32, 3,activation='relu',padding='same'))#output_dim,kernel_size,stride

        self.add(MaxPooling1D(2))

        

        self.add(BatchNormalization())

        self.add(Conv1D(64, 3,activation='relu',padding='same'))#output_dim,kernel_size,stride

        self.add(GlobalMaxPooling1D())

        self.add(BatchNormalization())

        self.add(Sequential([Dense(64),

                             AlphaDropout(0.2),

                             Dense(32),

                             Dense(self.__number_of_classes),

                             Activation('sigmoid')]))

        

        self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.keras import optimizers

model=IntentionCNN(max_length,10,embedding_matrix=embedding_matrix)
model.summary()
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1)

callbacks_list = [early_stopping]
#training params

batch_size = 64

num_epochs = 20
hist = model.fit(X_train_preprocessed, y_train,

                 batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,

                 validation_data=(X_val_preprocessed,y_validation),

                 shuffle=True)
loss, accuracy = model.evaluate(X_val_preprocessed,y_validation, verbose=1)

print('Accuracy: %f' % (accuracy*100),'loss: %f' % (loss*100))
def plot_model_perfomance(hist,name):

    plt.style.use('fivethirtyeight')

    plt.figure(1)

    plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')

    plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')

    plt.title(name)

    plt.xlabel('Epochs')

    plt.ylabel('Cross-Entropy Loss')

    plt.legend(loc='upper right')

    plt.figure(2)

    plt.plot(hist.history['accuracy'], lw=2.0, color='b', label='train')

    plt.plot(hist.history['val_accuracy'], lw=2.0, color='r', label='val')

    plt.title(name)

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper left')

    plt.show()
plot_model_perfomance(hist,'Intention CNN')
IntCNNPredict=model.predict(X_val_preprocessed)
def get_label_vec(prediction):

    return np.where(prediction > 0.5, 1.0, 0.0)
IntCNNPredict=get_label_vec(IntCNNPredict)
print(classification_report(y_validation, IntCNNPredict,target_names=label_names))
model=IntentionCNN(max_length,10,embedding_matrix=embedding_matrix_lemmatized)
hist = model.fit(X_train_lemmatized, y_train,

                 batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,

                 validation_data=(X_val_lemmatized,y_validation),

                 shuffle=True)
plot_model_perfomance(hist,'Intention CNN')
lemmatize_predict=model.predict(X_val_lemmatized)
lemmatize_predict=get_label_vec(lemmatize_predict)
print(classification_report(y_validation, lemmatize_predict,target_names=label_names))
sentences=laser.embed_sentences(['I want to make an investment for my own family business',

                                 'I want a new tv'],lang='en')
probs=clf.best_estimator_.predict(norm_data(sentences))
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

mlb.fit([label_names])

mlb.classes

mlb.inverse_transform(probs.toarray())
sentences_es=laser.embed_sentences(['Quiero pagar la colegiatura de la escuela y comprar una computadora',

                                 'Para realizar una inversion en mi negocio de transporte'],lang='es')

probs_es=clf.best_estimator_.predict(norm_data(sentences_es))

mlb.inverse_transform(probs_es.toarray())
import pandas as pd

intent = pd.read_csv("../input/intention/intent.csv")

intent_fixed = pd.read_csv("../input/intention/intent_fixed.csv")