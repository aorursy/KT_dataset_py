import numpy as np 

import pandas as pd

import string

from tqdm import tqdm

import math

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import  hstack

import nltk

import re

_wnl = nltk.WordNetLemmatizer()

from sklearn import feature_extraction

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Dropout

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

def normalize_word(w):

    return _wnl.lemmatize(w).lower()





def get_tokenized_lemmas(s):

    return [normalize_word(t) for t in nltk.word_tokenize(s)]





def clean(text):

    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation),''))

    return " ".join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()





def remove_stopwords(l):

    # Removes stopwords from a list of tokens

    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def join_tok(text):

    return " ".join(text).lower()
def pre_process(texts):

    lst=[]

    for text in tqdm(texts):

        clean_text= clean(text)

        tok_text= get_tokenized_lemmas(clean_text)

        remov_stp= remove_stopwords(tok_text)

        lst.append(join_tok(remov_stp))

    return lst
data= pd.read_csv('../input/aossie-click-bait-dataset/clickBait_Data.csv')
data.head()
titles=  pre_process(data['titles'])
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=300, stop_words='english',ngram_range=(1,4))

tfidf_vect.fit(titles)

titleFeatures= tfidf_vect.transform(titles).toarray()
labels= data['clickbait'].tolist()
train_feat, test_feat, train_labels, test_labels= train_test_split(titleFeatures,labels,test_size= 0.2,shuffle=True)


from sklearn.naive_bayes import MultinomialNB



mnb = MultinomialNB()
mnb.fit(train_feat,train_labels)
prediction= mnb.predict(test_feat)
accuracy_score(test_labels,prediction)
featureLen= train_feat.shape[1]
model= Sequential()

model.add(Dense(300,activation='relu', input_shape=(featureLen,)))

model.add(Dense(160,activation='relu'))

model.add(Dense(80,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(20,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

filepath=r"CB1-tfidf_nn_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

callbacks_list = [checkpoint]
model.compile(loss = 'mse', optimizer='adam',metrics = ['accuracy'])

model.summary()
model.fit(train_feat,train_labels,epochs=25,batch_size=3,verbose=1,validation_split=0.05, shuffle=True,callbacks=callbacks_list)
prediction= model.predict(test_feat)

YClass= np.zeros((len(prediction)))



acc,scor= model.evaluate(test_feat,test_labels)
acc, scor
for i in range(len(prediction)):

    if prediction[i][0]>=0.5:

        YClass[i]=1

    else:

        YClass[i]=0
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()

    

#matrix1 = confusion_matrix(test_labels, prediction)



matrix1 = confusion_matrix(test_labels, YClass)

plot_confusion_matrix(cm=matrix1,target_names=['Non_ClickBait', 'ClickBait'])