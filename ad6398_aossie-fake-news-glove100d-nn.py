import numpy as np 

import pandas as pd

import string

from tqdm import tqdm

import math



from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import  hstack

from keras.layers import Input, concatenate,LSTM,Flatten, Dense,Dropout

from keras.models import Model

from keras.callbacks import ModelCheckpoint 

# from handFeatures import hand_features, clean, get_tokenized_lemmas, remove_stopwords

# from glove50_embedding import glove_sentence_embeddings
# run and tested

import os

import re

import nltk

import numpy as np

from sklearn import feature_extraction

from tqdm import tqdm





_wnl = nltk.WordNetLemmatizer()





def normalize_word(w):

    return _wnl.lemmatize(w).lower()





def get_tokenized_lemmas(s):

    return [normalize_word(t) for t in nltk.word_tokenize(s)]





def clean(s):

    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()





def remove_stopwords(l):

    # Removes stopwords from a list of tokens

    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]





def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):

    if not os.path.isfile(feature_file):

        feats = feat_fn(headlines, bodies)

        np.save(feature_file, feats)



    return np.load(feature_file)









def word_overlap_features(features, headline, body):

    # common word/ total word

    clean_headline = clean(headline)

    clean_body = clean(body)

    clean_headline = get_tokenized_lemmas(clean_headline)

    clean_body = get_tokenized_lemmas(clean_body)

    feature = len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))

    features.append(feature)

    return features





def refuting_features(features,headline, body):

    _refuting_words = [

        'fake',

        'fraud',

        'hoax',

        'false',

        'deny', 'denies',

        # 'refute',

        'not',

        'despite',

        'nope',

        'doubt', 'doubts',

        'bogus',

        'debunk',

        'pranks',

        'retract'

    ]

    X = []

    clean_headline = clean(headline)

    clean_headline = get_tokenized_lemmas(clean_headline)

    feature=0

    for wrd in clean_headline:

        if wrd in _refuting_words:

            feature=1

    features.append(feature)

    return features





def polarity_features(features,headline, body):

    _refuting_words = [

        'fake',

        'fraud',

        'hoax',

        'false',

        'deny', 'denies',

        'not',

        'despite',

        'nope',

        'doubt', 'doubts',

        'bogus',

        'debunk',

        'pranks',

        'retract'

    ]



    def calculate_polarity(text):

        tokens = get_tokenized_lemmas(text)

        return sum([t in _refuting_words for t in tokens]) % 2

    X = []

    clean_headline = clean(headline)

    clean_body = clean(body)

    features.append(calculate_polarity(clean_headline))

    features.append(calculate_polarity(clean_body))

    return features



def ngrams(input, n):

    input = input.split(' ')

    output = []

    for i in range(len(input) - n + 1):

        output.append(input[i:i + n])

    return output





def chargrams(input, n):

    output = []

    for i in range(len(input) - n + 1):

        output.append(input[i:i + n])

    return output





def append_chargrams(features, text_headline, text_body, size):

    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]

    grams_hits = 0

    grams_early_hits = 0

    grams_first_hits = 0

    for gram in grams:

        if gram in text_body:

            grams_hits += 1

        if gram in text_body[:255]:

            grams_early_hits += 1

        if gram in text_body[:100]:

            grams_first_hits += 1

    features.append(grams_hits)

    features.append(grams_early_hits)

    features.append(grams_first_hits)

    return features





def append_ngrams(features, text_headline, text_body, size):

    grams = [' '.join(x) for x in ngrams(text_headline, size)]

    grams_hits = 0

    grams_early_hits = 0

    for gram in grams:

        if gram in text_body:

            grams_hits += 1

        if gram in text_body[:255]:

            grams_early_hits += 1

    features.append(grams_hits)

    features.append(grams_early_hits)

    return features





def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):

        # Count how many times a token in the title

        # appears in the body text.

        bin_count = 0

        bin_count_early = 0

        for headline_token in clean(headline).split(" "):

            if headline_token in clean(body):

                bin_count += 1

            if headline_token in clean(body)[:255]:

                bin_count_early += 1

        return [bin_count, bin_count_early]



    def binary_co_occurence_stops(headline, body):

        # Count how many times a token in the title

        # appears in the body text. Stopwords in the title

        # are ignored.

        bin_count = 0

        bin_count_early = 0

        for headline_token in remove_stopwords(clean(headline).split(" ")):

            if headline_token in clean(body):

                bin_count += 1

                bin_count_early += 1

        return [bin_count, bin_count_early]



    def count_grams(headline, body):

        # Count how many times an n-gram of the title

        # appears in the entire body, and intro paragraph



        clean_body = clean(body)

        clean_headline = clean(headline)

        features = []

        features = append_chargrams(features, clean_headline, clean_body, 2)

        features = append_chargrams(features, clean_headline, clean_body, 8)

        features = append_chargrams(features, clean_headline, clean_body, 4)

        features = append_chargrams(features, clean_headline, clean_body, 16)

        features = append_ngrams(features, clean_headline, clean_body, 2)

        features = append_ngrams(features, clean_headline, clean_body, 3)

        features = append_ngrams(features, clean_headline, clean_body, 4)

        features = append_ngrams(features, clean_headline, clean_body, 5)

        features = append_ngrams(features, clean_headline, clean_body, 6)

        return features



    def other_feature(headline, body):

        features=[]

        features= word_overlap_features(features, headline, body)

        features= refuting_features(features, headline, body)

        features= polarity_features(features,headline,body)

        return features

    X = []

    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):

        lst= binary_co_occurence(headline, body) + binary_co_occurence_stops(headline, body) + count_grams(headline, body)+ other_feature(headline, body)

        X.append(lst)





    X =np.array(X)

    return X









# x= hand_features(h,t)



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

#plot_confusion_matrix(cm=matrix1,target_names=['agree', 'disagree', 'discuss', 'unrelated'])

# run and tested

#this script returns Google's word2vec embedding of texts

from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import string

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

import numpy as np

from tqdm import tqdm

filename = '../input/glove.6B.100d.txt'

glove= dict()

f = open(filename,encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    glove[word] = coefs

f.close()



def pre_process(text):

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    return text



def make_embed(text):

    feature= np.zeros((len(text),100),dtype= object)

    c=0

    for wrd in text:

        if wrd in glove:

            feature[c]=glove[wrd].reshape(1,100)

            c=c+1

    res= np.mean(feature,axis=0)

    return res



def glove_sentence_embeddings(text):

    embed=[]

    for i in tqdm(range(len(text))):

        clean_text= pre_process(text[i])

        embed.append(make_embed(clean_text))

    return np.array(embed, dtype= 'float')

trainFile=pd.read_csv('../input/train_merged.csv',encoding='latin-1')

testFile=pd.read_csv('../input/test_merged.csv', encoding='latin-1')
trainBody= trainFile['articleBody'].tolist()

trainHead= trainFile['Headline'].tolist()

testBody= testFile['articleBody'].tolist()

testHead= testFile['Headline'].tolist()

train_handF= hand_features(trainHead, trainBody)

test_handF=hand_features(testHead, testBody)
trainBodyFeat= glove_sentence_embeddings(trainBody)

trainHeadFeat= glove_sentence_embeddings(trainHead)

testBodyFeat= glove_sentence_embeddings(testBody)

testHeadFeat= glove_sentence_embeddings(testHead)
np.save('glove100_trainBodyFeat',trainBodyFeat)

np.save('glove100_trainHeadFeat',trainHeadFeat)

np.save('glove100_testBodyFeat',testBodyFeat)

np.save('glove100_testHeadFeat',testHeadFeat)
trainFeat= np.concatenate((trainHeadFeat, trainBodyFeat, train_handF), axis=-1)

testFeat= np.concatenate((testHeadFeat, testBodyFeat, test_handF), axis=-1)
np.save('trainFeatMatrix_glove_100',trainFeat)

np.save('testFeatMatrix_glove_100',testFeat)
print(trainFeat.shape,testFeat.shape)
train_labels= trainFile['Stance'].copy()

test_labels= testFile['Stance'].copy()
inp = Input(shape=(trainFeat.shape[1],))

lay1= Dense(200, activation= 'relu')(inp)

lay1= Dropout(0.3)(lay1)

lay2= Dense(80, activation= 'relu')(lay1)

outp= Dense(4,activation='sigmoid')(lay2)

model= Model(inputs=[inp], outputs=[outp])

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.summary()
model_json = model.to_json()

with open("model_FNC.json", "w") as json_file:

    json_file.write(model_json)
filepath=r"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

callbacks_list = [checkpoint]
trainY = pd.get_dummies(trainFile['Stance']).values

testY = pd.get_dummies(testFile['Stance'] ).values
model.fit([trainFeat],[trainY],epochs=5,batch_size=13,verbose=1, validation_split=0.1, callbacks=callbacks_list, shuffle=True)
scor, acc = model.evaluate([testFeat],[testY])
acc,scor