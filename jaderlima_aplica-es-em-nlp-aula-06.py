!pip install sklearn_crfsuite
!pip install eli5
import nltk

import sklearn_crfsuite

import eli5
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))

test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
import pprint



pprint.pprint(len(train_sents))

pprint.pprint(len(test_sents))



pprint.pprint(train_sents[2])

pprint.pprint(test_sents[2])
def word2features(sent, i):

    word = sent[i][0]

    postag = sent[i][1]



    features = {

        'bias': 1.0,

        'word.lower()': word.lower(),

        'word[-3:]': word[-3:],

        'word.isupper()': word.isupper(),

        'word.istitle()': word.istitle(),

        'word.isdigit()': word.isdigit(),

        'postag': postag,

        'postag[:2]': postag[:2],

        'hasnumbers': any(char.isdigit() for char in word)

    }

    if i > 0:

        word1 = sent[i-1][0]

        postag1 = sent[i-1][1]

        features.update({

            '-1:word.lower()': word1.lower(),

            '-1:word.istitle()': word1.istitle(),

            '-1:word.isupper()': word1.isupper(),

            '-1:postag': postag1,

            '-1:postag[:2]': postag1[:2],

            '-1:hasnumbers': any(char.isdigit() for char in word1)

        })

    else:

        features['BOS'] = True



    if i < len(sent)-1:

        word1 = sent[i+1][0]

        postag1 = sent[i+1][1]

        features.update({

            '+1:word.lower()': word1.lower(),

            '+1:word.istitle()': word1.istitle(),

            '+1:word.isupper()': word1.isupper(),

            '+1:postag': postag1,

            '+1:postag[:2]': postag1[:2],

            '+1:hasnumbers': any(char.isdigit() for char in word1)

        })

    else:

        features['EOS'] = True



    return features

def sent2features(sent):

    return [word2features(sent, i) for i in range(len(sent))]



def sent2labels(sent):

    return [label for token, postag, label in sent]



def sent2tokens(sent):

    return [token for token, postag, label in sent]



X_train = [sent2features(s) for s in train_sents]

y_train = [sent2labels(s) for s in train_sents]



X_test = [sent2features(s) for s in test_sents]

y_test = [sent2labels(s) for s in test_sents]
X_train[0][1]
crf = sklearn_crfsuite.CRF(

    algorithm='lbfgs',

    c1=0.1,

    c2=0.1,

    max_iterations=20,

    all_possible_states=False # Replica os atributos para cada categoria. 

                              # Isso permite aumentar a performance para categorias específicas, mas deixa o treino mais lento. 

                              # Não vamos usar neste exemplo.

)

crf.fit(X_train, y_train);

from sklearn_crfsuite import metrics



labels = list(crf.classes_)

labels.remove('O')

labels



y_pred = crf.predict(X_test)

metrics.flat_f1_score(y_test, y_pred,

                      average='weighted', labels=labels)
# Agrupandos resultados de B e I

sorted_labels = sorted(

    labels,

    key=lambda name: (name[1:], name[0])

)

print(metrics.flat_classification_report(

    y_test, y_pred, labels=sorted_labels, digits=3

))
eli5.show_weights(crf, top=30)
import pickle

import os



path = "../input"



x_train = pickle.load(open(os.path.join(path, 'football_ner_x_train.pkl'), 'rb'), encoding='UTF-8',)

y_train = pickle.load(open(os.path.join(path, 'football_ner_y_train.pkl'), 'rb'),encoding='UTF-8')

        

x_test = pickle.load(open(os.path.join(path, 'football_ner_x_test.pkl'), 'rb'),encoding='UTF-8')

y_test = pickle.load(open(os.path.join(path, 'football_ner_y_test.pkl'), 'rb'),encoding='UTF-8')


print(x_train[:10])

print(y_train[:10])
print(x_train[:30][2])

print(y_train[:30][1])
from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import wordnet



stopwords_list = stopwords.words('english')



lemmatizer = WordNetLemmatizer()



x_posttag_train = []

for i in x_train:

    x_posttag_train.append(pos_tag(i))



    

x_posttag_test = []

for i in x_test:

    x_posttag_test.append(pos_tag(i))

def sent2features(sent):

    return [word2features(sent, i) for i in range(len(sent))]



def sent2labels(sent):

    return [label for token, postag, label in sent]



def sent2tokens(sent):

    return [token for token, postag, label in sent]





X_train = [sent2features(s) for s in x_posttag_train]

#y_train = y_train



X_test = [sent2features(s) for s in x_posttag_test]

#y_test = y_test

y_train2 =[['O' if y==0 else 'I' for y in y_train_] for y_train_ in y_train ]

y_test2 =[['O' if y==0 else 'I' for y in y_test_] for y_test_ in y_test ]
crf = sklearn_crfsuite.CRF(

    algorithm='lbfgs',

    c1=0.1,

    c2=0.1,

    max_iterations=20,

    all_possible_states=False # Replica os atributos para cada categoria. 

                              # Isso permite aumentar a performance para categorias específicas, mas deixa o treino mais lento. 

                              # Não vamos usar neste exemplo.

)

crf.fit(X_train, y_train2);

from sklearn_crfsuite import metrics



labels = list(crf.classes_)

labels.remove('O')

labels



y_pred = crf.predict(X_test)

metrics.flat_f1_score(y_test2, y_pred,average='weighted', labels=labels)

# Agrupandos resultados de B e I

sorted_labels = sorted(

    labels,

    key=lambda name: (name[1:], name[0])

)

print(metrics.flat_classification_report(

    y_test2, y_pred, labels=sorted_labels, digits=3

))
eli5.show_weights(crf, top=30)