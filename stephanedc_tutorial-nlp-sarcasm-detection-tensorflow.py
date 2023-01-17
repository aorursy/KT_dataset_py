###########################

# Configuration Principale

# #########################



vocab_size= 22000        # Taille maximale du corpus de vocabulaire du jeu de données

embedding_dim=32         # Nombre d'embedding dimensions permettant de qualifier un mot

max_length = 18          # Longueur maximale en nombre de mots des titres avant de tronquer

trunc_type='post'        # Méthode pour trunquer si dépassement (avant ou après)

padding_type='post'      # Paramètre du padding (après la phrase)

oov_tok='<OOV>'          # Token à utiliser quand un mot est manquant

training_size=22000      # Taille du training Set (pour le split entre train/test)

num_epochs = 10          # Nombre epoch pour entrainement du réseau

DEL_STOPWORDS = False    # Pour la suppression des Stop Words

DO_LEMMATIZE  = True     # Lemmatization des mots

PATIENCE_STOPPING = 5    # Patience en nombre d'epoch avant de stopper l'entrainement

OPTIMIZER = 'adam'       # Optimizer pour la descente de gradient (optimisation fonction coût)
# Lib Standards

import os

import io



# Désactiver les avertissements

import warnings

warnings.filterwarnings("ignore", category=FutureWarning) 



# Verification de l'import des Data

print("Les données importées dans Kaggle: ")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import des Librairies Datascience

import numpy as np 

import pandas as pd       

import random

import nltk



# Import TensorFlow et vérification de la version

import tensorflow as tf

print("Version de TensorFlow: {}".format(tf.__version__))

# Version majeure:

vers_tf = int((tf.__version__).split(sep='.')[0])



if vers_tf < 2:

    # eager_execution nécessaire si TensorFlow < 2.0

    print('Version < 2 : activation eager execution')

    tf.enable_eager_execution()



# Import des librairies de NLP

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.text import text_to_word_sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint





# Librairies nltk pour la tokenization et la Lemmatization

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize
# Toujours frâce à la librairie Panda nous procédons à l'ouverture du fichier de data

# Ouverture du fichier de données contenant les titres

df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)



# Pour voir le nom des colonnes et la taille des objects du dataset importé

# Objet dataframe, nous observons grâce à cette sortie : 

#     Les colonnes du DataFrame

#     leur Type

#     Le nombre d'enregistrements

df.info()
# On examine les premières lignes du dataset

df.head()
# denombrage du nombre articles classifié sarcastic et des autres 

nb_sarcastic = (df['is_sarcastic'] == 1).sum()

nb_not_sarcastic = (df['is_sarcastic'] == 0).sum()

print("Il y a {} titres sarcastiques vs {} qui ne le sont pas".format(nb_sarcastic, nb_not_sarcastic))
# Affichage au hasard de quelques titres et de leur label

for i in range(3):

    n = random.randrange(len(df['headline']))

    print(df['article_link'][n])

    print(df['headline'][n])

    print("Sarcasme: {}".format(df['is_sarcastic'][n]))

    print("---")
# Contruction des listes de phrases(X) et de labels(y)

X=list(df['headline'])

y=list(df['is_sarcastic'])
# Import des stopwords grace à nltk

stopwords = nltk.corpus.stopwords.words('english')

print("Nous avons {} stopwords pour le language anglais. Exemple:".format(len(stopwords)))

print(stopwords[0:50])
# Fonction pour gérer la suppression des stopwords

def process_stop_words(X, display_res=True):

    

    if DEL_STOPWORDS:

        # Une autre façon de Tokeniser:

        # avec fonctions lambda + fonction map

        # tokenize = lambda x: text_to_word_sequence(x, 

        #                                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 

        #                                    lower=True, split=' ')

        # X_seq = list(map(tokenize, X))

        

        # avec la liste comprehension et la librairie NLTK:

        X_seq = [[word for word in word_tokenize(s)] for s in X]

        

        # Suppression des Stopwords 

        X_seq_no_stops = [[word for word in s if word not in stopwords] for s in X_seq]

        

        if display_res:

            print("Avant Tokenization:     {}".format(X[10]))

            print("Après Tokenization:     {}".format(X_seq[10]))

            print("Sans Stopwords:         {}".format(X_seq_no_stops[10]))

        return X_seq_no_stops

    else:

        print("Pas de traitement stopwords")

        return X
# ---------------------------------------------------

# Gain apporté par les 'list comprehension' de Python

# ---------------------------------------------------



# Grâce a une seule ligne ci-dessous...

#[[word for word in s if word not in stopwords] for s in X_seq]



# ...factoriser le code suivant:



# X_no_stops = [[0] * 0 for i in range(len(X))]

# for n,title in enumerate(X):

#    for word in title:

#        if word not in stopwords:

#           X_no_stops[n].append(word)
# Processing des Stop Words

X = process_stop_words(X)
print("Exemple de Lemme:")

list_lemm_exemple = ['cars', 'tools', 'airplane', 'nicely', 'sarcastic']

for w in list_lemm_exemple:

    print("    {}   =>   {}".format(w, WordNetLemmatizer().lemmatize(w)))
def process_lemmatization(X, display_res=True):

    if DO_LEMMATIZE:

        lemm = WordNetLemmatizer()

        if display_res:

            print("Avant Lemmatization :    {}".format(X[10]))

        Xlem = X.copy()



        # Si nous avons une liste de liste 

        # suite à la tokenization des titres pour suppression des stopword

        

        if DEL_STOPWORDS:

            # Ancienne méthode avec parcourt de liste

            # for row,title in enumerate(X):

            #    for col,word in enumerate(title):

            #        Xlem[row][col] = lemm.lemmatize(word)

            

            # Plus simple Avec les génerateurs de liste :-)

            Xlem = [[lemm.lemmatize(word) for word in s] for s in X]



        # une liste de titre

        else:

            Xlem = [[lemm.lemmatize(word) for word in word_tokenize(s)] for s in X]



        if display_res:

            print("Apres Lemmatization :    {}".format(Xlem[10]))



        return Xlem

    else:

        print("Pas de lemmatization demandée")

        return X
# Traitement de la lemmatization

X = process_lemmatization(X)
# Constitution des jeux d'entrainements et de test

X_train = X[:training_size]

y_train = y[:training_size]

X_test  = X[training_size:]

y_test  = y[training_size:]



print("Taille jeu d'entrainement : {}".format(len(X_train)))

print("Taille jeu de test : {}".format(len(X_test)))
# Instanciation du Tokenizer avec les paramètres

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

# Adaptation du Tokenizer au jeu d'entrainement

tokenizer.fit_on_texts(X_train)

# Création d'un dictionnaire d'index des mots

word_index = tokenizer.word_index                             



# Conversion en sequences d'entier

X_train_sequences = tokenizer.texts_to_sequences(X_train)



# Ajout du padding 

# Objectif: permettre à chaque séquence de possèder la même longueur

X_train_padded = pad_sequences(X_train_sequences, maxlen = max_length, truncating = trunc_type)
# Traitement du jeu de test à l'identique du jeu d'entrainement

X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_test_padded = pad_sequences(X_test_sequences, maxlen = max_length, truncating = trunc_type)
# Nombre de mot dans le corpus

print("Nombre de mot dans le corpus du jeu d'entrainement : {}".format(len(word_index)))
# Création d'un index inversé des mots / digit associé

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])



# Affichage de quelques phrases aléatoires:

for i in range(3):

    alea = random.randrange(len(X_train))

    print("phrase originale: {}".format(X_train[alea]))

    print("sequence : {}".format(X_train_sequences[alea]))

    print("sequence + bourrage: {}".format(X_train_padded[alea]))

    print("reconstitution: {}".format(decode_review(X_train_padded[alea])))

    print("----------------------------------------------------------------------")
# Création du réseau neuronal avec LSTM

model = tf.keras.Sequential([

    

    ### Couche d'entrée

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    

    ### Couche des LSTM

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),

    

    ### Couche de neurones connectés

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    

    ### Couche de Sortie

    tf.keras.layers.Dense(1, activation='sigmoid')

    ])



model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['acc'])

model.summary()
# Callbacks (Expliqué dans les tutoriaux CNN)

checkpoint = ModelCheckpoint("model_sarcasm.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE_STOPPING, verbose=1, mode='auto')
### Entrainement du modèle

# NB : il est possible d'inclure le paramètre validation_split dans fit() 

# afin de ne pas traiter le split train/test plus tôt



history=model.fit(X_train_padded,

                  y_train,

                  epochs=num_epochs,

                  validation_data=(X_test_padded, y_test),

                  callbacks = [checkpoint, early],

                  verbose=1)
# Graphing de l'apprentissage



import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

fig.suptitle("Performance")

ax1.plot(history.history['acc'])

ax1.plot(history.history['val_acc'])

vline_cut = np.where(history.history['val_acc'] == np.max(history.history['val_acc']))[0][0]

ax1.axvline(x=vline_cut, color='k', linestyle='--')

ax1.set_title("Model Accuracy")

ax1.legend(['train', 'test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

ax2.axvline(x=vline_cut, color='k', linestyle='--')

ax2.set_title("Model Loss")

ax2.legend(['train', 'test'])

plt.show()
# Test sur deux phrases de présence ou nom de saracasme

sentence = ["granny starting to fear spiders in the garden might be real!", "Doctor House season finale this sunday evening on TV."]

sentence = process_stop_words(sentence, display_res=False)

sentence = process_lemmatization(sentence, display_res=False)
sequences = tokenizer.texts_to_sequences(sentence)

padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print("Probabilité de détection du sarcasme: \n")

print(sentence)

print(model.predict(padded))