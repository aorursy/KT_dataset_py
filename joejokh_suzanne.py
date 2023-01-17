# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout, TimeDistributed

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras.preprocessing import sequence

import keras





import nltk

import itertools

import pickle



import numpy as np

import pickle as cPickle

import os

import time

import random

import gc



print(os.listdir("../input/inputs"))

class Chatbot():

    # params

    WORD2VEC_DIMS = 300

    ENCODER_UNITS = 1024

    DECODER_UNITS = 2048



    DICTIONARY_SIZE = 6000

    MAX_INPUT_LENGTH = 20

    MAX_OUTPUT_LENGTH = 35



    BATCH_SIZE = 128  # 4-50 49min, 0.12E / 128-50 1 min

    NUM_EPOCHS = 100



    NUM_SUBSETS = 1



    Max_samples = 3000



    unknown_token = 'quelquechose'



    BOS = 0  # la position du symbole BOS

    EOS = 1  # la position du symbole EOS



    # fichies

    In_Dir = "../input/inputs/"

    Out_Dir = ""#../input/suzanne/



    out_vocabulary_file = Out_Dir + 'vocabulaire'

    in_vocabulary_file = In_Dir + 'vocabulaire'



    input_questions_file = In_Dir + 'Q_FR_text.txt'

    input_answers_file = In_Dir + 'R_FR_text.txt'



    input_padded_questions_file = In_Dir + 'Padded_questions'

    input_padded_answers_file = In_Dir + 'Padded_reponses'



    output_padded_questions_file = Out_Dir + 'Padded_questions'

    output_padded_answers_file = Out_Dir + 'Padded_reponses'



    input_modele_file = In_Dir + 'modele'

    output_modele_file = Out_Dir + 'modele'



    GLOVE_FILE = In_Dir+"multilingual_embeddings.fr"

    

    def __init__(self, mode="chat", save=False, ep = 500 , in_modele = None):

        self.NUM_EPOCHS = ep

        if mode == "train":

            self.Out_Dir = ""

            try:

                # préparer la base d'apprentissage

                self.Q, self.A, self.vocab, self.word_to_index, self.index_to_word = self.prepare_db()

                self.BOS = self.word_to_index["BOS"]

                self.EOS = self.word_to_index["EOS"]

                self.word2vec_index, self.word_embedding_matrix = self.indexing_vocab()

                # lancer l'apprentissage

                self.modele = self.train(save)

            except Exception as e:

                print("error in train")

                print(str(e))

        elif mode == "chat":

            try:

                self.input_modele_file = self.Out_Dir + in_modele

                # importer la base d'apprentissage

                self.Q, self.A, self.vocab, self.word_to_index, self.index_to_word = self.read_db()

                self.BOS = self.word_to_index["BOS"]

                self.EOS = self.word_to_index["EOS"]

                self.word2vec_index, self.word_embedding_matrix = self.indexing_vocab()

                # importer le modele

                self.modele = self.read_model()

            except Exception as e:

                print("error in chat")

                print(str(e))



    def read_db(self):

        Q = cPickle.load(open(self.output_padded_questions_file, 'rb'))

        A = cPickle.load(open(self.output_padded_answers_file, 'rb'))

        vocab = pickle.load(open(self.out_vocabulary_file, 'rb'))

        index_to_word = [x[0] for x in vocab]

        index_to_word.append(self.unknown_token)

        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        return Q, A, vocab, word_to_index, index_to_word



    def read_model(self):

        return load_model(self.input_modele_file)



    def prepare_db(self):

        # BD

        print("Lecture des questions...")

        q = open(self.input_questions_file, 'r')

        questions = q.read()



        print("Lecture des reponses...")

        a = open(self.input_answers_file, 'r')

        answers = a.read()



        all = answers + questions

        print("Tokeniser les reponses...")

        paragraphs_a = [p.lower() for p in answers.split('\n')[0:self.Max_samples]]

        paragraphs_b = [p.lower() for p in all.split('\n')[0:self.Max_samples]]

        paragraphs_a = ['BOS ' + p + ' EOS' for p in paragraphs_a]

        paragraphs_b = ['BOS ' + p + ' EOS' for p in paragraphs_b]

        paragraphs_b = ' '.join(paragraphs_b)

        tokenized_text = paragraphs_b.split()

        paragraphs_q = [p.lower() for p in questions.split('\n')[0:self.Max_samples]]

        tokenized_answers = [p.split() for p in paragraphs_a]

        tokenized_questions = [p.split() for p in paragraphs_q]



        # Préparer et Enregitrer le vocabulaire



        ### calculer les frequences des mots

        word_freq = nltk.FreqDist(itertools.chain(tokenized_text))

        print(" %d mots uniques trouver" % len(word_freq.items()))



        ##

        ### Garder que les mots les plus utiliser au cas du dépacement de la taille max du dicionnaire

        vocab = word_freq.most_common(self.DICTIONARY_SIZE - 1)

        if len(word_freq.items()) < self.DICTIONARY_SIZE:

            self.DICTIONARY_SIZE = len(word_freq.items()) + 1

        print(self.DICTIONARY_SIZE)

        ##

        ### Enregitrer le vocabulaire:

        with open(self.out_vocabulary_file, 'wb') as v:

            pickle.dump(vocab, v)



        index_to_word = [x[0] for x in vocab]

        index_to_word.append(self.unknown_token)

        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])



        # Replacer les mots inconu par quelquechose

        for i, sent in enumerate(tokenized_answers):

            tokenized_answers[i] = [w if w in word_to_index else self.unknown_token for w in sent]



        for i, sent in enumerate(tokenized_questions):

            tokenized_questions[i] = [w if w in word_to_index else self.unknown_token for w in sent]



        # Creations de la base d'apprentissage

        X = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_questions])

        Y = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_answers])



        Q = sequence.pad_sequences(X, maxlen=self.MAX_INPUT_LENGTH)

        A = sequence.pad_sequences(Y, maxlen=self.MAX_OUTPUT_LENGTH, padding='post')





        with open(self.output_padded_questions_file, 'wb') as q:

            pickle.dump(Q, q)



        with open(self.output_padded_answers_file, 'wb') as a:

            pickle.dump(A, a)



        return Q, A, vocab, word_to_index, index_to_word



    def indexing_vocab(self):

        # ======================================================================

        # Lire un modele pré-formé incorporant des mots et adaptant à notre vocabulaire:

        # ======================================================================



        # Fabrication de dictionnaire

        word2vec_index = {}

        f = open(self.GLOVE_FILE, encoding="utf8")

        for line in f:

            words = line.split()

            word = words[0]

            index = np.asarray(words[1:], dtype="float32")

            word2vec_index[word] = index

        f.close()

        print("Le nombre des mots vectoriser dans le modele pré-formé est :", len(word2vec_index))



        word_embedding_matrix = np.zeros((self.DICTIONARY_SIZE, self.WORD2VEC_DIMS))

        i = 0

        for word in self.vocab:

            word2vec = word2vec_index.get(word[0])

            if word2vec is not None:

                word_embedding_matrix[i] = word2vec

            i += 1

        return word2vec_index, word_embedding_matrix



    def cree_model(self):



        """

        Input Layer #Document*2

        """

        input_context = Input(shape=(self.MAX_INPUT_LENGTH,), dtype="int32", name="input_context")

        input_answer = Input(shape=(self.MAX_OUTPUT_LENGTH,), dtype="int32", name="input_answer")



        """

        Embedding Layer: 

        """



        Shared_Embedding = Embedding(input_dim=self.DICTIONARY_SIZE, output_dim=self.WORD2VEC_DIMS,

                                     input_length=self.MAX_INPUT_LENGTH, weights=[self.word_embedding_matrix],

                                     trainable=False)



        Shared_Embedding2 = Embedding(input_dim=self.DICTIONARY_SIZE, output_dim=self.WORD2VEC_DIMS,

                                      input_length=self.MAX_OUTPUT_LENGTH, weights=[self.word_embedding_matrix],

                                      trainable=False)



        """

        Shared Embedding Layer #Doc2Vec(Document*2)

        """

        shared_embedding_context = Shared_Embedding(input_context)

        shared_embedding_answer = Shared_Embedding2(input_answer)



        """

        LSTM Layer #

        """

        Encoder_LSTM = LSTM(units=self.ENCODER_UNITS, use_bias=True, recurrent_dropout=0.15)

        Decoder_LSTM = LSTM(units=self.DECODER_UNITS, use_bias=True, recurrent_dropout=0.15)



        embedding_context = Encoder_LSTM(shared_embedding_context)

        embedding_answer = Decoder_LSTM(shared_embedding_answer)



        """

        Merge Layer #

        """

        merge_layer = concatenate([embedding_context, embedding_answer])



        """

        Dense Layer #

        """

        #dence_layer = Dense(int(self.DICTIONARY_SIZE * 2), activation="relu")(merge_layer)



        """

        Output Layer #

        """

        outputs = Dense(self.DICTIONARY_SIZE, activation="softmax")(merge_layer)



        """

        Modele

        """

        model = Model(inputs=[input_context, input_answer], outputs=[outputs])

        model.compile(loss="categorical_crossentropy", optimizer="adam")

        model.summary()

        return model



    def train(self, save=False):

        

        model = self.cree_model()

        N_SAMPLES, N_WORDS = self.A.shape

        start_time = time.time()

        print("Apprentissage avec", N_SAMPLES, "Questions/réponses de taille maximale de ", N_WORDS, "mots pour :",

              self.NUM_EPOCHS,"itération")

        Step = np.around((N_SAMPLES) / self.NUM_SUBSETS)

        SAMPLE_ROUNDS = Step * self.NUM_SUBSETS



        for n_epoch in range(self.NUM_EPOCHS):

            # Boucle sur les batches en raison de contraintes de mémoire

            for n_batch in range(0, int(SAMPLE_ROUNDS), int(Step)):

                print(gc.collect(), "objet libérer")

                Q2 = self.Q[n_batch:n_batch + int(Step)]



                counter = 0

                for id, sentence in enumerate(self.A[n_batch:n_batch + int(Step) + 1]):

                    l = np.where(sentence == self.EOS)

                    limit = l[0][0]

                    counter += limit + 1



                question = np.zeros((counter, self.MAX_INPUT_LENGTH))

                answer = np.zeros((counter, self.MAX_OUTPUT_LENGTH))

                target = np.zeros((counter, self.DICTIONARY_SIZE))



                # Boucle sur les exemples de formation

                counter = 0

                for i, sentence in enumerate(self.A[n_batch:n_batch + int(Step)]):

                    ans_partial = np.zeros((1, self.MAX_OUTPUT_LENGTH))



                    # Boucle sur les positions de la sortie cible actuelle (la séquence de sortie actuelle)

                    l = np.where(sentence == self.EOS)  #

                    limit = l[0][0]



                    for k in range(1, limit + 1):

                        # Mapping the target output (the next output word) for one-hot codding:

                        target_ = np.zeros((1, self.DICTIONARY_SIZE))

                        target_[0, sentence[k]] = 1



                        # préparer la réponse partielle à la saisie

                        ans_partial[0, -k:] = sentence[0:k]



                        question[counter, :] = Q2[i:i + 1]

                        answer[counter, :] = ans_partial

                        target[counter, :] = target_

                        counter += 1



                # entrainer le modèle pour une itération

                print("Epoche", n_epoch)

                model.fit([question, answer], target, batch_size=self.BATCH_SIZE, epochs=1, verbose=1)

                del target

                del question

                del answer

        elapsed_time = time.time() - start_time

        print("Apprentissage términer dans", elapsed_time / 60, "minutes")

        if save:

            modele_name = self.output_modele_file + str(self.NUM_EPOCHS) + "EP_" + str(N_SAMPLES) + "_Sam" + str(

                int(elapsed_time / 60)) + "min.h5"

            model.save(modele_name)

            print("modele enregitrer à :",modele_name)

        return model



    def vector_to_phrase(self, vector):

        X = np.asarray([Mychatbot.index_to_word[int(i)] for i in vector[0] if i != self.EOS and i != self.BOS])

        return " ".join(X)



    def phrase_to_vector(self, sent="bonjour"):

        sent = [p.lower() for p in sent.split()]

        x = [self.word_to_index[w] for w in sent if w in self.word_to_index.keys()]

        x = np.asarray(x)

        Q = sequence.pad_sequences([x], maxlen=self.MAX_INPUT_LENGTH)

        return Q



    def get_answer(self, inputstr):

        inputvec = self.phrase_to_vector(inputstr)

        ans_partial = np.zeros((1, self.MAX_OUTPUT_LENGTH))

        ans_partial[0, -1] = self.BOS

        for k in range(self.MAX_OUTPUT_LENGTH - 1):

            ye = self.modele.predict([inputvec, ans_partial])

            mp = np.argmax(ye)

            ans_partial[0, 0:-1] = ans_partial[0, 1:]

            ans_partial[0, -1] = mp



        return self.vector_to_phrase(ans_partial)
# main

Mychatbot = Chatbot("train", save = True, ep=300)

#Mychatbot = Chatbot("chat", save = True, in_modele = "modele2000EP_247_Sam88min.h5")



#print(Q[0:])

#print(A)
# tests

def suzanne_start():

    print("################# Suzanne ################")

    print("Suzanne : Bonjour ! Je peux vous aider ?") #

    c = input("Hamza : ")

    while c!="bye":

        print("Suzanne :", Mychatbot.get_answer(c))

        c = input("Hamza : ")

#suzanne_start()
Mychatbot.modele.history.losses