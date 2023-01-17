import numpy as np # linear algebra

import pandas as pd
import os

files = os.listdir("/kaggle/input/alif-ner/alif/drive/My Drive/NER/Alif")

txt_files = [I for I in files if I.endswith(".txt")]
def is_good_label(label):

    valued_labels = ["ART" , "Event" , "Language" , "Location" , "Miscellaneous" , "NHFC" , 

                  "NORP" , 'Organization' , "Person" , "Product" , "Time" ]

    for I in valued_labels:

        if I in label:

            return True

    return label == 'O'
X = []

y = []

error = 0



for file in txt_files:

    file = open("/kaggle/input/alif-ner/alif/drive/My Drive/NER/Alif/%s" % (file))

    rows = [I.strip() for I in file.readlines()]

    flag = True

    X.append([])

    y.append([])



    for row in rows:

        tokens = row.strip().split(" ")

        if len(row) == 0:

            X.append([])

            y.append([])

            flag = True

        else:

            if flag:

                if len(tokens) == 1 or not is_good_label(tokens[-1]):

                    flag = False

                    error += 1

                    X.pop(-1)

                    y.pop(-1)

                else:

                    for I in range(len(tokens) - 1):

                        X[-1].append(tokens[I])

                        y[-1].append(tokens[-1])



    assert len(X) == len(y)



    for M in range(len(X)):

        assert len(X[M]) == len(y[M])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def process(label):

    valued_labels = ["ART" , "Event" , "Language" , "Location" , "Miscellaneous" , "NHFC" , 

                  "NORP" , 'Organization' , "Person" , "Product" , "Time" , "O"]

    for I in valued_labels:

        if I in label:

            return I
file = open("train.txt" , "w", encoding = "utf-8")



for row in range(len(X_train)):

    file.write("text ner\n")

    if len(X_train[row]) > 0:

        for index in range(len(X_train[row])):

            file.write("%s %s\n" % (X_train[row][index] , 

                                    process(y_train[row][index])))

        file.write("\n")

file.close()
file = open("test.txt" , "w", encoding = "utf-8")



for row in range(len(X_test)):

    file.write("text ner\n")

    if len(X_test[row]) > 0:

        for index in range(len(X_test[row])):

            file.write("%s %s\n" % (X_test[row][index] , 

                                    process(y_test[row][index])))

        file.write("\n")

file.close()
!pip uninstall typing -y

!pip install git+https://github.com/flairNLP/flair.git
import flair
from flair.data import Corpus

from flair.datasets import ColumnCorpus



# define columns

columns = {0: 'text', 1: 'ner'}





# init a corpus using column format, data folder and the names of the train, dev and test files

corpus: Corpus = ColumnCorpus(".", columns,

                              train_file='train.txt',

                              test_file='test.txt')
tag_dictionary = corpus.make_tag_dictionary(tag_type = 'ner')

print(tag_dictionary)
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

from typing import List



stacked_embeddings = StackedEmbeddings([FlairEmbeddings('id-forward'),

                                        FlairEmbeddings('id-backward'),

                                       ])
from flair.models import SequenceTagger

from flair.trainers import ModelTrainer

from flair.training_utils import EvaluationMetric
tagger: SequenceTagger = SequenceTagger(hidden_size=128,

                                        embeddings=stacked_embeddings,

                                        tag_dictionary=tag_dictionary,

                                        tag_type='ner',

                                        use_crf=True)

    

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('resources/taggers/example-ner',

              learning_rate=0.1,

              mini_batch_size=16,

              embeddings_storage_mode='none',

              max_epochs=10,

              checkpoint=True)