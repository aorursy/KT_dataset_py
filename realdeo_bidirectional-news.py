import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ast import literal_eval

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/toxic-span-detection/tsd_train.csv")

train["spans"] = train.spans.apply(literal_eval)

train.head(5)



trial = pd.read_csv("/kaggle/input/toxic-span-detection/tsd_trial.csv")

trial["spans"] = trial.spans.apply(literal_eval)

trial.head(5)
from nltk import word_tokenize

from tqdm import tqdm_notebook as tqdm

import numpy as np

from keras.preprocessing import sequence
def prepare(text , span):

    text = text.replace('"' , "'")

    string = list(text)



    tokens_marked = word_tokenize("".join(string))

    tokens = word_tokenize(text)



    pointer = 0

    itemSpan = []

    marking = [None for I in range(len(text))]



    count = 0

    try:

        for token in tokens:

            start = text[pointer:].index(token) + pointer

            end = start + len(token) - 1

            pointer = end

            itemSpan.append((start , end))



            for L in range(start , end + 1):

                marking[L] = count

            count += 1

    except:

        print(tokens)

        print(token)

        print(text)

        raise



    try:

        labels = ["False" for token in tokens]



        for M in span:

            if marking[M] is not None:

                labels[marking[M]] = "True"

    except:

        print(tokens)

        print(token)

        print(text)

        print(M)

        print(marking)

        print(labels)

        raise



    assert len(tokens) == len(labels)

    return {'tokens' : tokens , 'labels' : labels , 

          'spans' : itemSpan , 'text' : text}
trial_processed = []



trial_spans = trial['spans'].values

trial_texts = trial['text'].values

for M in range(len(trial)):

    trial_processed.append(prepare(trial_texts[M] , trial_spans[M]))
train_processed = []



train_spans = train['spans'].values

train_texts = train['text'].values

for M in range(len(train)):

    train_processed.append(prepare(train_texts[M] , train_spans[M]))
!pip uninstall typing -y

!pip install git+https://github.com/flairNLP/flair.git
train_processed[0]
file = open("train.txt" , "w", encoding = "utf-8")





for row in range(len(train_processed)):

    if len(train_processed[row]['tokens']) > 0:

        for index in range(len(train_processed[row]['tokens'])):

            file.write("%s %s\n" % (train_processed[row]['tokens'][index] , 

                                    train_processed[row]['labels'][index]))

        file.write("\n")

file.close()
file = open("dev.txt" , "w", encoding = "utf-8")





for row in range(len(trial_processed)):

    if len(trial_processed[row]['tokens']) > 0:

        for index in range(len(trial_processed[row]['tokens'])):

            file.write("%s %s\n" % (trial_processed[row]['tokens'][index] , 

                                    trial_processed[row]['labels'][index]))

        file.write("\n")

file.close()
import flair
from flair.data import Corpus

from flair.datasets import ColumnCorpus



# define columns

columns = {0: 'text', 1: 'ner'}





# init a corpus using column format, data folder and the names of the train, dev and test files

corpus: Corpus = ColumnCorpus(".", columns,

                              train_file='train.txt',

                              test_file='dev.txt')
tag_dictionary = corpus.make_tag_dictionary(tag_type = 'ner')

print(tag_dictionary)
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

from typing import List



stacked_embeddings = StackedEmbeddings([FlairEmbeddings('news-forward'),

                                        FlairEmbeddings('news-backward'),

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

              max_epochs=5,

              checkpoint=True)
def extract_spans(text , tokens):

    text = text.replace('"' , "'")

    string = list(text)



    pointer = 0

    itemSpan = []

    marking = [None for I in range(len(text))]



    count = 0

    

    for token in tokens:

        start = text[pointer:].index(token.text) + pointer

        end = start + len(token.text) - 1

        pointer = end

        itemSpan.append((start , end))



    return itemSpan
def accuracy(ground, prediction):

    if len(ground) == 0:

        if len(prediction) == 0:

            return 1

        else:

            return 0

        

    if len(prediction) == 0:

        return 0

    

    precision = len(set(ground) & set(prediction)) / len(set(prediction))

    recall = len(set(ground) & set(prediction)) / len(set(ground))



    return (2 * precision * recall) / (precision + recall)
from flair.data import Sentence 
kalimat = list(sentence)
sentence = Sentence("You are an idiot")

tagger.predict(sentence)

print(sentence.to_tagged_string())
entity.text , entity.tag
for entity in sentence.get_spans('ner'):

    print(entity)
total = 0

processed = 0

for K in tqdm(range(len(trial_spans))):

    try:

        sentence = Sentence(trial_processed[K]['text'])

        tagger.predict(sentence)

        result = [I for I in sentence.get_spans()]

        spans = extract_spans(trial_processed[K]['text'] , result)

        prediction = []



        for M in range(len(result)):

            if result[M].tag == 'True':

                for Z in range(spans[M][0] , spans[M][1] + 1):

                    prediction.append(Z)



        total += accuracy(trial_spans[K] , prediction)

        processed += 1

    except:

        print(K)
total / processed