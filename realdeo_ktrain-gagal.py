!pip install ktrain
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ast import literal_eval



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
import ktrain

from ktrain import text as txt
train , test , preproc = txt.entities_from_array([I['tokens'] for I in train_processed] , 

                                                 [I['labels'] for I in train_processed] ,

                                                 [I['tokens'] for I in trial_processed] , 

                                                 [I['labels'] for I in trial_processed] ,)
sum([I['labels'].count("True") for I in train_processed]) / sum([len(I['labels']) for I in train_processed])
sum(['True' in I['labels'] for I in train_processed]) / len(train_processed)
model = txt.sequence_tagger('bilstm-bert', preproc,

                            bert_model='bert-base-cased')
learner = ktrain.get_learner(model, train_data=train, val_data=test, batch_size=128)
learner.fit(0.01, 1, cycle_len=4, checkpoint_folder='/tmp/saved_weights')
predictor = ktrain.get_predictor(model, preproc)
def extract_spans(text , tokens):

    text = text.replace('"' , "'")

    string = list(text)



    pointer = 0

    itemSpan = []

    marking = [None for I in range(len(text))]



    count = 0

    

    for token in tokens:

        start = text[pointer:].index(token) + pointer

        end = start + len(token) - 1

        pointer = end

        itemSpan.append((start , end))



    return itemSpan
trial_spans[0]
from tqdm import tqdm_notebook as tqdm
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
total = 0

processed = 0

for K in tqdm(range(len(trial_spans))):

    try:

        result = predictor.predict(trial_processed[K]['text'])

        spans = extract_spans(trial_processed[K]['text'] , [I[0] for I in result])

        prediction = []



        for M in range(len(result)):

            if result[M][1] == 'True':

                for Z in range(spans[M][0] , spans[M][1] + 1):

                    prediction.append(Z)



        total += accuracy(trial_spans[K] , prediction)

        processed += 1

    except:

        print(K)
total / processed