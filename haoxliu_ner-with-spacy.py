# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from __future__ import unicode_literals, print_function

import plac

import random

from pathlib import Path

import spacy

from spacy.util import minibatch, compounding
import os
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print(train_data.shape)

print(test_data.shape)
train_data.describe()
train_data.text.dtype
train_data['n_text_words'] = train_data['text'].apply(lambda text: len(str(text).split()))
test_data['n_text_words'] = test_data['text'].apply(lambda text: len(str(text).split()))
test_data.head()
train_data_positive = train_data[(train_data.sentiment == 'positive') & (train_data.n_text_words > 3)]
train_data_positive.shape
train_data_negative = train_data[(train_data.sentiment == 'negative') & (train_data.n_text_words > 3)]

train_data_negative.shape
def load_model(pre_model = None, label = None):

    if pre_model is not None:

        nlp = spacy.load(pre_model)

        print("Loaded model '%s'" % pre_model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")



        if "ner" not in nlp.pipe_names:

            ner = nlp.create_pipe("ner")

            nlp.add_pipe(ner)

        else:

            ner = nlp.get_pipe("ner")



        if label is not None:

            ner.add_label(label)



    return nlp



def train_model(model, nlp, train_datas, n_iter = 30):

    if model is None:

        optimizer = nlp.begin_training()

    else:

        optimizer = nlp.resume_training()

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]



    with nlp.disable_pipes(*other_pipes):

        sizes = compounding(1.0, 64.0, 1.001)



        for itn in range(n_iter):

            random.shuffle(train_datas)

            batches = minibatch(train_datas, size=sizes)

            losses = {}

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)

            print(itn, "Losses", losses)



    return nlp



def test_model(ner_model, text):

    doc = ner_model(text)

    ent_array = []

    for ent in doc.ents:

        start = text.find(ent.text)

        end = start + len(ent.text)

        new_int = [start, end, ent.label_]

        

        if new_int not in ent_array:

            ent_array.append([start, end, ent.label_])

        

    return text[ent_array[0][0]:ent_array[0][1]] if len(ent_array) > 0 else text





def save_model(ner_model, output_dir = None, new_model_name = None):

    if output_dir is not None:

        output_dir = Path(output_dir)

        if not output_dir.exists():

            output_dir.mkdir()

        ner_model.meta["name"] = new_model_name

        ner_model.to_disk(output_dir)

        print("Saved model to", output_dir)



def get_train_datas(data):

    train_datas = []

    texts = data.text

    selected_texts = data.selected_text

    for selected_text, text in zip(selected_texts, texts):

        start = text.find(selected_text)

        end = start + len(selected_text)

        train_datas.append((text, {"entities":[(start, end, "selected_text")]}))

    return train_datas



def get_model(sentiment, train_datas, more_iters = 30):

    if sentiment == 'positive':

        positive_model_path = "/kaggle/working/models"

        positive_datas = train_datas

        if not os.path.exists(positive_model_path):

            nlp = load_model(label = 'selected_text')

            ner_model_positive = train_model(None, nlp, positive_datas, n_iter=50)

            save_model(ner_model_positive, output_dir = "/kaggle/working/models", new_model_name = "posi_model")

        else:

            ner_model_positive = load_model(positive_model_path)

#             ner_model_positive = spacy.load("/kaggle/working/models")

            if more_iters > 0:

                ner_model_positive = train_model(positive_model_path, ner_model_positive, positive_datas, more_iters)

                save_model(ner_model_positive, output_dir = "/kaggle/working/models", new_model_name = "posi_model")

        return ner_model_positive

    else:

        negative_model_path = "/kaggle/working/models_nega"

        negative_datas = train_datas

        if not os.path.exists(negative_model_path):

            nlp = load_model(label = 'selected_text')

            ner_model_negative = train_model(None, nlp, negative_datas, n_iter=50)

            save_model(ner_model_negative, output_dir = "/kaggle/working/models_nega", new_model_name = "nega_model")

        else:

            ner_model_negative = load_model(negative_model_path)

#             ner_model_negative = spacy.load("/kaggle/working/models_nega")

            if more_iters > 0:

                ner_model_negative = train_model(negative_model_path, ner_model_negative, negative_datas, more_iters)

                save_model(ner_model_negative, output_dir = "/kaggle/working/models_nega", new_model_name = "nega_model")

        return ner_model_negative
print(os.path.exists("/kaggle/working/models_nega"))

print(os.path.exists("/kaggle/working/models"))
positive_datas = get_train_datas(train_data_positive)

negative_datas = get_train_datas(train_data_negative)
print(len(positive_datas))

print(len(negative_datas))
ner_model_positive = get_model('positive', positive_datas, more_iters=50)

ner_model_negative = get_model('negative', negative_datas, more_iters=50)
pre_list = []

for i in range(test_data.shape[0]):

    t_data = test_data.iloc[i]

    if t_data.sentiment == 'neutral' or t_data.n_text_words <= 3:

        pre_list.append(t_data.text)

    elif t_data.sentiment == 'positive':

        pre_list.append(test_model(ner_model_positive, t_data.text))

    else:

        pre_list.append(test_model(ner_model_negative, t_data.text))
submission['selected_text'] = pre_list

display(submission.head(10))
submission.to_csv("submission.csv", index = False)