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
import spacy

import re

from spacy.util import minibatch

import random

nlp=spacy.blank('en')

textcat=nlp.create_pipe("textcat", config={"exclusive_classes": True,"architecture": "bow"})

nlp.add_pipe(textcat)



#load the Sentences_AllAgree.txt data and build the  data

def load_data(filename):

    texts=[]

    labels=[]

    with open(filename,'rb') as file:

        data=file.readlines()

    for data_ in data:

        result=re.match(r'''^b['"](.*?)@(.*?)\\r\\n['"]''',str(data_))

        texts.append(result.group(1))

        labels.append(result.group(2))

    textcat.add_label("positive")

    textcat.add_label("negative")

    textcat.add_label("neutral")

    labels=[{"cats":{"positive":label=="positive","negative":label=="negative","neutral":label=="neutral"}}\

                  for label in labels]

    datas=list(zip(texts,labels))

    return datas

#get the train data and the evaluation data

def split_datas(datas,split=0.2):

    random.shuffle(datas)

    train_datas=datas[:round(len(datas)*split)]

    eval_datas=datas[round(len(datas)*split):]

    return train_datas,eval_datas



#train the model

def train_model(model,train_datas):

    optimizer=model.begin_training()

    losses={}

    #shuffle the train_datas

    for i in range(10):

        print(f'已运行{i/10:.2%}')

        random.shuffle(train_datas)

        for batch in minibatch(train_datas):

            texts,labels=zip(*batch)

            model.update(texts,labels,sgd=optimizer,losses=losses)

    return losses



#predict the model

def predict(model,texts):

    docs=[model.tokenizer(text) for text in texts]

    textcat=model.get_pipe('textcat')

    scores,_=textcat.predict(docs)

    indexs=scores.argmax(axis=1)

    sentiments=[textcat.labels[index] for index in indexs]

    return sentiments



#evaluate the model

def evaluate_model(model,eval_datas):

    true_labels=[]

    eval_texts,eval_labels=zip(*eval_datas)

    for eval_label in eval_labels:

        for key, val in eval_label["cats"].items():

            if val == True:

                true_labels.append(key)

    predict_labels=predict(model,eval_texts)

    predict_labels=np.array(predict_labels)

    correct_predictions=true_labels==predict_labels

    accuracy=correct_predictions.mean()

    return accuracy

filename='../input/sentiment-analysis-for-financial-news/FinancialPhraseBank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'

datas=load_data(filename)

train_datas,eval_datas=split_datas(datas)

train_model(nlp,train_datas)

accuracy=evaluate_model(nlp,eval_datas)

print(f"this model's accuracy is {accuracy:.2%}")








