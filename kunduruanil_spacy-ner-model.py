# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import spacy

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
BASE_PATH = '../input/tweet-sentiment-extraction/'

os.listdir(BASE_PATH)
df = pd.read_csv(BASE_PATH + 'train.csv')

print(df.shape)

df.head()
print(df.shape)

df=df[df["sentiment"]!="neutral"]

df.shape
df['text']=df['text'].apply(str)

df['selected_text']=df["selected_text"].apply(str)
def create_data(df):

    train=[]

    for i,row in df.iterrows():

        text=row['text']

        st=row["selected_text"]

        start=text.find(st)

        end=start+len(st)

        train.append((text,{"entities":[(start,end,row['sentiment'])]}))

    return train

train=create_data(df)
nlp = spacy.blank("en")

ner = nlp.create_pipe("ner")

nlp.add_pipe(ner, last=True)

from spacy.util import minibatch, compounding

import random

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*other_pipes):

    for _, annotations in train:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])

    losses = {}

    nlp.begin_training()

    for i in range(100):

      random.shuffle(train)

      batches = minibatch(train, size=compounding(4.0, 32.0, 1.001))

      for batch in batches:

          texts, annotations = zip(*batch)

          nlp.update(

              texts,  # batch of texts

              annotations,  # batch of annotations

              drop=0.5,  # dropout - make it harder to memorise data

              losses=losses,

          )

      print("Losses", losses)
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
def predict(x):

    doc=nlp(x)

    out=[ent.text for ent in doc.ents]

    if out:

        return out[0]

    else:

        return x
def predict(x):

    doc=nlp(x)

    out=[ent.text for ent in doc.ents]

    if out:

        return out[0]

    else:

        return x

def final_text(data):

  pred=[]

  for i in range(len(data)):

    if data.loc[i,"sentiment"]=="neutral":

      pred.append(data.loc[i,"text"])

    else:

      pred.append(predict(data.loc[i,"text"]))

  return pred
test=pd.read_csv(BASE_PATH+"test.csv")

sub=pd.read_csv(BASE_PATH+"sample_submission.csv")
sub['selected_text']=final_text(test)

sub.to_csv("submission.csv", index=False)