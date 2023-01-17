# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def load_data(csv_file, split=0.9): 
    data = pd.read_csv(csv_file)
    data = data.dropna()
    
    train_data = data.sample(frac=0.25, random_state=7)
    
    comments = train_data.comment.values 
    labels = [{'Sarcastic': bool(y), 'Not Sarcastic': not bool(y)} 
             for y in train_data.label.values]
    split = int(len(train_data) * split)
    
    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]
    
    return comments[:split], train_labels, comments[split:], val_labels
train_comments, train_labels, val_comments, val_labels = load_data('/kaggle/input/sarcasm/train-balanced-sarcasm.csv')
print("Comments from training data\n---------")
print(train_comments[:2])
print("Labels from training data\n---------")
print(train_labels[:2])

import spacy 

# creating a blank model
nlp = spacy.blank("en")

# creating a text categorizer
textcat = nlp.create_pipe("textcat", 
                         config = {
                             "exclusive_classes":True,
                             "architecture":"bow"})

nlp.add_pipe(textcat)

# adding the labels to our text categorizer
textcat.add_label("Sarcastic")
textcat.add_label("Not Sarcastic")
from spacy.util import minibatch 
import random

def train(model, train_data, optimizer, batch_size=8): 
    losses = {}
    random.shuffle(train_data)
    batches = minibatch(train_data, size=batch_size)
    for batch in batches: 
        comments, labels = zip(*batch)
        model.update(comments, labels, sgd=optimizer, losses=losses)
    return losses
spacy.util.fix_random_seed(1)
random.seed(1)

optimizer = nlp.begin_training()
train_data = list(zip(train_comments, train_labels))
losses = train(nlp, train_data, optimizer)

comment = 'You are doing great'
doc = nlp(comment)
print(doc.cats)
def predict(nlp, texts): 
    
    docs = [nlp.tokenizer(text) for text in texts]
    
    textcat = nlp.get_pipe('textcat')
    scores, _ = textcat.predict(docs)
    
    predicted_class = scores.argmax(axis=1)
    
    return predicted_class
comments = val_comments[0:15]
predictions = predict(nlp, comments)

for p, c in zip(predictions, comments): 
    print(f"{textcat.labels[p]}: {c} \n")
comments = ['Hello my name is Avery', 'Great caption', 'You are kidding me', 'Nice perfume, how long did you marinate it?']
predictions = predict(nlp, comments)

for p, c in zip(predictions, comments):
    print(f"'{c}': {textcat.labels[p]}  \n")
comments = []
comments.append(input())
predictions = predict(nlp, comments)

for p, c in zip(predictions, comments): 
    print(f"'{c}': {textcat.labels[p]} \n")
    

# example phrases to enter: 
# If stupidity were a profession then you'd be a billionare
# I am taking classes online 
# This tea is lovely
# If you find me offensive then I suggest you quit finding me 