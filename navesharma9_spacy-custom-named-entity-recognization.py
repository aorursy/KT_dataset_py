url = 'https://meenavyas.files.wordpress.com/2018/06/namedentityextraction.png'
response = requests.get(url)
Image.open(BytesIO(response.content))
import spacy
import pandas as pd
from spacy import displacy
from spacy.matcher import Matcher
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import requests
from io import BytesIO
model = spacy.load('en_core_web_sm') #Pre-trained model
doc = model('Hi I am John and I was born on 18th July 1987. \
           I work at Pramati from Hyderabad. I just bought a cricket bat \
           cost $100 from amazon and I will get is knock here for $5. I love Java')
displacy.render(doc,style='ent')
url = 'https://user-images.githubusercontent.com/13643239/55229632-dbff9480-521d-11e9-8499-efb2a9c948db.png'
response = requests.get(url)
Image.open(BytesIO(response.content))
model.pipeline
doc = model('My name is Jhon and I was born on 23rd June 1987')
doc.ents
model.remove_pipe('ner')
doc = model('My name is Jhon and I was born on 23rd June 1987')
doc.ents
TRAIN_DATA = [
   ("Python is cool", {"entities": [(0, 6, "PROGLANG")]}),
   ("Me like golang", {"entities": [(8, 14, "PROGLANG")]}),
   (("Yu like Java", {"entities": [(8, 14, "PROGLANG")]})),
   ('How to set up unit testing for Visual Studio C++',{'entities': [(45, 48, 'PROGLANG')]}),
   ('How do you pack a visual studio c++ project for release?',{'entities': [(32, 35, 'PROGLANG')]}),
   ('How do you get leading wildcard full-text searches to work in SQL Server?',{'entities': [(62, 65, 'PROGLANG')]}) 
]
TRAIN_DATA
def create_blank_nlp(train_data):
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    return nlp  
import random 
import datetime as dt

nlp = create_blank_nlp(TRAIN_DATA)
optimizer = nlp.begin_training()  
for i in range(10):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], sgd=optimizer, losses=losses)
    print(f"Losses at iteration {i} - {dt.datetime.now()}", losses)
doc = nlp("i write code in datascience") #ignore the sentense if it doen't make sense :)
displacy.render(doc, style="ent")
doc = nlp("i write code in javascript")
displacy.render(doc, style="ent")
doc = nlp("Python will be most use language")
displacy.render(doc, style="ent")
