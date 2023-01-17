from IPython.display import YouTubeVideo      
YouTubeVideo('fYEf8kjPuV8')
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
import random    
import datetime as dt
doc1 = nlp('John lives in New York, where is Mr.Abhishek its already 11PM, he likes burger so he went to KFC')
for i in doc1.ents:
  print(i.text,'-',i.label_)
spacy.explain('GPE')
#Visualize
displacy.render(doc1,style='ent',jupyter=True)
doc = nlp('Hi i am Jhon')
doc
nlp.pipeline
doc.ents
nlp.remove_pipe('ner')
doc = nlp('Hi i am Jhon')
doc.ents
# Training  data
train = [("I love burger", {"entities" : [(7,13 , "FOOD")]}),
         ("pizza with more cheese",  {"entities" : [(0,5, "FOOD")]}),
         ("chips is soo crispy", {"entities" : [(0,5 , "FOOD")]}),                          
        ]
train
def create_blank_nlp(train_data):
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    return nlp 
nlp = create_blank_nlp(train)
optimizer = nlp.begin_training()  
for i in range(7):
    random.shuffle(train)
    losses = {}
    for text, annotations in train:
        nlp.update([text], [annotations], sgd=optimizer, losses=losses)
    print(f"Losses at iteration {i} - {dt.datetime.now()}", losses)
doc = nlp("yummy burger")
displacy.render(doc, style="ent")
doc = nlp("I ordered pizza")
displacy.render(doc, style="ent")