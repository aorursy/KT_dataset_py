import pandas as pd
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
doc=nlp("Narendra Modi Meets Trump tommorrow at 10am")
for token in doc:
    print(token.text,"--->",token.pos_)
for token in doc:
    print(token.text,"--->",token.dep_)
for ent in doc.ents:
    print(ent.text, ent.label_)