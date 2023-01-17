import numpy as np
import pandas as pd
import spacy

# Import the english language model
nlp = spacy.load('en')
df = pd.read_csv("../input/fake.csv")
df.shape
df.head()
df["title"].head()
txt = df["title"][1009]
txt
doc = nlp(txt)    
olist = []
for token in doc:
    l = [token.text,
        token.idx,
        token.lemma_,
        token.is_punct,
        token.is_space,
        token.shape_,
        token.pos_,
        token.tag_]
    olist.append(l)
    
odf = pd.DataFrame(olist)
odf.columns= ["Text", "StartIndex", "Lemma", "IsPunctuation", "IsSpace", "WordShape", "PartOfSpeech", "POSTag"]
odf
doc = nlp(txt)
olist = []
for ent in doc.ents:
    olist.append([ent.text, ent.label_])
    
odf = pd.DataFrame(olist)
odf.columns = ["Text", "EntityType"]
odf
from spacy import displacy
displacy.render(doc, style='ent', jupyter=True)
txt = df["title"][3003]
doc = nlp(txt)
colors = {'GPE': 'lightblue', 'NORP':'lightgreen'}
options = {'ents': ['GPE', 'NORP'], 'colors': colors}
displacy.render(doc, style='ent', jupyter=True, options=options)
txt = df["title"][2012]
print(txt)
doc = nlp(txt)
olist = []
for chunk in doc.noun_chunks:
    olist.append([chunk.text, chunk.label_, chunk.root.text])
odf = pd.DataFrame(olist)
odf.columns = ["NounPhrase", "Label", "RootWord"]
odf
doc = nlp(df["title"][1009])
olist = []
for token in doc:
    olist.append([token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children]])
odf = pd.DataFrame(olist)
odf.columns = ["Text", "Dep", "Head text", "Head POS", "Children"]
odf
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
doc = nlp(df["title"][3012])
displacy.render(doc, style='dep', jupyter=True, options={'distance': 60})
nlp = spacy.load('en_core_web_lg')
from scipy import spatial
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

queen = nlp.vocab['Queen'].vector
computed_similarities = []
for word in nlp.vocab:
    # Ignore words without vectors
    if not word.has_vector:
        continue
    similarity = cosine_similarity(queen, word.vector)
    computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
print([w[0].text for w in computed_similarities[:10]])
queen = nlp.vocab['Queen']
elizabeth = nlp.vocab['Elizabeth']
britain = nlp.vocab['Britain']
dolphin = nlp.vocab['Dolphin']
king = nlp.vocab['King']
 
print("Word similarity score between Queen and Elizabeth : ",queen.similarity(elizabeth))
print("Word similarity score between Queen and Britain : ",queen.similarity(britain))
print("Word similarity score between Queen and Dolphin : ",queen.similarity(dolphin))
print("Word similarity score between Queen and King : ",queen.similarity(king))