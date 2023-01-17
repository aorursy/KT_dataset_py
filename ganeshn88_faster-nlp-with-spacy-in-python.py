import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import random
train = pd.read_csv("../input/train.csv")
random.seed(22)
text = train.text[random.sample(range(1,50),5)]

train.head()
train.shape
import spacy
nlp = spacy.load('en_core_web_sm')
text1 = str(text)
doc = nlp(text1)
df = pd.DataFrame()

for i, token in enumerate(doc):
    df.loc[i, 'text'] = token.text
    df.loc[i, 'pos'] = token.pos_
    df.loc[i, 'dep'] = token.dep_

    
df.head(15)
df1 = pd.DataFrame()
for i,token in enumerate(doc):
    df1.loc[i,'text'] = token.text
print(df1.head(15))
df3 = pd.DataFrame()
for i,token in enumerate(doc):
    df3.loc[i,'text'] = token.text
    df3.loc[i,'lemma_'] = token.lemma_
    df3.loc[i,'pos_'] = token.pos_
    df3.loc[i,'tag_'] = token.tag_
    df3.loc[i,'dep_'] = token.dep_
    df3.loc[i,'shape_'] = token.shape_
    df3.loc[i,'is_alpha'] = token.is_alpha
    df3.loc[i,'is_stop'] = token.is_stop
df3.head(15)
spacy.displacy.render(doc, style='ent',jupyter=True)
spacy.displacy.render(doc, style='dep',jupyter=True,options = {'compact':60})
df2 = pd.DataFrame()

for i, ent in enumerate(doc):
    df2.loc[i, 'text'] = ent.text
    df2.loc[i, 'pos'] = ent.pos_
    df2.loc[i, 'dep'] = ent.dep_
df2.head(15)
df3 = pd.DataFrame()
for i,token in enumerate(doc):
    df3.loc[i,'text'] = token.text
    df3.loc[i,'has_vector'] = token.has_vector
    df3.loc[i,'vector_norm'] = token.vector_norm
    df3.loc[i,'is_oov'] = token.is_oov
df3.head(15)
for word in doc:
    lexeme = doc.vocab[word.text]
    print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix_, lexeme.suffix_,
          lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)