import spacy
from spacy import displacy
#spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm') 

import pandas as pd
from bs4 import BeautifulSoup as bs4
winners = pd.read_csv('../input/kaggle-blog-winners-posts/WinnersInterviewBlogPosts.csv')
doc = nlp(bs4(winners.content[172]).get_text())
doc_sents = doc.sents
from IPython.display import Image
Image(url='https://spacy.io/assets/img/architecture.svg')
displacy.render(list(doc_sents)[:1], style='dep', page=False, jupyter=True, options={'compact':True}) #test style='ent' with different options
tokens = pd.DataFrame(
    [[token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, [child for child in token.children]] for token in doc],
    columns=['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'child'])
tokens.head()
pd.DataFrame(
    [[chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text] for chunk in doc.noun_chunks],
    columns=['text', 'root.text', 'root.dep', 'root.head.text']).head()
ent = pd.DataFrame(
    [[ent.text, ent.start_char, ent.end_char, ent.label_] for ent in doc.ents],
    columns=['text', 'start_char', 'end_char', 'label'])
ent.head()
ent[ent['label'].isin(['ORG','NORP','PERSON'])]['text'].value_counts()[:20].reset_index().plot(kind='barh', x='index', y='text')
tokens[((tokens['is_stop']==False) & (tokens['is_alpha']==True))]['text'].str.lower().value_counts()[:20].reset_index().plot(kind='bar', x='index', y='text')
tokens[((tokens['is_stop']==True) & (tokens['is_alpha']==True))]['text'].str.lower().value_counts()[:20].reset_index().plot(kind='bar', x='index', y='text')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
np.random.seed(14)

p = '../input/street-view-getting-started-with-julia/'
st_chars = pd.read_csv(p + '/trainLabels.csv')
st_chars['path'] = st_chars['ID'].map(lambda x: p + 'train/train/' + str(x) + '.Bmp')

s = 'Happy   Kaggling'
fig=plt.figure(figsize=(6, 4))
for i in range(len(s)):
    fig.add_subplot(2, 8, i+1)
    if s[i]==' ':
        img = Image.new('RGB', (10,10), (255, 255, 255))
    else:
        p = np.random.choice(st_chars[st_chars['Class']==s[i]]['path'].values)
        img = Image.open(p)
    plt.imshow(img); plt.axis('off')
plt.show()