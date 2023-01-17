import pickle
import glob
import re, os
import pandas as pd

from gensim import corpora
from gensim.models import TfidfModel
from gensim import similarities

import spacy
from spacy.lang.en import English

import nltk
from nltk.stem import PorterStemmer

from scipy.cluster import hierarchy

from tqdm.notebook import trange, tqdm
from tqdm import tqdm_gui
import time

import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
nltk.download('averaged_perceptron_tagger')
# defining folder where data is kept
# using glob to import the files from the defined folder

folder = "/kaggle/input/book-dataset/"
files = glob.glob(folder+ '*.txt')
files.sort()
# inspecting list of files, to ensure dataset was propertly loaded
files
# loading in book content and titles into seperate lists we can use later

txts = []
titles = []

for n in files:
    f = open(n, encoding='utf-8-sig')
    # remove all non alpha numeric characters
    text = re.sub('[\W_]+',' ',f.read())
    # load titles and text into two sepereate lists
    titles.append(os.path.basename(n).replace('.txt', ''))
    txts.append(text)
# taking a look at the first 200 characters of the first book title to ensure we're pulling the titles and text in correctly.
print(titles[0])
print(txts[0][1:400])
for i in range(len(titles)):
    if titles[i] == 'DescentofMan':
        dom = i
# Print the stored index
print(dom)
# using spacey's stop word set
stopwords = spacy.lang.en.stop_words.STOP_WORDS

# inspecting 10 in the set
list(stopwords)[:10]
txts_lower_split = [txt.lower().split() for txt in txts]
texts = [[word for word in txt if word not in stopwords] for txt in txts_lower_split]

print(texts[2][:100])
porter = PorterStemmer()
stem_texts = [[porter.stem(token) for token in text] for text in texts]
# dumping to pickle so we don't have to repeat the stemming step when session ends
with open('/kaggle/working/stem_texts.p', 'wb') as f:
    pickle.dump(stem_texts, f)
# open pickled stemmed tokens
with open('/kaggle/working/stem_texts.p', 'rb') as f:
    stem_texts = pickle.load(f)
# remove pickled file from working directory if needed

# os.remove("/kaggle/working/stem_texts.p")
# previewing first 20 stemmed tokens from Descent of Man using its index.
stem_texts[2][:20]
dictionary = corpora.Dictionary(texts_stem)
bows = [dictionary.doc2bow(i) for i in texts_stem]

# Print the first five elements of the Descent of Mans Bag of words model
print(bows[2][:5])
df_bow_dom = pd.DataFrame(bows[2])
df_bow_dom.columns = ['index', 'occurrences']
df_bow_dom['token'] = [dictionary[index] for index in df_bow_dom["index"]]

# sort the created dataframe by occurences
df_bow_dom_sorted = df_bow_dom.sort_values(by='occurrences', ascending=False)
display(df_bow_dom_sorted)
# defining part of speech names from NLTK docs that we can use to isolate tokens
adjectives = ['JJ', 'JJR', 'JJS']
nouns = ['NN', 'NNS', 'NNP', 'NNPS']
verbs = ['VB','VBD','VBG','VBN','VBP','VBZ']
# adding part of speech column for each token
df_bow_dom_sorted['pos'] = [i[1] for i in list(nltk.pos_tag(df_bow_dom_sorted['token']))]
df_bow_dom_sorted.head()
# using pandas query function to create a new dataframe of just nouns and adjectives
# how cool is df.query!?

df_dom_nouns = df_bow_dom_sorted.query(f'pos in {nouns}')
df_dom_adj = df_bow_dom_sorted.query(f'pos == {adjectives}')
custom = go.layout.Template()

custom.layout = go.Layout(
    margin=dict(t=120, r=50, b=90, l=100),
    yaxis = dict( title_standoff = 10, gridcolor="#3B5CAB"),
    xaxis = dict( title_standoff = 20, gridcolor="#213A78"),
    plot_bgcolor="#213A78",
    paper_bgcolor="#213A78",
    font=dict(
        family='Montserrat, proportional',
        color='white',
        size=13
    ),
    title_font=dict(
    size=22
    ),
    autosize=True
)

custom.data.scatter = [
    go.Scatter(
        marker=dict(
            symbol="circle",
            size=8,
            color="#3EFFE8",
        ),
        line=dict(color='#3EFFE8'),
    )
]

pio.templates['custom'] = custom
fig = go.Figure(data=go.Bar(y=df_dom_nouns['occurrences'][:20], x = df_dom_nouns['token'][:20]))
  
fig.update_layout(title="Frequency Of Top 20 Nouns In Text",
                  yaxis = dict( title_text = "Frequency"),
                  xaxis = dict( title_text = "Top 20 nouns"),
                  template='plotly_white+custom')

fig.show()
fig = go.Figure(data=go.Bar(y=df_dom_adj['occurrences'][:20], x = df_dom_adj['token'][:20]))
  
fig.update_layout(title="Frequency Of Top 20 Adjectives In Text",
                  yaxis = dict( title_text = "Frequency"),
                  xaxis = dict( title_text = "Top 20 adjectives"),
                  template='plotly_white+custom')

fig.show()
