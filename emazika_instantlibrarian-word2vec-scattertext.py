

import os



import numpy as np

import pandas as pd

from os import walk

for (dirpath, dirnames, filenames) in walk("../input"):

    print("Directory path: ", dirpath)

    print("Folder name: ", dirnames)

    print("File name: ", filenames)
data = pd.read_csv("../input/df-cov/df_COV.csv")

data.head()
%matplotlib inline

import scattertext as st #the main library used for corpous exploration

from pprint import pprint

import pandas as pd

import numpy as np

from scipy.stats import rankdata, hmean, norm

import spacy

import os, pkgutil, json, urllib

from urllib.request import urlopen

from IPython.display import IFrame

from IPython.core.display import display, HTML

from scattertext import CorpusFromPandas, produce_scattertext_explorer

from collections import OrderedDict

from spacy import displacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from spacy.matcher import Matcher

from gensim.models import word2vec

import re, io, itertools

import os, pkgutil, json, urllib

from urllib.request import urlopen

from IPython.display import IFrame

from IPython.core.display import display, HTML



nlp = spacy.load('en_core_web_sm')

display(HTML("<style>.container { width:98% !important; }</style>"))
data = data.dropna(subset=['abstract_x'])
data = data.loc[data['year'] >= 2019]
data['parsed_text'] = data.text.apply(nlp)
data.head()
corpus = (st.CorpusFromParsedDocuments(data, category_col='Category', parsed_col='parsed_text')

          .build()

          .get_unigram_corpus())
model = word2vec.Word2Vec(size=100, window=5, min_count=10, workers=4)

model = st.Word2VecFromParsedCorpus(corpus, model).train(epochs=10000)
model.wv.most_similar('hypercoagulable'),model.wv.most_similar('hypercoagulability'), model.wv.most_similar('clots')
hypercoagulable = ['hypercoagulable','hyperinflammatory', 'hypercoagulability', 'clots','coagulopathy', 'microcirculation','hypertrophy', 'vasoconstriction','stasis', 'vessel',  'transfusions']

model.wv.most_similar('efficacy'), model.wv.most_similar('therapeutics'), model.wv.most_similar('treatment'), model.wv.most_similar('inhibitor')
therapeutic = ['immunogenicity','potency', 'potent','pharmacokinetics', 'antivirals', 'therapies','vaccines', 'drugs','therapeutic','repurposing','treatments','immunotherapy','adjuvants','countermeasures','prophylaxis','cure','oseltamivir','inhibitors','analog', 'protease','camostat','pikfyve', 'mesylate', 'rapamycin', 'adenosine' ]

temp=data.text.fillna("0")



data['Category'] = pd.np.where(temp.str.contains('|'.join(therapeutic)), "therapeutic",

                       pd.np.where(temp.str.contains('|'.join(hypercoagulable)), "hypercoagulable","other"))



data['Category'].value_counts()
corpus = st.CorpusFromParsedDocuments(data, category_col='Category', parsed_col='parsed_text').build()
target_term = 'coagulation'



html = st.word_similarity_explorer_gensim(corpus,

                                          category='therapeutic',

                                          category_name='therapeutic',

                                          not_category_name='hypercoagulable',

                                          target_term=target_term,

                                          minimum_term_frequency=200,

                                          width_in_pixels=1000,

                                          word2vec=model,

                                          metadata=data['title_x'])

file_name = 'COVID19_DEMO_similarity_gensim.html'

#open(file_name, 'wb').write(html.encode('utf-8'))

#IFrame(src=file_name, width = 1200, height=700)
Final_submission = pd.read_csv("../input/final-submission/final_doc.csv")

Final_submission