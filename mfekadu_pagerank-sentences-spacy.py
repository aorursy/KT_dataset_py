# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip uninstall -y cupy-cuda101

!pip uninstall -y spacy

!pip install cupy-cuda101==6.5.0

!pip install spacy[cuda101]==2.2.4

!pip list | grep cupy

!echo "....."

!pip list | grep spacy

!echo "....."

# !pip install spacy[cuda]

!echo "....."

!pip list | grep spacy

!echo "....."

!pip list | grep cuda
import cupy

import spacy

spacy.require_gpu()

# spacy.prefer_gpu()
!pip install text2text
!pip install textacy
df1 = pd.read_csv("/kaggle/input/make-final-plato-corpus/corpus_articles.csv")
df1.head()
nlp = spacy.load("en_core_web_lg")
nlp("\napple").similarity(nlp("banana"))
import networkx as nx
article = nlp(df1["article_text"][0])
article[:80]
import warnings
g = article.sents

a = next(g)

a
# # TODO: find a more efficient way.

# # because s1.sim(s2) === s2.sim(s1)

warnings.filterwarnings("ignore")

# memo = {}

# sim_mat = []

# for s1 in article.sents:

#     sims = []

#     for s2 in article.sents:

#         if s1 != s2:

#             sim = memo.get(

#                 s1,

#                 {}

#             ).get(

#                 s2,

#                 s1.similarity(s2)

#             )

#             memo[s1] = {

#                 s2: sim

#             }

#             memo[s2] = {

#                 s1: sim

#             }

#             sims.append(sim)

#         else:

#             sims.append(0.0)

#     sim_mat.append(sims)

sim_mat = np.array([[float(s1.similarity(s2)) for s2 in article.sents] for s1 in article.sents])

warnings.filterwarnings("default")
import networkx as nx



nx_graph = nx.from_numpy_array(sim_mat)

scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(article.sents)), reverse=True)
# Specify number of sentences to form the summary

sn = 100



# Generate summary

for i in range(sn):

    print(ranked_sentences[i][1])
import textacy

from textacy.ke import textrank
textrank(

    article, 

    window_size=2, 

    edge_weighting="binary",

    position_bias=True,

)
#!export CUDA_HOME=/usr/local/cuda-10.1

#!echo $CUDA_HOME

#!pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension
import text2text
from text2text.text_generator import TextGenerator

qg = TextGenerator(output_type="question")
qg.predict((str(x) for x in article.sents))