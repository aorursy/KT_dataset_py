# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install deepai-nlp
from deepai_nlp.wikicrawler.wiki_bs4 import WikiTextCrawler

from deepai_nlp.tokenization.crf_tokenizer import CrfTokenizer

from deepai_nlp.tokenization.utils import preprocess_text

from deepai_nlp.word_embedding import word2vec_gensim
import pandas as pd

import numpy as np



df = pd.read_csv('../input/kpdl-data/train_remove_noise.csv')

data = df['Content'].values
%%time

# Preprocess and tokenize

tokenizer = CrfTokenizer()

documents = preprocess_text(data, tokenizer=tokenizer) # Tách từ và clean
%%time

model = word2vec_gensim.Word2Vec (documents, size=400, window=5, min_count=1, workers=4, sg=1)

model.train(documents,total_examples=len(documents),epochs=30)
wv_path = "word2vec.model"

model.wv.save(wv_path)