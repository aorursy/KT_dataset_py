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
print("Durga")
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
!pip install spacy
import spacy

from spacy import displacy
nlp = spacy.load('en_core_web_sm')

Doc = "Hey! I am Durga. Today is Sunday and we are learning NLP with Usha"

doc1 = nlp(Doc)

for token in doc1:

    print(token)
for lem in doc1:

    print(lem.text,lem.lemma_)

    

for token in doc1:

    print(token.text,token.pos_)    
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
for token in doc1:

    if token.is_stop==False:

        print(token)
displacy.render(doc1,style = 'dep') #check in jupiter if the diagram appears
displacy.render(doc1,style = 'ent')#tagging entities
spacy.explain("DATE")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt