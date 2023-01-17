import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from __future__ import unicode_literals

import plac

import numpy

import spacy

from spacy.language import Language
vectors_loc="/kaggle/input/fasttext-hindi-300-vec/cc.hi.300.vec/cc.hi.300.vec"

lang = "hi"
"""Load vectors for a language trained using fastText

https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

Compatible with: spaCy v2.0.0+"""



nlp = spacy.blank(lang)    

with open(vectors_loc, "rb") as file_:        

    header = file_.readline()        

    nr_row, nr_dim = header.split()        

    nlp.vocab.reset_vectors(width=int(nr_dim))        

    for line in file_:            

        line = line.rstrip().decode("utf8")            

        pieces = line.rsplit(" ", int(nr_dim))            

        word = pieces[0]            

        vector = numpy.asarray([float(v) for v in pieces[1:]], dtype="f")            

        nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab    



# test the vectors and similarity    

text = "भारी बारिश के कारण आज कार्यालय बंद रहेगा"    

doc = nlp(text)    

print(text, doc[0].similarity(doc[1]))
print("Saved model to", 'hi_model')

nlp.to_disk('hi_model')
import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))