import re



from nltk.corpus import (LazyCorpusLoader, NombankCorpusReader)

from nltk.corpus import treebank
import nltk

# Removing the original path

if '/usr/share/nltk_data' in nltk.data.path:

    nltk.data.path.remove('/usr/share/nltk_data')

nltk.data.path.append('../input/')

nltk.data.path
nombank = LazyCorpusLoader(

    'nombank.1.0', NombankCorpusReader,

    'nombank.1.0', 'frames/.*\.xml', 'nombank.1.0.words',

    lambda filename: re.sub(r'^wsj/\d\d/', '', filename),

    treebank, nltk_data_subdir='') # Must be defined *after* treebank corpus.
import os

os.listdir('../input/nombank.1.0/')