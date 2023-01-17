"""  Towards finding Vector representation for words from Bollywood songs
     with a final objective of learning an AI model to generate good lyrics
     in the style of specific Lyricist(s) or combinations of some 
     e.g. Gulzar, S.D. Burman, R.D. Burman, MohRafi, 
     Javed Akhtar, Anand Bakshi and many more to pen down here.
"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Prepare a dataset of bollywood songs lyrics and upload the data set
# Data Augmentation: e.g. oo<->u, hh<->h, ei<-> e, aa<->a and many such rules
# Use Gensim to run word2vec on the documents
# Step 3: Analyze interesting cases/prepare some reports
# Step 4: Feed these results into an RNN/LSTM/GAN architecture to generate lyrics
# Step 5: 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import gensim
# Download Data using Python/Octoparse
"""
http://lyricsming.com about 20600 songs
Year
Singers
Lyricist
Music Director

It is important to have this skillset of downloading huge datasets in a structured manner from the internet
Python, OctoParse etc tools
"""

# Data Augmentation Rules
"""
oo<->u
hh<->h
ei<-> e
aa<->a
th<->t e.g. in tu, to etc
ee<->i
ii<->i

or simply remove duplication of characters...not correct
Remove integers from text
"""



asd