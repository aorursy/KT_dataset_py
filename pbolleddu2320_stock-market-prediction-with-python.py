# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.sentiment.util import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#print(check_output(["pwd"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Combined_News_DJIA.csv')
df
from nltk import tokenize

tricky_sentences = df['Top1']
sid = SentimentIntensityAnalyzer()
for sentence in tricky_sentences:
    #print(sentence)
    ss = sid.polarity_scores(sentence)
    print( ss['compound'] )
    
