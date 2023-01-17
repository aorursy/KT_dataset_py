# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Combined_News_DJIA.csv')

data.head()
cols = data.columns[2:27]

#print(cols)

data[cols].head()

headlines = data[cols]
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import tokenize
temp = aggregate_sentiment(headlines.iloc[0])

temp1 = [x for x in temp]

print(temp1[0])

sid.polarity_scores(tokenize.sent_tokenize(temp1[0]))

#takes in a row of headlines for a date

sid = SentimentIntensityAnalyzer()

def aggregate_sentiment(headlines):

    score = 0

    sents = [tokenize.sent_tokenize(x) for x in headlines]

    #for sent in sents:

     #   score = score + sid.polarity_scores(sent)[0]

    return sents
tokenized_lines = headlines.apply(lambda x: aggregate_sentiment(x))