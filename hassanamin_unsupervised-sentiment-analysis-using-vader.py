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
# install vader if not already available

!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):

    score = analyser.polarity_scores(sentence)

    print("{:-<40} {}".format(sentence, str(score)))
sentiment_analyzer_scores("The phone is super cool.")
sentiment_analyzer_scores("The phone is super cool!")

sentiment_analyzer_scores("The phone is super cool!!")

sentiment_analyzer_scores("The phone is super cool!!!")
sentiment_analyzer_scores("The phone is super COOL!")
sentiment_analyzer_scores("Food here is good.")

sentiment_analyzer_scores("Food here is moderately good.")

sentiment_analyzer_scores("Food here is extremely good.")
sentiment_analyzer_scores("Food here is extremely good but service is horrible.")
sentiment_analyzer_scores("The food here isn’t really all that great")

sentiment_analyzer_scores("The food here isn’t that great")



sentiment_analyzer_scores("The food here is not really all that great")

sentiment_analyzer_scores("The food here is not that great")

print(sentiment_analyzer_scores('I am 😄 today'))

print(sentiment_analyzer_scores('😊'))

print(sentiment_analyzer_scores('😥'))

print(sentiment_analyzer_scores('☹️'))
print(sentiment_analyzer_scores("Today SUX!"))

print(sentiment_analyzer_scores("Today only kinda sux! But I'll get by, lol"))
print(sentiment_analyzer_scores("Make sure you :) or :D today!"))
import pandas as pd



scores =[]

sentences = ["A really bad, horrible book.","A good, awesome, wonderful, cool book !!!  :)"]



for sentence in sentences:

    score = analyser.polarity_scores(sentence)

    scores.append(score)

    

#Converting List of Dictionaries into Dataframe

dataFrame= pd.DataFrame(scores)



print(dataFrame)



print("Overall Sentiment Score for the multiple sentences :- ",dataFrame.mean())