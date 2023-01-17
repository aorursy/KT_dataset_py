# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd

import nltk


from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

emails = pd.read_csv("../input/Emails.csv")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
sentences= []

for em in emails["ExtractedBodyText"]:
    if type(em) is str:
        emailtoken = tokenize.sent_tokenize(em)
        sentences.extend(emailtoken)
sid = SentimentIntensityAnalyzer()
dataneg=[]
datapos=[]
dataneutr=[]
datacompound=[]
for sentence in sentences:
    
    ss = sid.polarity_scores(sentence)
  
    dataneg.append(ss['neg'])
    datapos.append(ss['pos'])
    dataneutr.append(ss['neu'])
    datacompound.append(ss['compound'])

        
        
       

import matplotlib.pyplot as plt
#print(dataneutr)
data=[sum(np.array(dataneg)), sum(np.array(datapos)), sum(np.array(dataneutr)), sum(np.array(datacompound))]

y=len(data)
x = np.arange(y)
width = 0.35
fig = plt.figure()
ax = fig.add_subplot(111)


xTickMarks=('neg', 'pos', 'neu', 'comp')
xtickNames = ax.set_xticklabels(xTickMarks)
ax.set_xticks(x+(width/2))

rects1 = ax.bar(x,data,width)
plt.show()