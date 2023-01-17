"""

# nltk

# textblob

# flair



ToDo



1. Train Custome sentiment Analysis



2. Deep Pavlov rus sentiment Transfer learning to English



3. diffrent types of clean text



4. add Stanford’s CoreNLP



6. split by sentiment





""";
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
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train.head()
import nltk

#nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
def nltkpolar(row,threshold=0.05):

    

    polarityDict = sid.polarity_scores(row)

    comp = polarityDict['compound']

    

    if comp >= threshold:

        topSentiment = 'positive'

    elif comp <= -threshold:

        topSentiment = 'negative'

    else:

        topSentiment = 'neutral'

        

    #topSentiment = max(polarityDict, key=polarityDict.get)

    

    if polarityDict[max(polarityDict, key=polarityDict.get)] == 0:

        topSentiment =  'Zeroes'

    

    return topSentiment
%%time



try :

    train['selected_text'] = train['selected_text'].astype(str)

    train['text'] = train['text'].astype(str)



except:

    print("Already string")





train['NLTK_text_SENT'] = train['text'].apply(nltkpolar,threshold=0.05)  

train['NLTK_selected_text_SENT'] = train['selected_text'].apply(nltkpolar,threshold=0.05)
train.head()
frame = {'Real sentiment': train.sentiment.value_counts(),

        'NLTK_selected_text_SENT': train.NLTK_selected_text_SENT.value_counts(), 

        'NLTK_Full text_SENT': train.NLTK_text_SENT.value_counts()

        } 

  

result = pd.DataFrame(frame).T

result.plot.bar(rot=45);
# selected text nltk sentiment match with competition sentiment

len(train[train['NLTK_selected_text_SENT'] ==train.sentiment])/len(train)
# full text nltk sentiment match with competition sentiment

len(train[train['NLTK_text_SENT'] ==train.sentiment])/len(train)
from textblob import TextBlob



def textblobpolar(row,threshold):

    # todo change treshhold

    

    polarity = TextBlob(row).sentiment.polarity

    

    if polarity >= threshold:

        topSentiment = 'positive'

    elif polarity <= -threshold:

        topSentiment = 'negative'

    else:

        topSentiment = 'neutral'

    

    return topSentiment
%%time



train['TB_text_SENT'] = train['text'].apply(textblobpolar,threshold=0.1)  

train['TB_selected_text_SENT'] = train['selected_text'].apply(textblobpolar,threshold=0.1)
frame = {'Real sentiment': train.sentiment.value_counts(),

         'NLTK_selected_text_SENT': train.NLTK_selected_text_SENT.value_counts(), 

         'NLTK_Full text_SENT': train.NLTK_text_SENT.value_counts(),

         

         'TB_selected_text_SENT': train.TB_selected_text_SENT.value_counts(),

         'TB_Full_text_SENT': train.TB_text_SENT.value_counts()

        } 

  

result = pd.DataFrame(frame).T

result.plot.bar(rot=45);
len(train[train['TB_selected_text_SENT'] ==train.sentiment])/len(train)
len(train[train['TB_text_SENT'] ==train.sentiment])/len(train)
!pip install flair;

import flair

flair_sentiment = flair.models.TextClassifier.load('en-sentiment');
def flairSent(row,threshold=0.80):

    

    s = flair.data.Sentence(row)

    flair_sentiment.predict(s)

    topSentiment = s.labels[0].value

    score = s.labels[0].score

    

    if score < threshold:

        topSentiment = 'neutral'   

    elif topSentiment =='POSITIVE':

        topSentiment = 'positive'

    elif topSentiment == 'NEGATIVE' :

        topSentiment = 'negative'

    

    return topSentiment
from tqdm.notebook import tqdm

tqdm.pandas()
%%time



#train['flair_text_SENT'] = train['text'].progress_apply(flairSent)  

train['flair_selected_text_SENT'] = train['selected_text'].progress_apply(flairSent,threshold=0.8)
frame = {'Real sentiment': train.sentiment.value_counts(),

         'NLTK_selected_text_SENT': train.NLTK_selected_text_SENT.value_counts(), 

         'NLTK_Full text_SENT': train.NLTK_text_SENT.value_counts(),

         

         'TB_selected_text_SENT': train.TB_selected_text_SENT.value_counts(),

         'TB_Full_text_SENT': train.TB_text_SENT.value_counts(),

         

         'flair_selected_text_SENT': train.flair_selected_text_SENT.value_counts()

         #'flair_full_text_SENT': train.flair_text_SENT.value_counts()

         

        } 

  

result = pd.DataFrame(frame).T

result.plot.bar(rot=45);
len(train[train['flair_selected_text_SENT'] ==train.sentiment])/len(train)
#len(train[train['flair_full_text_SENT'] ==train.sentiment])/len(train)
def cleantext():

    pass



def cleanstring():

    pass
#raise Exception("the end")