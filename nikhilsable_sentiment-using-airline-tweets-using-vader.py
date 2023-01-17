import pandas as pd #Importing the PANDAS python library

import numpy as np #importing Numpy

%matplotlib inline 



#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #initiating VADER instance



import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer



analyser = SentimentIntensityAnalyzer()
#https://www.kaggle.com/crowdflower/twitter-airline-sentiment



sentences = pd.read_csv('../input/Tweets.csv')



len(sentences)
sentences.columns #I dont need all the columns for this demo
sentences.head()
sentences.groupby(['airline', 'airline_sentiment']).size().unstack().plot(kind='bar',figsize=(11, 5))
sentences = sentences[['airline_sentiment', 'airline','text' ]] #this is all I need

sentences.head()
sentences = sentences[sentences['airline']=='United'] #filtering dataset for United

print(len(sentences))

sentences = sentences.reset_index(drop = True)

sentences.head(10)
sentences.groupby('airline_sentiment').size().plot(kind='bar')
def print_sentiment_scores(sentence):

    snt = analyser.polarity_scores(sentence)  #Calling the polarity analyzer

    print("{:-<40} {}".format(sentence, str(snt)))
print_sentiment_scores("United flight was a bad experience") #Compound value scale = -1 to 1 (-ve to +ve)
%time   #to calulate the time it takes the algorithm to compute a VADER score



i=0 #counter



compval1 = [ ]  #empty list to hold our computed 'compound' VADER scores





while (i<len(sentences)):



    k = analyser.polarity_scores(sentences.iloc[i]['text'])

    compval1.append(k['compound'])

    

    i = i+1

    

#converting sentiment values to numpy for easier usage



compval1 = np.array(compval1)



len(compval1)
sentences['VADER score'] = compval1
sentences.head(20)
%time



#Assigning score categories and logic

i = 0



predicted_value = [ ] #empty series to hold our predicted values



while(i<len(sentences)):

    if ((sentences.iloc[i]['VADER score'] >= 0.7)):

        predicted_value.append('positive')

        i = i+1

    elif ((sentences.iloc[i]['VADER score'] > 0) & (sentences.iloc[i]['VADER score'] < 0.7)):

        predicted_value.append('neutral')

        i = i+1

    elif ((sentences.iloc[i]['VADER score'] <= 0)):

        predicted_value.append('negative')

        i = i+1

        
sentences['predicted sentiment'] = predicted_value
len(sentences['predicted sentiment'])
sentences.head(20)
madeit = sentences[sentences['airline_sentiment']== sentences['predicted sentiment']]
len(madeit)/len(sentences)
madeit.head(20)


sentences.groupby('predicted sentiment').size().plot(kind='bar')
didntmakeit = sentences[sentences['airline_sentiment'] != sentences['predicted sentiment']]
didntmakeit.reset_index(drop=True, inplace=True)

didntmakeit.head(20)
didntmakeit.iloc[8]
didntmakeit.iloc[8]['text']
from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt 
df = madeit[madeit['predicted sentiment']=='negative']



words = ' '.join(df['text'])

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])



stopwords = set(STOPWORDS)

stopwords.add("amp")

stopwords.add("flight")

stopwords.add("united")

stopwords.add("plane")

stopwords.add("now")



wordcloud = WordCloud(stopwords=stopwords,

                      background_color='black',

                      width=3000,

                      height=2500

                     ).generate(cleaned_word)
type(cleaned_word)
plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
df = madeit[madeit['predicted sentiment']=='positive']



words = ' '.join(df['text'])

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                                and word !='&amp'

                            ])



stopwords = set(STOPWORDS)

stopwords.add("amp")

stopwords.add("flight")

stopwords.add("flights")

stopwords.add("united")

stopwords.add("plane")



wordcloud = WordCloud(stopwords=stopwords,

                      background_color='black',

                      width=3000,

                      height=2500

                     ).generate(cleaned_word)
plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()