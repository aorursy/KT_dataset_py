import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')
t = open("../input/call-of-cthulhu-text-practice/The Call of Cthulhu.txt", "r")

t = t.read()



print(t[0:1000])
#eliminem els salts de linia i merdes

t = t.replace('\n',' ')
sentences=[]





for sentence in t.split('. '):

    if len(sentence) > 30:

        sentences.append(sentence)

    else:

        pass



len(sentences)
words_sentence = []

taula=[]



for sentence in sentences:

    words_sentence.append(len(re.findall(r'\w+', sentence)))

    #print(len(re.findall(r'\w+', sentence)))

    taula.append([len(re.findall(r'\w+', sentence)),sentence])

    

taula[5:10]

df = pd.DataFrame(taula, columns=['Word Count','Sentences'])

df.sort_values(['Word Count'], inplace=True, ascending=False)

df.head(5)
plt.hist(words_sentence, bins = 15)

plt.ylabel('counts')

plt.xlabel('paraules per frase')

plt.show()
!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

  

# function to print sentiments 

# of the sentence. 

def sentiment_scores(sentence): 

  

    # Create a SentimentIntensityAnalyzer object. 

    sid_obj = SentimentIntensityAnalyzer() 

  

    # polarity_scores method of SentimentIntensityAnalyzer 

    # oject gives a sentiment dictionary. 

    # which contains pos, neg, neu, and compound scores. 

    sentiment_dict = sid_obj.polarity_scores(sentence) 

      

    print("Overall sentiment dictionary is : ", sentiment_dict) 

    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 

    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 

    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 

  

    print("Sentence Overall Rated As", end = " ") 

  

    # decide sentiment as positive, negative and neutral 

    if sentiment_dict['compound'] >= 0.15 : 

        print("Positive") 

  

    elif sentiment_dict['compound'] <= - 0.15 : 

        print("Negative") 

  

    else : 

        print("Neutral") 

        

n=39

        

print(sentiment_scores(sentences[n]))

print(sentences[n])

from tqdm import tqdm



negl = []

neul = []

posl = []



def sentiment_return(sentence):  

    sid_obj = SentimentIntensityAnalyzer() 

    sentiment_dict = sid_obj.polarity_scores(sentence) 

    neg = sentiment_dict['neg']*100

    neu = sentiment_dict['neu']*100

    pos = sentiment_dict['pos']*100

    return neg,neu,pos



for i in tqdm(sentences):

    a,b,c = sentiment_return(i)

    negl.append(a)

    neul.append(b)

    posl.append(c)

    

negatiu = sum(negl)

positiu = sum(posl)

neutre = sum(neul)



print(len(posl),len(neul),len(negl))



print ('the author is',int((positiu-negatiu)),'a happy writer')
a = (positiu, negatiu)

labels = ('positive', 'negative')

y_pos = np.arange(len(a))



plt.bar(y_pos,a)

plt.title('Was Lovercraft a Happy writer?')

plt.xticks(y_pos, labels)

plt.show()
a = (positiu, negatiu)

a