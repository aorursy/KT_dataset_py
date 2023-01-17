import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

import seaborn

import collections

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import accuracy_score
#Reading data

def read_data():    

    true_data = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")

    fake_data = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")

    true_data['fake'] = 0

    fake_data['fake'] = 1

    

    return pd.concat([fake_data, true_data]).reset_index(drop = True)



data = read_data()





data.head()
#fake news analyses

fake_news = data[data.fake == 1]









#common words from fake news



#some bad words.

bad_words = ['a','the','and',

             's','he','she',

             'have','has','to',

             'new','all','any',

             'as','of','to','on',

             'in','at','that',

             'this','is','are'

             'for','it','was',

             'were','for','with',

             'his','be','by',

             'are','is','they',

             'not','i','who','where',

             'from','t','we','they',

             'you','an','about','then',

             'her','or','what','will',

             'but','would','been',

             'their','if','people',

             'when','out','had',

             'one','said','more','just',

             'our','can','so','there',

             'which','like','no','after',

             'up','because','also','do','how',

             'than','even','over','into','other',

             'him','only','news','being','us','some','against',

             're','should','state','these',

             'get','them','could','don',

             'time','now','its','going','while',

             'many','first','most','via','make','told',

             'my','very','those','your','during','house',

             'did','made','know','two','think','before',

             'last','may','back'

            ] 





words = []

common_words = []

for news in fake_news.text:

    for word in news.split():

        lower_word = word.lower()

        if lower_word not in bad_words:

            words.append(lower_word)

    

for item,count in collections.Counter(words).items():

    if count > 5000:

        common_words.append({'word':item,'count':count})



top_commons = sorted(common_words, key=lambda w: w['count'],reverse=True) 



#VISUALIZATONS



#TOP 10 COMMON WORDS FROM FAKE NEWS

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))



ax1.title.set_text('TOP 10 COMMON WORDS FROM FAKE NEWS')

plt.setp(ax1.xaxis.get_majorticklabels(), rotation='vertical')

ax1.bar([data['word'] for data in top_commons[0:10]],

        [data['count'] for data in top_commons[0:10]])



#fake news subject counts

fake_news_acc_subject = fake_news.subject



ax2.title.set_text('FAKE NEWS SUBJECT COUNTS')

ax2 = fake_news_acc_subject.value_counts().plot(kind='bar');

#FAKE NEW CLASSIFICATION





#logistic regression

x_train,x_test,y_train,y_test = train_test_split(data['text'], data['fake'], test_size=0.2, random_state=40)



pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LogisticRegression())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,4)))