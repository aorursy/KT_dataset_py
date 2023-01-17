# ! pip install textblob

from textblob import TextBlob

import nltk

nltk.download('vader_lexicon')

import numpy as np 

import pandas as pd 
feedback = "So I've been going to chili's since it opened but today the food was so bad , it had zero taste , the mushroom sauce chicken didn't have any mushrooms or sauce , the baked potatoes were without any seasoning and the beans were soggy , the burger's bun was weird tasting so was the patty and the onion rings in it . One star is for the chips and salsa , thankfully it's still the same"

testimonial = TextBlob(feedback)

testimonial.sentiment

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

def get_vader_score(sent):

    # Polarity score returns dictionary

    ss = sid.polarity_scores(sent)

    return ss

get_vader_score(feedback)        
# !ls ../input/sentimentanalysis/amazon_cells_labelled.txt

df = pd.read_csv('../input/sentimentanalysis/amazon_cells_labelled.txt',sep='\t', header=None, names=['Text', 'Sentiment'])

df.head()
def get_tb_sentiment(text, param):

#     print(text,param)

    textblob = TextBlob(text)

    if param =='polarity':

        return textblob.sentiment.polarity

    elif param == 'subjectivity':

        return textblob.sentiment.subjectivity

    else:

        return None



def get_vader_sentiment(text, param):

    vader = get_vader_score(text)

    if param =='pos':

        return vader['pos']

    elif param == 'neg':

        return vader['neg']

    elif param == 'neu':

        return vader['neu']

    elif param == 'compound':

        return vader['compound']

    return None





df['tb_polarity'] = df['Text'].apply(get_tb_sentiment, args=('polarity',))

df['tb_subjectivity'] = df['Text'].apply(get_tb_sentiment, args=('subjectivity',))

df['vader_pos'] = df['Text'].apply(get_vader_sentiment, args=('pos',))

df['vader_neg'] = df['Text'].apply(get_vader_sentiment, args=('neg',))

df['vader_neu'] = df['Text'].apply(get_vader_sentiment, args=('neu',))

df['vader_compound'] = df['Text'].apply(get_vader_sentiment, args=('compound',))



df.head()

import seaborn as sns

import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(20,20))



row = 2

col = 2



plt.subplot(row, col, 1)

sns.scatterplot(x='vader_compound', y='tb_polarity',data=df, hue='Sentiment')

plt.subplot(row, col, 2)

sns.scatterplot(x='vader_pos', y='tb_polarity',data=df, hue='Sentiment')

plt.subplot(row, col, 3)

sns.scatterplot(x='vader_neg', y='tb_polarity',data=df, hue='Sentiment')

plt.subplot(row, col, 4)

sns.scatterplot(x='vader_compound', y='tb_subjectivity',data=df, hue='Sentiment')



from sklearn import tree

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X = df[['tb_polarity', 'tb_subjectivity', 'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']]

y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = tree.DecisionTreeClassifier(min_samples_split=200)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

score = accuracy_score(y_test, predictions)

print('Model Classification Score:' , score)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = SVC(gamma='auto')

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

score = accuracy_score(y_test, predictions)

print('Model Classification Score:' , score)
feedback_test = pd.read_csv("../input/feedback-test1csv/feedback_test.csv")

len(feedback_test)
feedback_test['tb_polarity'] = feedback_test['Feedback'].apply(get_tb_sentiment, args=('polarity',))

feedback_test['tb_subjectivity'] = feedback_test['Feedback'].apply(get_tb_sentiment, args=('subjectivity',))

feedback_test['vader_pos'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('pos',))

feedback_test['vader_neg'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('neg',))

feedback_test['vader_neu'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('neu',))

feedback_test['vader_compound'] = feedback_test['Feedback'].apply(get_vader_sentiment, args=('compound',))



feedback_test.tail()

feedback_test1 = feedback_test[['tb_polarity', 'tb_subjectivity', 'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']]

predictions = clf.predict(feedback_test1)

predictions