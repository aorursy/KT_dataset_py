# Let's import all the necessary libraries for this project



import requests

import numpy as np

import pandas as pd

import json

from datetime import datetime

import plotly.express as px

import textblob

from textblob.classifiers import NaiveBayesClassifier

import nltk

import sklearn

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns
df_final = pd.read_csv('../input/covid19-reddit-comments/covid19_comments.csv')

df_final.head()
df_final.isnull().values.any()
df_final.groupby('score')['score'].count().sort_values(ascending=False)
def clean_negatives(dataset):

    for i in range(-34, 0):

        dataset.replace(to_replace=i, value=1, inplace=True)

    return dataset



df_final['score'] = clean_negatives(df_final['score'])
def groupbydate(dataset):

    newdataset = pd.DataFrame()

    dates = []

    sums = []

    dataset['date'] = pd.to_datetime(dataset['date']).dt.date

    for date in dataset['date']:

        if date not in dates:

            dates.append(date)

            sums.append((dataset['date'] == date).sum())

    newdataset['date'] = dates

    newdataset['comments'] = sums

    return newdataset



progression = groupbydate(df_final)



import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(x=progression['date'], y=progression['comments']))



fig.update_layout(title='Number of comments about COVID-19 on Reddit per day',

                   xaxis_title='Date',

                   yaxis_title='Number of Comments')



fig.show()
df_final['month'] = pd.DatetimeIndex(df_final['date']).month

df_final.groupby('month')['month'].count().sort_values(ascending=False)
january_comments = df_final.loc[df_final['month'] == 1]

january_comments.to_csv(path_or_buf='january_comments.csv')



february_comments = df_final.loc[df_final['month'] == 2]

february_comments.to_csv(path_or_buf='february_comments.csv')



march_comments = df_final.loc[df_final['month'] == 3]

march_comments.to_csv(path_or_buf='march_comments.csv')



april_comments = df_final.loc[df_final['month'] == 4]

april_comments.to_csv(path_or_buf='april_comments.csv')
fig = px.scatter(january_comments, x="date", 

           y="sentiment_polarity",

           hover_data=["author", "permalink", "preview"], 

           color_discrete_sequence=["green", "red"],

           color="sentiment", 

           size="score", 

           size_max=50,

           labels={"sentiment_polarity": "Comment positivity", "date": "Date comment was posted"}, 

           title=f"Positivity of Reddit Comments about COVID-19 on January 2020")

fig.update_layout(

    autosize=False,

    width=800,

    height = 800,

    plot_bgcolor = 'white'

)

fig.show()
fig = px.scatter(february_comments, x="date", 

           y="sentiment_polarity",

           hover_data=["author", "permalink", "preview"],

           color_discrete_sequence=["green", "red"], 

           color="sentiment", 

           size="score", 

           size_max=50,

           labels={"sentiment_polarity": "Comment positivity", "date": "Date comment was posted"}, 

           title=f"Positivity of Reddit Comments about COVID-19 on February 2020",

          )

fig.update_layout(

    autosize=False,

    width=800,

    height = 800,

    plot_bgcolor = 'white'

)

fig.show()
fig = px.scatter(march_comments, x="date",

           y="sentiment_polarity",

           hover_data=["author", "permalink", "preview"], 

           color_discrete_sequence=["green", "red"], 

           color="sentiment", 

           size="score", 

           size_max=50,

           labels={"sentiment_polarity": "Comment positivity", "date": "Date comment was posted"}, 

           title=f"Positivity of Reddit Comments about COVID-19 on March 2020",

          )

fig.update_layout(

autosize=False,

    width=800,

    height = 800,

    plot_bgcolor = 'white')

fig.show()
fig = px.scatter(april_comments, x="date",

           y="sentiment_polarity",

           hover_data=["author", "permalink", "preview"], 

           color_discrete_sequence=["green", "red"], 

           color="sentiment", 

           size="score", 

           size_max=50,

           labels={"sentiment_polarity": "Comment positivity", "date": "Date comment was posted"}, 

           title=f"Positivity of Reddit Comments about COVID-19 on April 2020",

          )

fig.update_layout(

    autosize=False,

    width=800,

    height = 800,

    plot_bgcolor = 'white'

)

fig.show()
df_km = pd.DataFrame(df_final['score'])

df_km['sentiment_polarity'] = df_final['sentiment_polarity']



# ELBOW METHOD



# calculate distortion for a range of number of cluster

distortions = []

for i in range(1, 11):

    km = KMeans(

        n_clusters=i, init='random',

        n_init=10, max_iter=300,

        tol=1e-04, random_state=0

    )

    km.fit(df_km)

    distortions.append(km.inertia_)



# plot

plt.plot(range(1, 11), distortions, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Distortion')

plt.show()
kclusters = 4



km = KMeans(n_clusters=kclusters, init='random',n_init=10, max_iter=300, 

    tol=1e-04, random_state=0).fit(df_km)



km_labels = km.labels_



# Let's change label numbers so they go from highest scores to lowest



replace_labels = {0:2, 1:0, 2:3, 3:1}



for i in range(len(km_labels)):

    km_labels[i] = replace_labels[km_labels[i]]

    

df_km['Cluster'] = km_labels

df_km.head()
import matplotlib.ticker as ticker



fig, axes = plt.subplots(1, kclusters, figsize=(20, 10), sharey=True)



axes[0].set_ylabel('Score & Sentiment Polarity', fontsize=25)



for k in range(kclusters):

    # We are going to set same y axis limits

    axes[k].set_ylim(-1,500)

    axes[k].xaxis.set_label_position('top')

    axes[k].set_xlabel('Cluster ' + str(k), fontsize=25)

    axes[k].tick_params(labelsize=20)

    plt.sca(axes[k])

    plt.xticks(rotation='vertical')

    sns.boxplot(data = df_km[df_km['Cluster'] == k].drop('Cluster',1), ax=axes[k])



plt.show()
df_final['Cluster'] = df_km['Cluster']



df_cluster1 = df_final.loc[df_final['Cluster'] == 1]



#df_cluster1['date'] = df_cluster1['date'].apply(lambda x: float(x))



fig = px.scatter(df_cluster1, x="date",

           y="sentiment_polarity", 

           trendline = 'lowess',

           hover_data=["author", "permalink", "preview"], 

           color_discrete_sequence=["green", "red"], 

           color="sentiment", 

           size="score", 

           size_max=50,

           labels={"sentiment_polarity": "Comment positivity", "date": "Date comment was posted"}, 

           title=f"Sentiment of Reddit's most-voted comments on COVID-19",

          )

fig.update_layout(

    autosize=False,

    width=800,

    height = 800,

    plot_bgcolor = 'white'

)

fig.show()
data = pd.read_csv('../input/figure-eight-labelled-textual-dataset/text_emotion.csv')



# Let's drop unnecessary columns from our dataset

data = data.drop('tweet_id', axis=1)

data = data.drop('author', axis=1)
data['sentiment'].unique()
data = data.drop(data[data.sentiment == 'boredom'].index)

data = data.drop(data[data.sentiment == 'surprise'].index)

data = data.drop(data[data.sentiment == 'enthusiasm'].index)

data = data.drop(data[data.sentiment == 'empty'].index)

data = data.drop(data[data.sentiment == 'fun'].index)

data = data.drop(data[data.sentiment == 'relief'].index)

data = data.drop(data[data.sentiment == 'love'].index)

data = data.drop(data[data.sentiment == 'neutral'].index)
# We are going to merge the category 'hate' to the 'anger' category to make a data augmentation

data['sentiment'].replace(to_replace='hate', value='anger', inplace=True)

# Let's replace 'worry' for 'fear'

data['sentiment'].replace(to_replace='worry', value='fear', inplace=True)

# Let's see how many tweets we have for each sentiment

data.groupby('sentiment')['sentiment'].count().sort_values(ascending=False)
# Synonym replacement



from nltk.corpus import wordnet

from nltk.corpus import stopwords

stop = stopwords.words('english')

import random



def get_synonyms(word):

    """

    Get synonyms of a word

    """

    synonyms = set()

    

    for syn in wordnet.synsets(word): 

        for l in syn.lemmas(): 

            synonym = l.name().replace("_", " ").replace("-", " ").lower()

            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])

            synonyms.add(synonym) 

    

    if word in synonyms:

        synonyms.remove(word)

    

    return list(synonyms)



def synonym_replacement(words, n):

    

    words = words.split()

    

    new_words = words.copy()

    random_word_list = list(set([word for word in words if word not in stop]))

    random.shuffle(random_word_list)

    num_replaced = 0

    

    for random_word in random_word_list:

        synonyms = get_synonyms(random_word)

        

        if len(synonyms) >= 1:

            synonym = random.choice(list(synonyms))

            new_words = [synonym if word == random_word else word for word in new_words]

            num_replaced += 1

        

        if num_replaced >= n: #only replace up to n words

            break



    sentence = ' '.join(new_words)



    return sentence



# Random Insertion



def random_insertion(words, n):

    

    words = words.split()

    new_words = words.copy()

    

    for _ in range(n):

        add_word(new_words)

        

    sentence = ' '.join(new_words)

    return sentence



def add_word(new_words):

    

    synonyms = []

    counter = 0

    

    while len(synonyms) < 1:

        random_word = new_words[random.randint(0, len(new_words)-1)]

        synonyms = get_synonyms(random_word)

        counter += 1

        if counter >= 10:

            return

        

    random_synonym = synonyms[0]

    random_idx = random.randint(0, len(new_words)-1)

    new_words.insert(random_idx, random_synonym)
anger = data.loc[data['sentiment'] == 'anger']

new_anger_comments = []

for content in anger['content']:

    new_anger_comments.append(synonym_replacement(content, 4))

    new_anger_comments.append(random_insertion(content, 4))

new_anger = pd.DataFrame()

new_anger['content'] = new_anger_comments

new_anger['sentiment'] = 'anger'

anger = anger.append(new_anger)
anger.shape
data = data.append(new_anger)
#Making all letters lowercase

data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))



#Removing Punctuation, Symbols

data['content'] = data['content'].str.replace('[^\w\s]',' ')



#Removing urls, links

data['content'] = data['content'].str.replace('(www|http)\S+', ' ')



#Removing Stop Words using NLTK

from nltk.corpus import stopwords

stop = stopwords.words('english')

data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#Lemmatisation

from textblob import Word

data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



#Correcting Letter Repetitions

import re



def de_repeat(text):

    pattern = re.compile(r"(.)\1{2,}")

    return pattern.sub(r"\1\1", text)



data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
# Code to find the top 1000 rarest words appearing in the data

freq = pd.Series(' '.join(data['content']).split()).value_counts()[-1000:]



# Removing all those rarely appearing words from the data

freq = list(freq.index)

data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#Encoding output labels

from sklearn import preprocessing

lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(data.sentiment.values)



# Splitting into training and testing data in 90:10 ratio

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
print(lbl_enc.classes_)
# Extracting TF-IDF parameters

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))

X_train_tfidf = tfidf.fit_transform(X_train)

X_val_tfidf = tfidf.fit_transform(X_val)
# Extracting Count Vectors Parameters

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word')

count_vect.fit(data['content'])

X_train_count =  count_vect.transform(X_train)

X_val_count =  count_vect.transform(X_val)
from sklearn.metrics import accuracy_score

# Model 1: Multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_tfidf, y_train)

y_pred = nb.predict(X_val_tfidf)

print('naive bayes tfidf accuracy %s' % accuracy_score(y_pred, y_val))



# Model 2: Linear SVM

from sklearn.linear_model import SGDClassifier

lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)

lsvm.fit(X_train_tfidf, y_train)

y_pred = lsvm.predict(X_val_tfidf)

print('svm using tfidf accuracy %s' % accuracy_score(y_pred, y_val))



# Model 3: logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1, max_iter=21000)

logreg.fit(X_train_tfidf, y_train)

y_pred = logreg.predict(X_val_tfidf)

print('log reg tfidf accuracy %s' % accuracy_score(y_pred, y_val))



# Model 4: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500)

rf.fit(X_train_tfidf, y_train)

y_pred = rf.predict(X_val_tfidf)

print('random forest tfidf accuracy %s' % accuracy_score(y_pred, y_val))
## Model 1: Multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_count, y_train)

y_pred = nb.predict(X_val_count)

print('naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_val))



# Model 2: Linear SVM

from sklearn.linear_model import SGDClassifier

lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)

lsvm.fit(X_train_count, y_train)

y_pred = lsvm.predict(X_val_count)

print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_val))



# Model 3: Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1, max_iter=21000)

logreg.fit(X_train_count, y_train)

y_pred = logreg.predict(X_val_count)

print('log reg count vectors accuracy %s' % accuracy_score(y_pred, y_val))



# Model 4: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500)

rf.fit(X_train_count, y_train)

y_pred = rf.predict(X_val_count)

print('random forest with count vectors accuracy %s' % accuracy_score(y_pred, y_val))
comments = df_final['body']



# Doing some preprocessing on these comments as done before

comments = comments.apply(lambda x: " ".join(x.lower() for x in x.split()))

comments = comments.str.replace('[^\w\s]',' ')

comments = comments.str.replace('(www|http)\S+', ' ')

from nltk.corpus import stopwords

stop = stopwords.words('english')

comments = comments.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

from textblob import Word

comments = comments.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

comments = comments.apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

# Extracting Count Vectors feature from our Reddit comments

comment_count = count_vect.transform(comments)



#Predicting the emotion of the comment using our already trained logistic regression

comment_pred = lsvm.predict(comment_count)
df_final['emotion'] = comment_pred.tolist()

df_final['emotion'].replace(to_replace=0, value='anger', inplace=True)

df_final['emotion'].replace(to_replace=1, value='fear', inplace=True)

df_final['emotion'].replace(to_replace=2, value='happiness', inplace=True)

df_final['emotion'].replace(to_replace=3, value='sadness', inplace=True)

df_final.groupby('emotion')['emotion'].count().sort_values(ascending=False)
# Let's group comments by emotion

anger_comments = df_final.loc[df_final['emotion'] == 'anger']

fear_comments = df_final.loc[df_final['emotion'] == 'fear']

happiness_comments = df_final.loc[df_final['emotion'] == 'happiness']

sadness_comments = df_final.loc[df_final['emotion'] == 'sadness']



# Let's find out progressions of each emotion



anger_progression = groupbydate(anger_comments)

fear_progression = groupbydate(fear_comments)

happiness_progression = groupbydate(happiness_comments)

sadness_progression = groupbydate(sadness_comments)
fig = go.Figure() 



fig.add_trace(go.Scatter(x=anger_progression["date"], y=anger_progression['comments'], 

                        name= 'Anger'))

fig.add_trace(go.Scatter(x=fear_progression["date"], y=fear_progression['comments'],

                        name = 'Fear'))

fig.add_trace(go.Scatter(x=happiness_progression["date"], y=happiness_progression['comments'],

                        name = 'Happiness'))

fig.add_trace(go.Scatter(x=sadness_progression["date"], y=sadness_progression['comments'],

                        name = 'Sadness'))



fig.update_layout(title='Emotions towards COVID-19 on Reddit',

                   xaxis_title='Date',

                   yaxis_title='Number of Comments')



fig.show()