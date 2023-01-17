# Principal ML and visualizations libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

import plotly.graph_objects as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



sns.set()



%matplotlib inline
dataset = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

dataset.head()
Labels = pd.DataFrame(dataset['Liked'].value_counts()).reset_index()

Labels.columns = ['Liked','Total']

Labels['Liked'] = Labels['Liked'].map({1: 'Positive', 0: 'Negative'})



fig = px.pie(Labels, values = 'Total', names = 'Liked', title='Percentage of reviews', hole=.4, color = 'Liked',

             width=800, height=400)

fig.show()
positive = dataset[dataset["Liked"] == 1][["Review", "Liked"]]
plt.subplots(figsize=(16,13))

wordcloud = WordCloud(

                          background_color='black',max_words = 10000,

                          width=1500, stopwords=STOPWORDS,

                          height=1080

                         ).generate(" ".join(positive.Review))

plt.title("Positive Reviews", fontsize=20)

plt.imshow(wordcloud.recolor( colormap= 'viridis'))

plt.axis('off')

plt.show()
negative = dataset[dataset["Liked"] == 0][["Review", "Liked"]]



plt.subplots(figsize=(16,13))

wordcloud = WordCloud(

                          background_color='black',max_words = 1000, 

                          width=1500, stopwords=STOPWORDS,

                          height=1080

                         ).generate(" ".join(negative.Review))

plt.title("Negative Reviews", fontsize=20)

plt.imshow(wordcloud.recolor( colormap= 'Pastel2'))

plt.axis('off')

plt.show()
# NLP libraries



import re # Regular expressions

import nltk # Natural language tool kit

from nltk.corpus import stopwords # This will help us get rid of useless words.



# Extra needed packages



nltk.download('punkt') 

nltk.download('stopwords')

nltk.download('wordnet')
first_review = dataset['Review'][0]

first_review
first_review = first_review.lower()

first_review = re.sub("[^a-zA-Z]", " ", first_review)

first_review
# Tokenization



first_review_list = nltk.word_tokenize(first_review)

first_review_list
stopwords = nltk.corpus.stopwords.words('english')



first_review_list_cleaned = [word for word in first_review_list if word.lower() not in stopwords]

print(first_review_list_cleaned)
from nltk.stem import PorterStemmer



stemmer = PorterStemmer()

stemmer.stem('lover'), stemmer.stem('cellphone')
for w in first_review_list_cleaned:

        print(stemmer.stem(w))
from nltk.stem import WordNetLemmatizer 



lemma = WordNetLemmatizer()

lemma.lemmatize('lover'), lemma.lemmatize('cellphone')
for w in first_review_list_cleaned:

        print(lemma.lemmatize(w))
corpus=[]



for review in dataset['Review']:

    review = review.lower()

    review = re.sub("[^a-zA-Z]", " ", review)

    review = nltk.word_tokenize(review)

    review = [word for word in review if word.lower() not in stopwords]

    lemma = WordNetLemmatizer()

    review = [lemma.lemmatize(word) for word in review]

    review = " ".join(review) # Building again the dataframe

    corpus.append(review)

    

dataset["Review"] = corpus

dataset.head(5)
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer()

bag_of_words = cv.fit_transform(corpus).toarray()

bag_of_words
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
y = dataset.iloc[:,1].values



X_train, X_test, y_train, y_test = train_test_split(bag_of_words, y, test_size = 0.20, random_state = 0)
def clf_model(model):

    clf = model

    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test).round())

    recall = recall_score(y_test, clf.predict(X_test).round())

    precision = precision_score(y_test, clf.predict(X_test).round())

    return clf, accuracy, recall, precision
model_performance = pd.DataFrame(columns = ["Model", "Accuracy", "Recall", "Precision"])



models_to_evaluate = [DecisionTreeClassifier(), LogisticRegression(), RandomForestClassifier(n_estimators=1000),

                      KNeighborsClassifier(n_neighbors = 7, metric = "minkowski", p = 2),

                      SVC(kernel = 'rbf'), GaussianNB(), XGBClassifier(n_estimators=300, learning_rate=0.01)]



for model in models_to_evaluate:

    clf, accuracy, recall, precision = clf_model(model)

    model_performance = model_performance.append({"Model": model, "Accuracy": accuracy,

                                                  "Recall": recall, "Precision": precision}, ignore_index=True)



model_performance
my_reviews = {'Review': ["I highly recommend the restaurant.","I will never go back!!", 

                        "Disgusting food and poor service.","Lovely evening and delicious dessert."]}

my_reviewsDF = pd.DataFrame.from_dict(my_reviews)

my_reviewsDF
corpus=[]



for review in my_reviewsDF['Review']:

    review = review.lower()

    review = re.sub("[^a-zA-Z]", " ", review)

    review = nltk.word_tokenize(review)

    review = [word for word in review if word.lower() not in stopwords]

    lemma = WordNetLemmatizer()

    review = [lemma.lemmatize(word) for word in review]

    review = " ".join(review) # Building again the dataframe

    corpus.append(review)

    

my_reviewsDF["Review"] = corpus



bag_of_words = cv.transform(corpus).toarray() # Using the same CV as before!!
LR = LogisticRegression()

LR.fit(X_train, y_train)



Label = LR.predict(bag_of_words)



Label = pd.DataFrame({'Label':Label})

joined = my_reviewsDF.join(Label)

joined
def unigram(corpus, n=None):

    cv = CountVectorizer().fit(corpus)

    bag_of_words = cv.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



def bigram(corpus, n=None):

    cv = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = cv.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]





def trigram(corpus, n=None):

    cv = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = cv.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
positive = dataset[dataset["Liked"] == 1][["Review", "Liked"]]
pos_uni = unigram(positive['Review'], 20)

temp = pd.DataFrame(pos_uni, columns = ['words' ,'count'])

fig = px.bar(temp, x = 'words', y = 'count', color = 'words', title='Top 20 unigrams in positive reviews')        

fig.show()
pos_bi = bigram(positive['Review'], 20)

temp = pd.DataFrame(pos_bi, columns = ['words' ,'count'])

fig = px.bar(temp, x = 'words', y = 'count', color = 'words', title='Top 20 bigrams in positive reviews')        

fig.show()
pos_tri = trigram(positive['Review'], 20)

temp = pd.DataFrame(pos_tri, columns = ['words' ,'count'])

fig = px.bar(temp, x = 'words', y = 'count', color = 'words', title='Top 20 trigrams in positive reviews')        

fig.show()
negative = dataset[dataset["Liked"] == 0][["Review", "Liked"]]

neg_uni = unigram(negative['Review'], 20)

temp = pd.DataFrame(neg_uni, columns = ['words' ,'count'])
fig = go.Figure(data =[go.Bar(x = temp['words'].tolist(), y= temp['count'].tolist())])

fig.update_traces(marker_color='rgb(0,0,139)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Top 20 unigrams in negative reviews')

fig.show()
neg_bi = bigram(negative['Review'], 20)

temp = pd.DataFrame(neg_bi, columns = ['words' ,'count'])



fig = go.Figure(data =[go.Bar(x = temp['words'].tolist(), y= temp['count'].tolist())])

fig.update_traces(marker_color='rgb(0,0,139)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Top 20 bigrams in negative reviews')

fig.show()
neg_tri = trigram(negative['Review'], 20)

temp = pd.DataFrame(neg_tri, columns = ['words' ,'count'])



fig = go.Figure(data =[go.Bar(x = temp['words'].tolist(), y= temp['count'].tolist())])

fig.update_traces(marker_color='rgb(0,0,139)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Top 20 trigrams in negative reviews')

fig.show()
!pip install NRCLex
from nrclex import NRCLex
text_object = NRCLex(' '.join(dataset['Review']))



EmotionDF = pd.DataFrame.from_dict(text_object.affect_frequencies, orient='index').sort_values(by=0, ascending=False).reset_index()

EmotionDF.columns = ['Emotion', 'Frequency']
fig = px.pie(EmotionDF, values = 'Frequency', names='Emotion',

             title='Emotion Frequency',

             hover_data=['Emotion'], labels={'Emotion':'Emotion'})

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()