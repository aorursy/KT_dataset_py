"""

!pip install comet_ml

from comet_ml import Experiment



# Setting the API key (saved as environment variable)

experiment = Experiment(api_key="THysD8zqvW8wCiFTidV67jLP2",

                        project_name="climate-change-belief-analysis", 

                        workspace="jamakasilwane")                      

"""
# Standard libraries

import re

import csv

import nltk

import spacy

import string

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 



# Style

import matplotlib.style as style 

sns.set(font_scale=1.5)

style.use('seaborn-pastel')

style.use('seaborn-poster')

from PIL import Image

from wordcloud import WordCloud



# Downloads

nlp = spacy.load('en')

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')



# Preprocessing

import en_core_web_sm

from collections import Counter

from nltk.probability import FreqDist

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords, wordnet  

from sklearn.feature_extraction.text import CountVectorizer   

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer 

from sklearn.model_selection import train_test_split, RandomizedSearchCV



# Building classification models

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



# Model evaluation

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
# import dataset 

train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')

sample = pd.read_csv('../input/climate-change-belief-analysis/sample_submission.csv')
# Taking general look at both datasets



print(train.shape)

print(test.shape)



display(train.head())

display(test.head())



percent_duplicates = round((1-(train['message'].nunique()/len(train['message'])))*100,2)

print('Duplicated tweets in train data:')

print(percent_duplicates,'%')
def update(df):

    

    """

    This function creates a copy of the original train data and 

    renames the classes, converting them from numbers to words

    

    Input: 

    df: original dataframe

        datatype: dataframe

    

    Output:

    df: modified dataframe

        datatype: dataframe 

        

    """



    df = train.copy()

    sentiment = df['sentiment']

    word_sentiment = []



    for i in sentiment :

        if i == 1 :

            word_sentiment.append('Pro')

        elif i == 0 :

            word_sentiment.append('Neutral')

        elif i == -1 :

            word_sentiment.append('Anti')

        else :

            word_sentiment.append('News')



    df['sentiment'] = word_sentiment

    

    return df



df = update(train)

df.head()
def hashtag_extract(tweet):

    

    """

    This function takes in a tweet and extracts the top 15 hashtag(s) using regular expressions

    These hashtags are stored in a seperate dataframe 

    along with a count of how frequenty they occur

    

    Input:

    tweet: original tweets

           datatype: 'str'

           

    Output:

    hashtag_df: dataframe containing the top hashtags in the tweets

              datatype: dataframe         

    """

    

    hashtags = []

    

    for i in tweet:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)

        

    hashtags = sum(hashtags, [])

    frequency = nltk.FreqDist(hashtags)

    

    hashtag_df = pd.DataFrame({'hashtag': list(frequency.keys()),

                       'count': list(frequency.values())})

    hashtag_df = hashtag_df.nlargest(15, columns="count")



    return hashtag_df



# Extracting the hashtags from tweets in each class

pro = hashtag_extract(df['message'][df['sentiment'] == 'Pro'])

anti = hashtag_extract(df['message'][df['sentiment'] == 'Anti'])

neutral = hashtag_extract(df['message'][df['sentiment'] == 'Neutral'])

news = hashtag_extract(df['message'][df['sentiment'] == 'News'])



pro.head()
def TweetCleaner(tweet):

    

    """

    This function uses regular expressions to remove url's, mentions, hashtags, 

    punctuation, numbers and any extra white space from tweets after converting 

    everything to lowercase letters.



    Input:

    tweet: original tweet

           datatype: 'str'



    Output:

    tweet: modified tweet

           datatype: 'str'

    """

    # Convert everything to lowercase

    tweet = tweet.lower() 

    

    # Remove mentions   

    tweet = re.sub('@[\w]*','',tweet)  

    

    # Remove url's

    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)

    

    # Remove hashtags

    tweet = re.sub(r'#\w*', '', tweet)    

    

    # Remove numbers

    tweet = re.sub(r'\d+', '', tweet)  

    

    # Remove punctuation

    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)

    

    # Remove that funny diamond

    tweet = re.sub(r"U+FFFD ", ' ', tweet)

    

    # Remove extra whitespace

    tweet = re.sub(r'\s\s+', ' ', tweet)

    

    # Remove space in front of tweet

    tweet = tweet.lstrip(' ')                        

    

    return tweet



# Clean the tweets in the message column

df['message'] = df['message'].apply(TweetCleaner)

df['message'] = df['message'].apply(TweetCleaner)



df.head()
def lemma(df):

    

    """

    This function modifies the original train dataframe.

    A new column for the length of each tweet is added.

    The tweets are then tokenized and each word is assigned a part of speech tag 

    before being lemmatized

    

    Input:

    df: original dataframe

        datatype: dataframe 

        

    Output:

    df: modified dataframe

        datatype: dataframe

    """

    

    df['length'] = df['message'].str.len()

    df['tokenized'] = df['message'].apply(word_tokenize)

    df['pos_tags'] = df['tokenized'].apply(nltk.tag.pos_tag)



    def get_wordnet_pos(tag):



        if tag.startswith('J'):

            return wordnet.ADJ



        elif tag.startswith('V'):

            return wordnet.VERB



        elif tag.startswith('N'):

            return wordnet.NOUN



        elif tag.startswith('R'):

            return wordnet.ADV

    

        else:

            return wordnet.NOUN

        

    wnl = WordNetLemmatizer()

    df['pos_tags'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

    df['lemmatized'] = df['pos_tags'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

    df['lemmatized'] = [' '.join(map(str, l)) for l in df['lemmatized']]  

    return df



df = lemma(df)

df.head()
def frequency(tweet):

    

    """

    This function determines the frequency of each word in a collection of tweets 

    and stores the 25 most frequent words in a dataframe, 

    sorted from most to least frequent

    

    Input: 

    tweet: original tweets

           datatype: 'str'

           

    Output: 

    frequency: dataframe containing the top 25 words 

               datatype: dataframe          

    """

    

    # Count vectorizer excluding english stopwords

    cv = CountVectorizer(stop_words='english')

    words = cv.fit_transform(tweet)

    

    # Count the words in the tweets and determine the frequency of each word

    sum_words = words.sum(axis=0)

    words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    

    # Create a dataframe to store the top 25 words and their frequencies

    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

    frequency = frequency.head(25)

    

    return frequency



# Extract the top 25 words in each class

pro_frequency = frequency(df['lemmatized'][df['sentiment']=='Pro'])

anti_frequency = frequency(df['lemmatized'][df['sentiment']=='Anti'])

news_frequency = frequency(df['lemmatized'][df['sentiment']=='News'])

neutral_frequency = frequency(df['lemmatized'][df['sentiment']=='Neutral'])
# Extract the words in the tweets for the pro and anti climate change classes 

anti_words = ' '.join([text for text in anti_frequency['word']])

pro_words = ' '.join([text for text in pro_frequency['word']])

news_words = ' '.join([text for text in news_frequency['word']])

neutral_words = ' '.join([text for text in neutral_frequency['word']])



# Create wordcloud for the anti climate change class

anti_wordcloud = WordCloud(width=800, 

                           height=500, 

                           random_state=110, 

                           max_font_size=110, 

                           background_color='white',

                           colormap="Reds").generate(anti_words)



# Create wordcolud for the pro climate change class

pro_wordcloud = WordCloud(width=800, 

                          height=500, 

                          random_state=73, 

                          max_font_size=110, 

                          background_color='white',

                          colormap="Greens").generate(pro_words)



# Create wordcolud for the news climate change class

news_wordcloud = WordCloud(width=800, 

                          height=500, 

                          random_state=0, 

                          max_font_size=110, 

                          background_color='white',

                          colormap="Blues").generate(news_words)



# Create wordcolud for the neutral climate change class

neutral_wordcloud = WordCloud(width=800, 

                          height=500, 

                          random_state=10, 

                          max_font_size=110, 

                          background_color='white',

                          colormap="Oranges").generate(neutral_words)



pro_frequency.tail()
def entity_extractor(tweet):

    

    """

    This function extracts the top 10 people, organizations and geopolitical entities 

    in a collection of tweets. 

    The information is then saved in a new dataframe



    Input:

    tweet: lemmatized tweets

           datatype: 'str'



    Output:

    df: dataframe containing the top 10 people, organizations and gpe's in a collection of tweets

        datatype: dataframe ('str')

    """

    

    def get_people(tweet):  

        words = nlp(tweet)

        people = [w.text for w in words.ents if w.label_== 'PERSON']

        return people

    

    def get_org(tweet):

        words = nlp(tweet)

        org = [w.text for w in words.ents if w.label_== 'ORG']

        return org

    

    def get_gpe(tweet):

        words = nlp(tweet)

        gpe = [w.text for w in words.ents if w.label_== 'GPE']

        return gpe

    

    # Extract the top 10 people

    people = tweet.apply(lambda x: get_people(x)) 

    people = [x for sub in people for x in sub]

    people_counter = Counter(people)

    people_count = people_counter.most_common(10)

    people, people_count = map(list, zip(*people_count))

    

    # Extract the top 10 organizations

    org = tweet.apply(lambda x: get_org(x)) 

    org = [x for sub in org for x in sub]

    org_counter = Counter(org)

    org_count = org_counter.most_common(10)

    org, org_count = map(list, zip(*org_count))

    

    # Extract the top 10 geopolitical entities

    gpe = tweet.apply(lambda x: get_gpe(x)) 

    gpe = [x for sub in gpe for x in sub]

    gpe_counter = Counter(gpe)

    gpe_count = gpe_counter.most_common(10)

    gpe, gpe_count = map(list, zip(*gpe_count))

    

    # Create a dataframe to store the information

    df = pd.DataFrame({'people' : people})

    df['geopolitics'] = gpe

    df['organizations'] = org

    

    return df



# Extract top entities for each class

anti_info = entity_extractor(df['lemmatized'][df['sentiment']=='Anti'])

pro_info = entity_extractor(df['lemmatized'][df['sentiment']=='Pro'])

news_info = entity_extractor(df['lemmatized'][df['sentiment']=='News'])

neutral_info = entity_extractor(df['lemmatized'][df['sentiment']=='Neutral'])
# Display target distribution

style.use('seaborn-pastel')



fig, axes = plt.subplots(ncols=2, 

                         nrows=1, 

                         figsize=(20, 10), 

                         dpi=100)



sns.countplot(df['sentiment'], ax=axes[0])



labels=['Pro', 'News', 'Neutral', 'Anti'] 



axes[1].pie(df['sentiment'].value_counts(),

            labels=labels,

            autopct='%1.0f%%',

            shadow=True,

            startangle=90,

            explode = (0.1, 0.1, 0.1, 0.1))



fig.suptitle('Tweet distribution', fontsize=20)

plt.show()
# Plot the distribution of the length tweets for each class using a box plot

sns.boxplot(x=df['sentiment'], y=df['length'], data=df, palette=("Blues_d"))

plt.title('Tweet length for each class')

plt.show()
# Plot pro and anti wordclouds next to one another for comparisson

f, axarr = plt.subplots(2,2, figsize=(35,25))

axarr[0,0].imshow(pro_wordcloud, interpolation="bilinear")

axarr[0,1].imshow(anti_wordcloud, interpolation="bilinear")

axarr[1,0].imshow(neutral_wordcloud, interpolation="bilinear")

axarr[1,1].imshow(news_wordcloud, interpolation="bilinear")



# Remove the ticks on the x and y axes

for ax in f.axes:

    plt.sca(ax)

    plt.axis('off')



axarr[0,0].set_title('Pro climate change\n', fontsize=35)

axarr[0,1].set_title('Anti climate change\n', fontsize=35)

axarr[1,0].set_title('Neutral\n', fontsize=35)

axarr[1,1].set_title('News\n', fontsize=35)

#plt.tight_layout()

plt.show()



print("Pro climate change buzzwords 20-25 shown here for clarity \n- The wordcloud doesn't seem to pick up on 'http'")

display(pro_frequency.tail())
# Plot the frequent hastags for pro and anti climate change classes

sns.barplot(data=pro,y=pro['hashtag'], x=pro['count'], palette=("Blues_d"))

plt.title('Frequent PRO climate change hashtags')

plt.tight_layout()
sns.barplot(data=anti,y=anti['hashtag'], x=anti['count'], palette=("Blues_d"))

plt.title('Frequent ANTI climate change hashtags')

plt.tight_layout()
# Plot the frequent hastags for the news and neutral classes

sns.barplot(y=news['hashtag'], x=news['count'], palette=("Blues_d"))

plt.title('Frequent climate change NEWS hashtags')

plt.tight_layout()
sns.barplot(y=neutral['hashtag'], x=neutral['count'], palette=("Blues_d"))

plt.title('Frequent NEUTRAL climate change hashtags')

plt.tight_layout()
print('Pro climate change information')

display(pro_info.head(9))

print('Anti climate change information')

display(anti_info)
# Split the dataset into train & validation (25%) for model training



# Seperate features and tagret variables

X = train['message']

y = train['sentiment']



# Split the train data to create validation dataset

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)
# Random Forest Classifier

rf = Pipeline([('tfidf', TfidfVectorizer()),

               ('clf', RandomForestClassifier(max_depth=5, 

                                              n_estimators=100))])



# Naïve Bayes:

nb = Pipeline([('tfidf', TfidfVectorizer()),

               ('clf', MultinomialNB())])



# K-NN Classifier

knn = Pipeline([('tfidf', TfidfVectorizer()),

                ('clf', KNeighborsClassifier(n_neighbors=5, 

                                             metric='minkowski', 

                                             p=2))])



# Logistic Regression

lr = Pipeline([('tfidf',TfidfVectorizer()),

               ('clf',LogisticRegression(C=1, 

                                         class_weight='balanced', 

                                         max_iter=1000))])

# Linear SVC:

lsvc = Pipeline([('tfidf', TfidfVectorizer()),

                 ('clf', LinearSVC(class_weight='balanced'))])
# Random forest 

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_valid)



# Niave bayes

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_valid)



# K - nearest neighbors

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_valid)



# Linear regression

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_valid)



# Linear SVC

lsvc.fit(X_train, y_train)

y_pred_lsvc = lsvc.predict(X_valid)
# Generate a classification Report for the random forest model

print(metrics.classification_report(y_valid, y_pred_rf))



# Generate a normalized confusion matrix

cm = confusion_matrix(y_valid, y_pred_rf)

cm_norm = cm / cm.sum(axis=1).reshape(-1,1)



# Display the confusion matrix as a heatmap

sns.heatmap(cm_norm, 

            cmap="YlGnBu", 

            xticklabels=rf.classes_, 

            yticklabels=rf.classes_, 

            vmin=0., 

            vmax=1., 

            annot=True, 

            annot_kws={'size':10})



# Adding headings and lables

plt.title('Random forest classification')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()

# Generate a classification Report for the Naive Bayes model

print(metrics.classification_report(y_valid, y_pred_nb))



# Generate a normalized confusion matrix

cm = confusion_matrix(y_valid, y_pred_nb)

cm_norm = cm / cm.sum(axis=1).reshape(-1,1)



# Display the confusion matrix as a heatmap

sns.heatmap(cm_norm, 

            cmap="YlGnBu", 

            xticklabels=nb.classes_, 

            yticklabels=nb.classes_, 

            vmin=0., 

            vmax=1., 

            annot=True, 

            annot_kws={'size':10})



# Adding headings and lables

plt.title('Naive Bayes classification')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()

# Generate a classification Report for the K-nearest neighbors model

print(metrics.classification_report(y_valid, y_pred_knn))



# Generate a normalized confusion matrix

cm = confusion_matrix(y_valid, y_pred_knn)

cm_norm = cm / cm.sum(axis=1).reshape(-1,1)



# Display the confusion matrix as a heatmap

sns.heatmap(cm_norm, 

            cmap="YlGnBu", 

            xticklabels=knn.classes_, 

            yticklabels=knn.classes_, 

            vmin=0., 

            vmax=1., 

            annot=True, 

            annot_kws={'size':10})



# Adding headings and lables

plt.title('K - nearest neighbors classification')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# Generate a classification Report for the model

print(metrics.classification_report(y_valid, y_pred_lr))



cm = confusion_matrix(y_valid, y_pred_lr)

cm_norm = cm / cm.sum(axis=1).reshape(-1,1)



sns.heatmap(cm_norm, 

            cmap="YlGnBu", 

            xticklabels=lr.classes_, 

            yticklabels=lr.classes_, 

            vmin=0., 

            vmax=1., 

            annot=True, 

            annot_kws={'size':10})



# Adding headings and lables

plt.title('Logistic regression classification')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# Generate a classification Report for the linear SVC model

print(metrics.classification_report(y_valid, y_pred_lsvc))



# Generate a normalized confusion matrix

cm = confusion_matrix(y_valid, y_pred_lsvc)

cm_norm = cm / cm.sum(axis=1).reshape(-1,1)



# Display the confusion matrix as a heatmap

sns.heatmap(cm_norm, 

            cmap="YlGnBu", 

            xticklabels=lsvc.classes_, 

            yticklabels=lsvc.classes_, 

            vmin=0., 

            vmax=1., 

            annot=True, 

            annot_kws={'size':10})



# Adding headings and lables

plt.title('Linear SVC classification')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
# This code is intentionally commented out - Code takes >10 minutes to run. 



"""

# Set ranges for the parameters that we want to tune

params = {'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5],

          'tfidf__ngram_range' : [(1,1),(1,2)],

          'clf__max_iter': [1500, 2000, 2500, 3000],

          'tfidf__min_df': [2, 3, 4],

          'tfidf__max_df': [0.8, 0.9]}



# Perform randomized search & extract the optimal parameters

Randomized = RandomizedSearchCV(text_clf_lsvc, param_distributions=params, cv=5, scoring='accuracy', n_iter=5, random_state=42)

Randomized.fit(X_train,y_train)

Randomized.best_estimator_

"""



# Retrain linear SVC using optimal hyperparameters:

lsvc_op = Pipeline([('tfidf', TfidfVectorizer(max_df=0.8,

                                                    min_df=2,

                                                    ngram_range=(1,2))),

                  ('clf', LinearSVC(C=0.3,

                                    class_weight='balanced',

                                    max_iter=3000))])



# Fit and predict

lsvc_op.fit(X_train, y_train)

y_pred = lsvc_op.predict(X_valid)



print('F1 score improved by',

      round(100*((metrics.accuracy_score(y_pred, y_valid) - metrics.accuracy_score(y_pred_lsvc, y_valid)) /metrics.accuracy_score(y_pred_lsvc, y_valid)),0), 

      '%')

"""

# Saving each metric to add to a dictionary for logging

f1 = f1_score(y_valid, y_pred, average='weighted')

precision = precision_score(y_valid, y_pred, average='weighted')

recall = recall_score(y_valid, y_pred, average='weighted')



# Create dictionaries for the data we want to log          

metrics = {"f1": f1,

           "recall": recall,

           "precision": precision}



params= {'classifier': 'linear SVC',

         'max_df': 0.8,

         'min_df': 2,

         'ngram_range': '(1,2)',

         'vectorizer': 'Tfidf',

         'scaling': 'no',

         'resampling': 'no',

         'test_train random state': '0'}

  

# Log info on comet

experiment.log_metrics(metrics)

experiment.log_parameters(params)



# End experiment

experiment.end()



# Display results on comet page

experiment.display()



"""
test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')

y_test = lsvc_op.predict(test['message'])

output = pd.DataFrame({'tweetid': test.tweetid,

                       'sentiment': y_test})

output.to_csv('submission.csv', index=False)

output