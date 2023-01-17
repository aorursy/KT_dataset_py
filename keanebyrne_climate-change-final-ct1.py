# !pip install comet_ml
# # importing comet_ml

# from comet_ml import Experiment

    

# # Creating an experiment instance with the correct credentials

# experiment = Experiment(api_key="rBqQ3hDuEa6xVpT9ns5Tz1dVt",

#                         project_name="nlp-climate-change",

#                         workspace="monicafar147")
# general

import numpy as np 

import pandas as pd



# plotting

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-deep')



# text preprocessing

import re

from string import punctuation

import nltk

nltk.download(['stopwords','punkt'])

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from textblob import Word

from wordcloud import WordCloud, STOPWORDS



# models

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import resample



#model analysis

from sklearn.metrics import classification_report
# importing the DataSets

train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')
print("Train\n")

print(train.head(5))

print("\nTest")

print(test.head(5))

print("\n(Datasets were imported correctly)")
# Checking that there is no repeats in the Data

unique = [row for row in train['tweetid'].unique()]

print("Number of unique values")

print(train['tweetid'].nunique())

print("\nTotal number of values")

print(len(train['tweetid']))

print("\nNumber of null values:\n" + str(train.isnull().sum()))



print('\nWe can see the data does not contain any Null or repeated rows')
sum_df = train[['sentiment', 'message',]].groupby('sentiment').count()

sum_df
sum_df.sort_values('message', ascending=True).plot(kind='bar')

plt.title('Sentiment distribution')

plt.ylabel('count')

plt.xlabel('type of sentiment')

plt.show()
# Explore the word count and tweet lengths

train['length'] = train['message'].astype(str).apply(len)

train['word_count'] = train['message'].apply(lambda x: len(str(x).split()))

test['length'] = test['message'].astype(str).apply(len)

test['word_count'] = test['message'].apply(lambda x: len(str(x).split()))



# Creating the plot

plt.hist([test['length'], train['length']], bins=100, label=['test', 'train'])

plt.title('Tweet length distribution per tweet')

plt.xlabel('tweet length')

plt.ylabel('count')

plt.legend(loc='upper left')

plt.xlim([0,170])

plt.show()
plt.hist([test['word_count'], train['word_count']], bins=100, label=['test', 'train'])

plt.title('Word count distribution per tweet')

plt.xlabel('tweet word count')

plt.ylabel('count')

plt.legend(loc='upper left')

plt.xlim([0,35])

plt.show()
from nltk.tokenize import TweetTokenizer

tokeniser = TweetTokenizer()

train['tokens'] = train['message'].apply(tokeniser.tokenize)
def bag_of_words_count(words, word_dict={}):

    """ this function takes in a list of words and returns a dictionary 

        with each word as a key, and the value represents the number of 

        times that word appeared"""

    for word in words:

        if word in word_dict.keys():

            word_dict[word] += 1

        else:

            word_dict[word] = 1

    return word_dict
sentiment_labels = list(set(train['sentiment'].values))

sentiment = {}

for sent in sentiment_labels:

    df = train.groupby('sentiment')

    sentiment[sent] = {}

    for row in df.get_group(sent)['tokens']:

        sentiment[sent] = bag_of_words_count(row, sentiment[sent]) 
anti_tweets = {key: value for key, value in sorted(sentiment[-1].items(), key=lambda item: item[1], reverse=True)}

neutral_tweets = {key: value for key, value in sorted(sentiment[0].items(), key=lambda item: item[1], reverse=True)}

pro_tweets = {key: value for key, value in sorted(sentiment[1].items(), key=lambda item: item[1], reverse=True)}

news_tweets = {key: value for key, value in sorted(sentiment[2].items(), key=lambda item: item[1], reverse=True)}
random_characters = ['â','¢','‚','¬','Â','¦','’',"It's",'Ã','..','Å']
# anti

anti_keys = [key for key in anti_tweets.keys() if key not in set(random_characters+stopwords.words('english') + list(punctuation))]

anti_values = [value[1] for value in anti_tweets.items() if value[0] not in set(random_characters+stopwords.words('english') + list(punctuation))]



# neutral

neutral_keys = [key for key in neutral_tweets.keys() if key not in set(random_characters+stopwords.words('english') + list(punctuation))]

neutral_values = [value[1] for value in neutral_tweets.items() if value[0] not in set(random_characters+stopwords.words('english') + list(punctuation))]



# pro

pro_keys = [key for key in pro_tweets.keys() if key not in set(random_characters+stopwords.words('english') + list(punctuation))]

pro_values = [value[1] for value in pro_tweets.items() if value[0] not in set(random_characters+stopwords.words('english') + list(punctuation))]



# news

news_keys = [key for key in news_tweets.keys() if key not in set(random_characters+stopwords.words('english') + list(punctuation))]

news_values = [value[1] for value in news_tweets.items() if value[0] not in set(random_characters+stopwords.words('english') + list(punctuation))]
f, ax = plt.subplots(figsize=(6, 15))

sns.barplot(anti_values[0:20],anti_keys[0:20])

# Add a legend and informative axis label

ax.set(ylabel="",

       xlabel="counts")

ax.title.set_text('The most common words in anti tweets.')

sns.despine(left=True, bottom=True)
f, ax = plt.subplots(figsize=(6, 15))

sns.barplot(neutral_values[0:20],neutral_keys[0:20])

# Add a legend and informative axis label

ax.set(ylabel="",

       xlabel="counts")

ax.title.set_text('The most common words in neutral tweets.')

sns.despine(left=True, bottom=True)
f, ax = plt.subplots(figsize=(6, 15))

sns.barplot(pro_values[0:20],pro_keys[0:20])

# Add a legend and informative axis label

ax.set(ylabel="",

       xlabel="counts")

ax.title.set_text('The most common words in pro tweets.')

sns.despine(left=True, bottom=True)
f, ax = plt.subplots(figsize=(6, 15))

sns.barplot(news_values[0:20],news_keys[0:20])

# Add a legend and informative axis label

ax.set(ylabel="",

       xlabel="counts")

ax.title.set_text('The most common words in news tweets.')

sns.despine(left=True, bottom=True)
# Creating the word cloud

rnd_comments = train[train['sentiment']==1].sample(n=2000)['message'].values

wc = WordCloud(background_color='black', max_words=2000, stopwords=STOPWORDS)

wc.generate(''.join(rnd_comments))



# Plotting the word cloud

plt.figure(figsize=(15,10))

plt.axis('off')

plt.title('Frequent words of tweets from people who believe that climate change is man made', fontsize=20)

plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)

plt.show()
# Creating the word cloud

rnd_comment = train[train['sentiment']==-1].sample(n=200)['message'].values

wc = WordCloud(background_color='black', max_words=2000, stopwords=STOPWORDS)

wc.generate(''.join(rnd_comment))



#plotting the word cloud

plt.figure(figsize=(15,10))

plt.axis('off')

plt.title('Frequent words of tweets from people do not believe in man-made climate change', fontsize=20)

plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)

plt.show()
# Creating the word cloud

rnd_comment = train[train['sentiment']==0].sample(n=200)['message'].values

wc = WordCloud(background_color='black', max_words=2000, stopwords=STOPWORDS)

wc.generate(''.join(rnd_comment))



#plotting the word cloud

plt.figure(figsize=(15,10))

plt.axis('off')

plt.title('Frequent words of tweets that are neutral', fontsize=20)

plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)

plt.show()
# Creating the word cloud

rnd_comment = train[train['sentiment']==2].sample(n=200)['message'].values

wc = WordCloud(background_color='black', max_words=2000, stopwords=STOPWORDS)

wc.generate(''.join(rnd_comment))



#plotting the word cloud

plt.figure(figsize=(15,10))

plt.axis('off')

plt.title('Frequent words of tweets that are factual news', fontsize=20)

plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)

plt.show()
# Explore the word count and tweet lengths

train['length'] = train['message'].astype(str).apply(len)

train['word_count'] = train['message'].apply(lambda x: len(str(x).split()))

test['length'] = test['message'].astype(str).apply(len)

test['word_count'] = test['message'].apply(lambda x: len(str(x).split()))



# Creating the plot

plt.hist([test['length'], train['length']], bins=100, label=['test', 'train'])

plt.title('Tweet length distribution per tweet')

plt.xlabel('tweet length')

plt.ylabel('count')

plt.legend(loc='upper left')

plt.xlim([0,170])

plt.show()
plt.hist([test['word_count'], train['word_count']], bins=100, label=['test', 'train'])

plt.title('Word count distribution per tweet')

plt.xlabel('tweet word count')

plt.ylabel('count')

plt.legend(loc='upper left')

plt.xlim([0,35])

plt.show()
# Anti climate change hashtags

anti = train[train['sentiment']==-1]

a_tweets = anti['message']

series_a = a_tweets.str.extractall(r'(\#\w+)')[0].value_counts()



# Pro climate change hashtags

pro = train[train['sentiment']==1]

p_tweets = pro['message']

series_p = p_tweets.str.extractall(r'(\#\w+)')[0].value_counts()



# Neutral climate change hashtags

neutral = train[train['sentiment']==0]

neutral_tweets = neutral['message']

series_neutral = neutral_tweets.str.extractall(r'(\#\w+)')[0].value_counts()



# News climate change hashtags

news = train[train['sentiment']==2]

news_tweets = news['message']

series_news = news_tweets.str.extractall(r'(\#\w+)')[0].value_counts()



print("\033[1mAnti Climate change most common hashtags\033[0m\n" + str(series_a))

print("\n")

print("\033[1mPro Climate change most common hashtags\033[0m\n" + str(series_p))

print("\n")

print("\033[1mNeutral Climate change most common hashtags\033[0m\n" + str(series_neutral))

print("\n")

print("\033[1mNews Climate change most common hashtags\033[0m\n" + str(series_news))
#A nti climate change @'s

usernames_anti = a_tweets.str.extractall(r'(\@\w+)')[0].value_counts()



# Pro climate change @'s

usernames_pro = p_tweets.str.extractall(r'(\@\w+)')[0].value_counts()



# Neutral climate change @'s

usernames_neutral = neutral_tweets.str.extractall(r'(\@\w+)')[0].value_counts()



# News climate change @'s

usernames_news = news_tweets.str.extractall(r'(\@\w+)')[0].value_counts()



print("\033[1mAnti Climate change most common tweets\033[0m\n" + str(usernames_anti))

print("\n")

print("\033[1mPro Climate change most common tweets\033[0m\n" + str(usernames_pro))

print("\n")

print("\033[1mNeutral Climate change most common tweets\033[0m\n" + str(usernames_neutral))

print("\n")

print("\033[1mNews Climate change most common tweets\033[0m\n" + str(usernames_news))

rts = [0]

op =[]

for i in train['message']:

  if 'RT' in i:

    rts.append(i)

  else:

    op.append(i)



print("Number of Original Tweets: " + str(len(op)))

print("\nNumber of Retweets: " + str(len(rts)))

print("\nRatio of Orignal Tweets to retweets: " + str(round(len(op)/len(rts),2)))
def preprocess(tweet):

  tweet = tweet.lower()

  random_characters = ['â','¢','‚','¬','Â','¦','’',"It's",'Ã','..','Å']

  tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True)

  tweet = tokenizer.tokenize(tweet)

  stopwords_list = set(random_characters+list(punctuation))

  tweet = [word for word in tweet if word not in stopwords_list]

  tweet = re.sub(r'#([^\s]+)', r'\1', " ".join(tweet))

  tweet = re.sub(r'@([^\s]+)', r'\1', "".join(tweet))  

  return tweet
# Splitting the labels and features

train['processed'] = train['message'].apply(preprocess)

X = train['processed']

y = train['sentiment']
# printing out cleaned text

index = 1

for tweet in X[0:10]:

    print(str(index)+": " + tweet)

    print('\n')

    index += 1
# preprocess testing data by applying our function

test['processed'] = test['message'].apply(preprocess)
# explore the word count and tweet lengths

train['length after'] = train['processed'].astype(str).apply(len)

train['word_count after'] = train['processed'].apply(lambda x: len(str(x).split()))

test['length after'] = test['processed'].astype(str).apply(len)

test['word_count after'] = test['processed'].apply(lambda x: len(str(x).split()))
plt.hist([train['length'], train['length after']], bins=100, label=['before', 'after'])

plt.title('Tweet length distribution per tweet in training set')

plt.xlabel('tweet length')

plt.ylabel('count')

plt.legend(loc='upper left')

plt.xlim([0,170])

plt.show()
plt.hist([test['length'], test['length after']], bins=100, label=['before', 'after'])

plt.title('Tweet length distribution per tweet in test set')

plt.xlabel('tweet length')

plt.ylabel('count')

plt.legend(loc='upper left')

plt.xlim([0,170])

plt.show()
# Splitting the labels and fetures into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,random_state=42,stratify=y)
# Creating the unseen set, so that we can post to Kaggle and recieve a score based on the performance

x_unseen = test['processed']
#creating a pipeline with a tfidf vectorizer and a logistic regression model

LR_model = Pipeline([('tfidf',TfidfVectorizer()),('classify',(LogisticRegression(C=1.0,solver='lbfgs',random_state=42,max_iter=200)))])



#fitting the model

LR_model.fit(X_train, y_train)



#Apply model on test data

y_pred_lr = LR_model.predict(X_test)
#creating a pipeline with the tfid vectorizer and a linear svc model

svc = Pipeline([('tfidf',TfidfVectorizer()),('classify',LinearSVC(C=1))])



#fitting the model

svc.fit(X_train, y_train)



#apply model on test data

y_pred_svc = svc.predict(X_test)
#creating a pipeline with the DecisionTreeClassifier 

DT = Pipeline([('tfidf',TfidfVectorizer()),('classify',(DecisionTreeClassifier(max_depth=150,random_state=42, splitter='best')))])



#fitting the model

DT.fit(X_train, y_train)



#Apply model on test data

y_pred_DT = DT.predict(X_test)
#creating a pipeline with the RandomForest classifier  

RF_model = Pipeline([('tfidf', TfidfVectorizer()),('clf', (RandomForestClassifier(max_depth=200, random_state=42,n_estimators=10)))])



#fitting the model

RF_model.fit(X_train, y_train)



#Apply model on test data

y_pred_RF = RF_model.predict(X_test)
#creating a pipeline with the GradientBoosting classifier

gb_clf = Pipeline([('tfidf',TfidfVectorizer()),('classify',( GradientBoostingClassifier(n_estimators=20,  max_depth=150, random_state=42)))])



#fitting the model

gb_clf.fit(X_train, y_train)



#Apply model on test data

y_pred_gb = gb_clf.predict(X_test)
from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import make_pipeline

naive_bayes = pipeline = make_pipeline(

     CountVectorizer(), 

     FunctionTransformer(lambda x: x.todense(), accept_sparse=True), 

     GaussianNB())



naive_bayes.fit(X_train, y_train)



#Apply model on test data

y_pred_guassianNB = naive_bayes.predict(X_test)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
# Define component objects of our pipeline then create it!

objs = [("tfidf", TfidfVectorizer()),

        ("svm", SVC(kernel="linear"))]

pipe = Pipeline(objs)

# Specify parameters of the pipeline and their ranges for grid search

params = {'svm__C': [0.1,1,10],

          'svm__gamma': [0.01,0.1,1]}



# Construct our grid search object

search = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1, verbose=1)

search.fit(X_train,y_train)

y_pred_grid = search.best_estimator_.predict(X_test)
print(search.best_params_)
#creating variables for the various sentiments

pro = train[train['sentiment']==1]

news = train[train['sentiment']==2]

neutral = train[train['sentiment']==0]

anti = train[train['sentiment']==-1]
# upsampling the minority

upsampled_anti = resample(anti,

                    replace=True, 

                    n_samples=len(pro), 

                    random_state=42)



upsampled_news = resample(news,

                    replace=True, 

                    n_samples=len(pro),

                    random_state=42) 



upsampled_neutral = resample(neutral,

                    replace=True, 

                    n_samples=len(pro), 

                    random_state=42) 



# Combine upsampled minority class with majority class

upsampled = pd.concat([upsampled_anti,upsampled_news, upsampled_neutral, pro])
print('The distribution of samples before upsampling:')

print(train['sentiment'].value_counts())
# Check new class counts

print('The distribution of samples after upsampling:')

print(upsampled['sentiment'].value_counts())
# Splitting the labels and features into training and testing sets

upsampledX_train, upsampledX_test, upsampledy_train, upsampledy_test = train_test_split(upsampled['processed']

                                                                                        , upsampled['sentiment']

                                                                                        , test_size=0.1,random_state=42)
# Downsampling the majority

downsampled_anti = resample(pro,

                    replace=True, 

                    n_samples=len(anti), 

                    random_state=42) 



downsampled_news = resample(news,

                    replace=True, 

                    n_samples=len(anti), 

                    random_state=42) 



downsampled_neutral = resample(neutral,

                    replace=True, 

                    n_samples=len(anti),

                    random_state=42) 



# Combine downsampled majority class with minority class

downsampled = pd.concat([downsampled_anti,downsampled_news, downsampled_neutral, anti])
print('The distribution of samples before upsampling:')

print(train['sentiment'].value_counts())
# Check new class counts

print('The distribution of samples after upsampling:')

print(downsampled['sentiment'].value_counts())
# Splitting the labels and features into training and testing sets

downsampledX_train, downsampledX_test, downsampledy_train, downsampledy_test = train_test_split(downsampled['processed']

                                                                                        , downsampled['sentiment']

                                                                                        , test_size=0.1,random_state=42)
# Creating a pipeline for the new upsampled data with a LogisticRegression model

upsampled_LR = Pipeline([('tfidf', TfidfVectorizer()),

                     ('lr', LogisticRegression(solver='lbfgs', max_iter=1200000))

                      ])



# Feed the training data through the pipeline

upsampled_LR.fit(upsampledX_train, upsampledy_train)
#apply model on test data

y_pred_upsampled = upsampled_LR.predict(upsampledX_test)
# Creating another pipeline for the downsampled data with a LogisticRegression model

downsampled_LR = Pipeline([('tfidf', TfidfVectorizer()),

                     ('lr', LogisticRegression(solver='lbfgs', max_iter=1200000))

                      ])



# Feed the training data through the pipeline

downsampled_LR.fit(downsampledX_train, downsampledy_train)
#apply model on test data

y_pred_downsampled = downsampled_LR.predict(downsampledX_test)
# Creating another pipeline for the upsampled data with a LinearSVC model

upsampled_svc = Pipeline([('tfidf',TfidfVectorizer()),('classify',LinearSVC())])



# Feed the training data through the pipeline

upsampled_svc.fit(upsampledX_train, upsampledy_train)
#apply model on test data

y_pred_svc_upsampled = upsampled_svc.predict(upsampledX_test)
# Creating another pipeline for the downsampled data with a LinearSVC model

downsampled_svc = Pipeline([('tfidf',TfidfVectorizer()),('classify',LinearSVC())])



# Feed the training data through the pipeline

downsampled_svc.fit(downsampledX_train, downsampledy_train)
#apply model on test data

y_pred_svc_downsampled = downsampled_svc.predict(downsampledX_test)
print(classification_report(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_DT))
print(classification_report(y_test, y_pred_RF))
print(classification_report(y_test, y_pred_gb)) 
print(classification_report(y_test, y_pred_guassianNB)) 
print(classification_report(upsampledy_test, y_pred_upsampled))
print(classification_report(downsampledy_test, y_pred_downsampled))
print(classification_report(upsampledy_test, y_pred_svc_upsampled))
print(classification_report(downsampledy_test, y_pred_svc_downsampled))
print(classification_report(y_test, y_pred_grid))
# apply models on unseen data:



kaggle_lr = LR_model.predict(x_unseen) # Logistic Regression model

kaggle_svc = svc.predict(x_unseen) # SVC model

kaggle_DT = DT.predict(x_unseen) # Decision Tree model

kaggle_RF = RF_model.predict(x_unseen) # Random Forest model

kaggle_gb = gb_clf.predict(x_unseen) # Gradient Boosting model

kaggle_LR_upsampled = upsampled_LR.predict(x_unseen) # Logistic Regression model (upsampled)

kaggle_LR_downsampled = downsampled_LR.predict(x_unseen) # Logistic Regression model (downsampled)

kaggle_svc_upsampled = upsampled_svc.predict(x_unseen) # SVC model (upsampled)

kaggle_svc_downsampled = downsampled_svc.predict(x_unseen) # SVC model (downsampled)

kaggle_guassianNB =  naive_bayes.predict(x_unseen) # Naiave bayes model

kaggle_grid_search = search.best_estimator_.predict(x_unseen)
# create table to submit as .csv file

Table = {'tweetid': test['tweetid'], 'sentiment':kaggle_svc} #choose a model e.g kaggle_svc

submission = pd.DataFrame(data=Table)

submission.set_index('tweetid')

submission.head()
# Only run this code if wanting to save to a CSV file

# save to .csv file

submission.to_csv("SVC.csv",index  = False)
# # Create dictionaries for the data we want to log



# params = {"preprocessing":  "_preprocess(df)",

#           "keeps username":"True",

#           "keeps hashtags":"True",

#           "keeps URL":"urlweb",

#           "removes puncutation":"string punctuation",

#           "use stopwords":"False",

#           "model_type": "LinearSVC",

#           }

# Log our parameters and results

# experiment.log_parameters(params)
# experiment.end()
# experiment.display()