!pip install comet_ml
from comet_ml import Experiment
experiment = Experiment(api_key = 'VWmaCXXdpmeXoNUPZya4tPsmi',

                       project_name = 'Classification Predict',

                       workspace = 'menzi-mchunu')
!pip install emoji 
!pip install demoji
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import nltk

import emoji as emoj

import demoji



from nltk.stem.porter import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, recall_score, precision_score, classification_report,accuracy_score, log_loss, make_scorer, f1_score

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn.svm import LinearSVC

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud

from emoji import UNICODE_EMOJI

from gensim import models

from gensim.models import word2vec

from sklearn.manifold import TSNE

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.utils import resample

from sklearn import metrics



demoji.download_codes()

sns.set_style('white')

%matplotlib inline
df_train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

df_test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')
df_train.head()
df_test.head()
print(df_train.shape)

print(df_test.shape)
df_train.info()
df_test.info()
df_train[['message']].describe()
df_test[['message']].describe()
def text_has_emoji(text):

    """This function checks all the rows of data to see if we have any emojis in the tweet """

    for character in text:

        if character in emoj.UNICODE_EMOJI:

            return True

    return False

"""We apply the above function to check for emojis"""

df_train['emoji'] = df_train['message'].apply(text_has_emoji)

df_test['emoji'] = df_test['message'].apply(text_has_emoji)
df_train[df_train['emoji'] == True]
df_test[df_test['emoji'] == True]
def extra_all_the_emoji(strings):

    """This function extracts the emojis for every row in the dataframe"""

    return ''.join(character for character in strings if character in emoj.UNICODE_EMOJI)
#We apply the extra_all_the_emoji function to the message column

"""We see all the emojis in the train dataframe """

df_train['emojis'] = df_train['message'].apply(extra_all_the_emoji)

df_train[df_train['emojis'] != '']['emojis']
"""We see all the emojis in the train dataframe """

df_test['emojis'] = df_test['message'].apply(extra_all_the_emoji)

df_test[df_test['emojis'] != '']['emojis']
#The list of all emojis in the train data

list_of_emojis_train = df_train[df_train['emojis'] != '']['emojis'].tolist()
#The list of all emojis in the test data

list_of_emojis_test = df_test[df_test['emojis'] != '']['emojis'].tolist()
#We loop through the whole list to change every emoji to text in the train data

changed_emoji_to_text_train = []

for emojis in list_of_emojis_train:

    changed_emoji_to_text_train.append(emoj.demojize(emojis, delimiters=("", "")))

changed_emoji_to_text_train   
#We loop through the whole list to change every emoji to text in the train data

changed_emoji_to_text_test = []

for emojis in list_of_emojis_test:

    changed_emoji_to_text_test.append(emoj.demojize(emojis, delimiters=("", "")))

changed_emoji_to_text_test
def emojis_to_text(text):

    """This function changes all the emojis in the message column into words"""

    return emoj.demojize(text, delimiters=("", ""))

df_train['message'] = df_train['message'].apply(emojis_to_text)

df_test['message'] = df_test['message'].apply(emojis_to_text)
def remove_twitter_handles(tweet, pattern):

    """This function removes all the twitter handles on the dataframe"""

    r = re.findall(pattern, tweet)

    for text in r:

        tweet = re.sub(text, '', tweet)

    return tweet



df_train['clean_tweet'] = np.vectorize(remove_twitter_handles)(df_train['message'], "@[\w]*") 

df_test['clean_tweet'] = np.vectorize(remove_twitter_handles)(df_test['message'], "@[\w]*") 
df_train.head()
df_test.head()
# Lower Casing clean_tweet

df_train['clean_tweet']  = df_train['clean_tweet'].str.lower()

df_train.head()
# Lower Casing clean_tweet

df_test['clean_tweet']  = df_test['clean_tweet'].str.lower()

df_test.head()
#Links for train data

tweets = []

new = list(df_train['clean_tweet'])

for tweet in new:

    tweet = re.sub(r"http\S+", '', tweet, flags=re.MULTILINE)

    tweets.append(tweet)

    

df_train['clean_tweet'] = tweets



#Links for test data

test_tweets = []

new1 = list(df_test['clean_tweet'])

for tweet in new1:

    tweet = re.sub(r"http\S+", '', tweet, flags=re.MULTILINE)

    test_tweets.append(tweet)

    

df_test['clean_tweet'] = test_tweets





#Retweets for train data

tweets = []

new = list(df_train['clean_tweet'])

for tweet in new:

    tweet = re.sub("rt :", '', tweet, flags=re.MULTILINE)

    tweets.append(tweet)

    

df_train['clean_tweet'] = tweets





#Retweets test data

test_tweets = []

new1 = list(df_test['clean_tweet'])

for tweet in new1:

    tweet = re.sub("rt :", '', tweet, flags=re.MULTILINE)

    test_tweets.append(tweet)

    

df_test['clean_tweet'] = test_tweets
stop_words = nltk.corpus.stopwords.words('english')
df_train['tidy_tweet'] = df_train['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))

df_test['tidy_tweet'] = df_test['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
df_train.head()
df_test.head()
#Tokenization

def tokenizing(text):

    """This Function breaks up text into tokens"""

    text = re.split('\W+', text)

    return text



df_train['tokenized_tweet'] = df_train['tidy_tweet'].apply(lambda x: tokenizing(x))

df_test['tokenized_tweet'] = df_test['tidy_tweet'].apply(lambda x: tokenizing(x))
df_train.head()
df_test.head()
#Lemmatization

tokens = df_train['tokenized_tweet']

tokens_test = df_test['tokenized_tweet']



lemmatizer = WordNetLemmatizer()



tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

tokens_test = tokens_test.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])



df_train['lemmatized_tweet'] = tokens

df_test['lemmatized_tweet'] = tokens_test
df_train.head()
df_test.head()
df_train = df_train.drop(['tidy_tweet'],axis=1)

df_train = df_train.drop(['tokenized_tweet'], axis=1)

df_train = df_train.drop(['emoji'],axis=1)

df_train = df_train.drop(['emojis'], axis=1)



df_test = df_test.drop(['tidy_tweet'],axis=1)

df_test = df_test.drop(['tokenized_tweet'], axis=1)

df_test = df_test.drop(['emoji'],axis=1)

df_test = df_test.drop(['emojis'], axis=1)
df_train.head()
df_test.head()
plt.figure(figsize=(10,5))

sns.countplot(x='sentiment',data=df_train, palette='CMRmap')

plt.title('Number of Tweets per Class', fontsize=20)

plt.xlabel('Number of Tweets', fontsize=14)

plt.ylabel('Class', fontsize=14)

plt.show()
#ADD Y LABEL 

df_train['text length'] = df_train['message'].apply(len)

g = sns.FacetGrid(df_train,col='sentiment')

g.map(plt.hist,'text length')

plt.show()
fig,axis = plt.subplots(figsize=(12,5))

sns.boxplot(x='sentiment',y='text length',data=df_train, palette='rainbow')

plt.title('Distribution of Text Length for Different Sentiments', fontsize = 12)

plt.xlabel('Sentiment', fontsize = 12)

plt.ylabel('Text Length', fontsize = 12)

plt.show()
rate = df_train.groupby('sentiment').mean()

rate
rate.corr()
#ADD TITLE

plt.figure(figsize=(7,5))

sns.heatmap(rate.corr(),cmap='coolwarm',annot=True)

plt.title('Correlation between Tweetid and Text Length', fontsize = 15 )

plt.show()
df_pro = df_train[df_train.sentiment==1]

words = ' '.join([text for text in df_train['clean_tweet']])

wordcloud = WordCloud(width = 1000, height = 500).generate(words)

plt.figure(figsize=(15,5))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Support Belief of Man-Made Climate Change', fontsize = 20)

plt.show()
df_anti = df_train[df_train.sentiment==-1]

text= (' '.join(df_anti['clean_tweet']))

wordcloud = WordCloud(width = 1000, height = 500).generate(text)

plt.figure(figsize=(15,5))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Against Belief of Man-Made Climate Change', fontsize = 20)

plt.show()
df_neutral = df_train[df_train.sentiment==0]

text= (' '.join(df_neutral['clean_tweet']))

wordcloud = WordCloud(width = 1000, height = 500).generate(text)

plt.figure(figsize=(15,5))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Neutral About Climate Change', fontsize = 30)

plt.show()
df_factual = df_train[df_train.sentiment==2]

text= (' '.join(df_factual['clean_tweet']))

wordcloud = WordCloud(width = 1000, height = 500).generate(text)

plt.figure(figsize=(15,5))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('News About Climate Change', fontsize = 30)

plt.show()
pro_hashtags = []

for message in df_pro['message']:

    hashtag = re.findall(r"#(\w+)", message)

    pro_hashtags.append(hashtag)



pro_hashtags = sum(pro_hashtags,[])

a = nltk.FreqDist(pro_hashtags)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})



# selecting top 10 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(10,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

plt.setp(ax.get_xticklabels(),rotation='vertical', fontsize=10)

plt.title('Top 10 Hashtags in "Pro" Tweets', fontsize=14)

plt.show()
anti_hashtags = []

for message in df_anti['message']:

    hashtag = re.findall(r"#(\w+)", message)

    anti_hashtags.append(hashtag)



anti_hashtags = sum(anti_hashtags,[])





a = nltk.FreqDist(anti_hashtags)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})



# selecting top 20 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(10,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

plt.setp(ax.get_xticklabels(),rotation='vertical', fontsize=10)

plt.title('Top 10 Hashtags in "Anti" Tweets', fontsize=14)

plt.show()
neutral_hashtags = []

for message in df_neutral['message']:

    hashtag = re.findall(r"#(\w+)", message)

    neutral_hashtags.append(hashtag)



neutral_hashtags = sum(neutral_hashtags,[])





a = nltk.FreqDist(neutral_hashtags)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})



# selecting top 20 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(10,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

plt.setp(ax.get_xticklabels(),rotation='vertical', fontsize=10)

plt.title('Top 10 Hashtags in Neutral Tweets', fontsize=14)

plt.show()
factual_hashtags = []

for message in df_factual['message']:

    hashtag = re.findall(r"#(\w+)", message)

    factual_hashtags.append(hashtag)



factual_hashtags = sum(factual_hashtags,[])





a = nltk.FreqDist(factual_hashtags)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})



# selecting top 20 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(10,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

plt.setp(ax.get_xticklabels(),rotation='vertical', fontsize=10)

plt.title('Top 10 Hashtags in Factual Tweets', fontsize=14)

plt.show()
data = df_train.copy()
STOP_WORDS = nltk.corpus.stopwords.words()



def clean_sentence(val):

    "remove chars that are not letters or numbers, downcase, then remove stop words"

    regex = re.compile('([^\s\w]|_)+')

    sentence = regex.sub('', val).lower()

    sentence = sentence.split(" ")

    

    for word in list(sentence):

        if word in STOP_WORDS:

            sentence.remove(word)  

            

    sentence = " ".join(sentence)

    return sentence



def clean_dataframe(data):

    "drop nans, then apply 'clean_sentence' function to question1 and 2"

    data = data.dropna(how="any")

    

    for col in ['message']:

        data[col] = data[col].apply(clean_sentence)

    

    return data

data = clean_dataframe(data)

data.head(5)
def build_corpus(data):

    "Creates a list of lists containing words from each sentence"

    corpus = []

    for col in ['message']:

        for sentence in data[col].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

            

    return corpus



corpus = build_corpus(data)        

corpus[0:2]
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)

model.wv['warming']
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(13, 7)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
tsne_plot(model)

plt.show()
# A more selective model

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)

tsne_plot(model)
df_train = df_train.drop(['text length'],axis=1)
combi = df_train.append(df_test, ignore_index=True)



tfidf_vectorizer = TfidfVectorizer()

tfidf = tfidf_vectorizer.fit_transform(combi['message'])



train = tfidf[:15819,:]

test = tfidf[15819:,:]



X_tfidf = train

y_tfidf = df_train['sentiment']



X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_tfidf,  random_state=42, test_size=0.1)
df = df_train.copy()



# Separate majority and minority classes

df_majority = df[df.sentiment==1]

df_minority = df[(df.sentiment==-1) | (df.sentiment==0) | (df.sentiment==2)]

 

# Downsample majority class

df_majority_downsampled = resample(df_majority, 

                                 replace=True,    # sample without replacement

                                 n_samples=7000,     # to match minority class

                                 random_state=42) # reproducible results

 

# Combine minority class with downsampled majority class

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

 

# Display new class counts

df_downsampled.sentiment.value_counts()
# Separate majority and minority classes

df_majority = df_downsampled[(df_downsampled.sentiment==1) | (df_downsampled.sentiment==0) | (df_downsampled.sentiment==2)]

df_minority = df_downsampled[df_downsampled.sentiment==-1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=4000,    # to match majority class

                                 random_state=42) # reproducible results

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

df_upsampled.sentiment.value_counts()
# Separate majority and minority classes

df_majority = df_upsampled[(df_upsampled.sentiment==1) | (df_upsampled.sentiment==-1) | (df_upsampled.sentiment==2)]

df_minority = df_upsampled[df_upsampled.sentiment==0]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=4000,    # to match majority class

                                 random_state=42) # reproducible results

 

# Combine majority class with upsampled minority class

up_sampled = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

up_sampled.sentiment.value_counts()
# Separate majority and minority classes

df_majority = up_sampled[(up_sampled.sentiment==1) | (up_sampled.sentiment==-1) | (up_sampled.sentiment==0)]

df_minority = up_sampled[up_sampled.sentiment==2]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=4000,    # to match majority class

                                 random_state=42) # reproducible results

 

# Combine majority class with upsampled minority class

resampled_data = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

resampled_data.sentiment.value_counts()
#Create training test and testing set for the resampled data

resampled_data = resampled_data.append(df_test, ignore_index=True)

tfidf_vectorizer1 = TfidfVectorizer()

tfidf_resam = tfidf_vectorizer1.fit_transform(resampled_data['message'])

train_resam = tfidf_resam[:19000,:]

test_resam = tfidf_resam[19000:,:]



X_resampled = train_resam

y_resampled = resampled_data.iloc[:19000,:]['sentiment']

X_train_resam, X_test_resam, y_train_resam, y_test_resam = train_test_split(X_resampled, y_resampled, test_size=0.1)
modelstart= time.time()



classifier1 = LinearSVC()

classifier2 = LinearSVC()



#Fitting to Unbalanced data

lvc1 = classifier1.fit(X_train,y_train)

ypred_lvc1 = lvc1.predict(X_test)

lvc_score1 = f1_score(y_test, ypred_lvc1, average='macro')

precision_lvc1 = precision_score(y_test, ypred_lvc1, average='macro')

recall_lvc1 = recall_score(y_test, ypred_lvc1, average='macro')



#Fitting to Balanced data

lvc2 = classifier2.fit(X_train_resam,y_train_resam)

ypred2 = lvc2.predict(X_test_resam)

lvc_score2 = f1_score(y_test_resam, ypred2, average='macro')

precision_lvc2 = precision_score(y_test_resam, ypred2,average='macro')

recall_lvc2 = recall_score(y_test_resam, ypred2, average='macro')



print("Testing: Linear Support Vector")

print('F1 Score on unbalanced data', lvc_score1)

print('F1 Score on balanced data', lvc_score2)

print('Recall Score on unbalanced data', recall_lvc1)

print('Recall Score on balanced data', recall_lvc2)

print('PrecisionScore on unbalanced data', precision_lvc1)

print('Precision Score on balanced data', precision_lvc2)

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Unbalanced 

# create the grid

max_iter = [100,500,1000,2000,5000]

C = [1,10,50,100,200]

param_grid = dict(C =C ,max_iter=max_iter)





# search the grid

grid = GridSearchCV(estimator=lvc1, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=0,

                    n_jobs=-1)





lvc_best = grid.fit(X_train, y_train)

lvc_estimator = grid.best_estimator_

lvc_parameters = grid.best_params_

lvc_score = grid.best_score_



print('Linear Support Vector on Unbalanced Data')

print('Best Estimator: ', lvc_estimator)

print('Best Parameter: ', lvc_parameters)

print('Best Score: ', lvc_score)
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Balanced 

# create the grid

max_iter = [100,500,1000,2000,5000]

C = [1,10,50,100,200]

param_grid = dict(C =C ,max_iter=max_iter)





# search the grid

grid = GridSearchCV(estimator=lvc2, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=0,

                    n_jobs=-1)





lvc_best = grid.fit(X_train_resam, y_train_resam)

lvc_estimator = grid.best_estimator_

lvc_parameters = grid.best_params_

lvc_score = grid.best_score_



print('Linear Support Vector on Balanced Data')

print('Best Estimator: ', lvc_estimator)

print('Best Parameter: ', lvc_parameters)

print('Best Score: ', lvc_score)
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()



df_train['scores'] = df_train['message'].apply(lambda message: sid.polarity_scores(message))

df_train['compound']  = df_train['scores'].apply(lambda score_dict: score_dict['compound'])
modelstart= time.time()

lr_classifier1 = LogisticRegression(solver='lbfgs')

lr_classifier2 = LogisticRegression(solver='lbfgs')



#Fitting on Unbalanced data

lr_model1 = lr_classifier1.fit(X_train, y_train)

predictions_lr1 = lr_classifier1.predict(X_test)

lr_score1 = f1_score(y_test, predictions_lr1, average='macro')

precision_lr1 = precision_score(y_test, predictions_lr1, average='macro')

recall_lr1 = recall_score(y_test, predictions_lr1, average='macro')



#Fitting on Balanced data

lr_model2 = lr_classifier2.fit(X_train_resam,y_train_resam)

predictions_lr2 = lr_classifier2.predict(X_test_resam)

lr_score2 = f1_score(y_test_resam, predictions_lr2, average='macro')

precision_lr2 = precision_score(y_test_resam, predictions_lr2, average='macro')

recall_lr2 = recall_score(y_test_resam, predictions_lr2, average='macro')





print("Testing: Logistic Regression")

print('F1 Score on unbalanced data: ',lr_score1)

print('F1 Score on balanced data: ', lr_score2)

print('Recall Score on unbalanced data: ',recall_lr1)

print('Recall Score on balanced data: ', recall_lr2)

print('Precision Score on unbalanced data: ',precision_lr1)

print('Precision Score on balanced data: ', precision_lr2)

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Unbalanced 

# create the grid

C = np.logspace(-3,3,7)

penalty = ["l1","l2"]

param_grid = dict(C =C ,penalty = penalty)





# search the grid

grid = GridSearchCV(estimator=lr_model1, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=0,

                    n_jobs=-1)





lr_best = grid.fit(X_train, y_train)

lr_estimator = grid.best_estimator_

lr_parameters = grid.best_params_

lr_score = grid.best_score_



print('Logistic Regression on Unbalanced Data')

print('Best Estimator: ', lr_estimator)

print('Best Parameter: ', lr_parameters)

print('Best Score: ', lr_score)
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Balanced 

# create the grid

C = np.logspace(-3,3,7)

penalty = ["l1","l2"]

param_grid = dict(C =C ,penalty = penalty)





# search the grid

grid = GridSearchCV(estimator=lr_model2, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=0,

                    n_jobs=-1)





lr_best = grid.fit(X_train_resam, y_train_resam)

lr_estimator = grid.best_estimator_

lr_parameters = grid.best_params_

lr_score = grid.best_score_



print('Logistic Regression on Balanced Data')

print('Best Estimator: ', lr_estimator)

print('Best Parameter: ', lr_parameters)

print('Best Score: ', lr_score)
modelstart= time.time()



KNN_classifier1 = KNeighborsClassifier()

KNN_classifier2 = KNeighborsClassifier()



knn_model1 = KNN_classifier1.fit(X_train, y_train)

predictions_knn1 = KNN_classifier1.predict(X_test)

knn_score1 = f1_score(y_test, predictions_knn1, average='macro')

recall_knn1 = recall_score(y_test, predictions_knn1, average='macro')

precision_knn1 = precision_score(y_test, predictions_knn1, average='macro')



knn_model2 = KNN_classifier2.fit(X_train_resam,y_train_resam)

predictions_knn2 = KNN_classifier2.predict(X_test_resam)

knn_score2 = f1_score(y_test_resam, predictions_knn2, average='macro')

recall_knn2 = recall_score(y_test_resam, predictions_knn2, average='macro')

precision_knn2 = precision_score(y_test_resam, predictions_knn2, average='macro')





print("Testing: K-Nearest Neighbors")

print('F1 Score on unbalanced data: ', knn_score1)

print('F1 Score on balanced data: ', knn_score2)

print('Recall Score on unbalanced data: ', recall_knn1)

print('Recall Score on balanced data: ', recall_knn2)

print('Precision Score on unbalanced data: ', precision_knn1)

print('Precision Score on balanced data: ', precision_knn2)

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Unbalanced 

# create the grid

n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

param_grid = dict(n_neighbors = n_neighbors)



# search the grid

grid = GridSearchCV(estimator=knn_model1, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=0,

                    n_jobs=-1)



knn_best = grid.fit(X_train, y_train)

knn_estimator = grid.best_estimator_

knn_parameters = grid.best_params_

knn_score = grid.best_score_



print('K-Nearest Neighbours on Unbalanced Data')

print('Best Estimator: ', lr_estimator)

print('Best Parameter: ', lr_parameters)

print('Best Score: ', lr_score)
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Unbalanced 

# create the grid

n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

param_grid = dict(n_neighbors = n_neighbors)



# search the grid

grid = GridSearchCV(estimator=knn_model2, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 5,

                    verbose=0,

                    n_jobs=-1)



knn_best = grid.fit(X_train_resam, y_train_resam)

knn_estimator = grid.best_estimator_

knn_parameters = grid.best_params_

knn_score = grid.best_score_



print('K-Nearest Neighbours on Balanced Data')

print('Best Estimator: ', knn_estimator)

print('Best Parameter: ', knn_parameters)

print('Best Score: ', knn_score)
modelstart= time.time()

rf1 = RandomForestClassifier(n_estimators=200, random_state=11)

rf2 = RandomForestClassifier(n_estimators=200, random_state=11)

 

#Training on Unbalanced data    

rf_model1 = rf1.fit(X_train, y_train)

predictions_rf1 = rf1.predict(X_test)

rf_score1 = f1_score(y_test, predictions_rf1, average='macro')

recall_rf1 = recall_score(y_test, predictions_rf1, average='macro')

precision_rf1 = precision_score(y_test, predictions_rf1, average='macro')



#Training on Balanced data

rf_model2 = rf2.fit(X_train_resam,y_train_resam)

predictions_rf2 = rf2.predict(X_test_resam)

rf_score2 = f1_score(y_test_resam, predictions_rf2, average='macro')

recall_rf2 = recall_score(y_test_resam, predictions_rf2, average='macro')

precision_rf2 = precision_score(y_test_resam, predictions_rf2, average='macro')



print("Testing: Random Forest")

print('F1 Score on unbalanced data: ', rf_score1)

print('F1 Score on balanced data: ', rf_score2)

print('Recall Score on unbalanced data: ', recall_rf1)

print('Recall Score on balanced data: ', recall_rf2)

print('Precision Score on unbalanced data: ', precision_rf1)

print('Precision Score on balanced data: ', precision_rf2)

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Unbalanced 

# create the grid

n_estimators = [100, 1000, 2000]

max_features = [1, 3, 5]

max_depth = [5, 10, 20]

param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)





# search the grid

grid = GridSearchCV(estimator=rf_model1, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=2,

                    n_jobs=-1)





rf_best = grid.fit(X_train_resam, y_train_resam)

rf_estimator = grid.best_estimator_

rf_parameters = grid.best_params_

rf_score = grid.best_score_



print('Random Forest on Unbalanced Data')

print('Best Estimator: ', rf_estimator)

print('Best Parameter: ', rf_parameters)

print('Best Score: ', rf_score)
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Balanced 

# create the grid

n_estimators = [100, 1000, 2000]

max_features = [1, 3, 5]

max_depth = [5, 10, 20]

param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)





# search the grid

grid = GridSearchCV(estimator=rf_model2, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=2,

                    n_jobs=-1)





rf_best = grid.fit(X_train, y_train)

rf_estimator = grid.best_estimator_

rf_parameters = grid.best_params_

rf_score = grid.best_score_



print('Random Forest on Balanced Data')

print('Best Estimator: ', rf_estimator)

print('Best Parameter: ', rf_parameters)

print('Best Score: ', rf_score)
modelstart= time.time()

dt_classifier1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)



dt_model1 = dt_classifier1.fit(X_train, y_train)

predictions_dt1 = dt_classifier1.predict(X_test)

dt_score1 = f1_score(y_test, predictions_dt1, average='macro')

recall_dt1 = recall_score(y_test, predictions_dt1, average='macro')

precision_dt1 = precision_score(y_test, predictions_dt1, average='macro')



dt_model2 = dt_classifier2.fit(X_train_resam,y_train_resam)

predictions_dt2 = dt_classifier2.predict(X_test_resam)

dt_score2 = f1_score(y_test_resam, predictions_dt2, average='macro')

recall_dt2 = recall_score(y_test_resam, predictions_dt2, average='macro')

precision_dt2 = precision_score(y_test_resam, predictions_dt2, average='macro')



print("Testing: Decision Tree")

print('F1 Score on unbalanced data: ', dt_score1)

print('F1 Score on balanced data: ', dt_score2)

print('Recall Score on unbalanced data: ', recall_dt1)

print('Recall Score on balanced data: ', recall_dt2)

print('Precision Score on unbalanced data: ', precision_dt1)

print('Precision Score on balanced data: ', precision_dt2)

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Unbalanced 

# create the grid

max_depth = [5, 10, 20,100,200]

param_grid = dict(max_depth=max_depth)



# search the grid

grid = GridSearchCV(estimator=dt_model1, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv=2,

                    verbose=2,

                    n_jobs=-1)



dt_best = grid.fit(X_train, y_train)

dt_estimator = grid.best_estimator_

dt_parameters = grid.best_params_

dt_score = grid.best_score_



print('Decision Tree on Unbalanced Data')

print('Best Estimator: ', dt_estimator)

print('Best Parameter: ', dt_parameters)

print('Best Score: ', dt_score)
#Create Scorer

f1 = make_scorer(f1_score , average='macro')



#Balanced 

# create the grid

max_depth = [5, 10, 20,100,200]

param_grid = dict(max_depth=max_depth)



# search the grid

grid = GridSearchCV(estimator=dt_model2, 

                    param_grid=param_grid,

                    scoring = f1,

                    cv= 2,

                    verbose=2,

                    n_jobs=-1)



dt_best = grid.fit(X_train_resam, y_train_resam)

dt_estimator = grid.best_estimator_

dt_parameters = grid.best_params_

dt_score = grid.best_score_



print('Decision Tree on Balanced Data')

print('Best Estimator: ', dt_estimator)

print('Best Parameter: ', dt_parameters)

print('Best Score: ', dt_score)
# Compare F1_score values between models

fig,axis = plt.subplots(figsize=(12, 6))

f1_x = ['Linear Support Vector','Logistic Regression','K-Nearest Neighbors','Random Forest','Decsion Tree']

f1_y = [lvc_score2,lr_score2,knn_score2,rf_score2,dt_score2]

ax = sns.barplot(x=f1_x, y=f1_y,palette='plasma')

plt.title('Classification Model F1 Score Values',fontsize=14)

plt.ylabel('F1 Score')

plt.xticks(rotation=90)



for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2, p.get_y() + p.get_height(), round(p.get_height(),2), fontsize=12, ha="center", va='bottom')

plt.show()
# Compare Precision score values between models

fig,axis = plt.subplots(figsize=(12, 6))

f1_x = ['Linear Support Vector','Logistic Regression','K-Nearest Neighbors','Random Forest','Decsion Tree']

f1_y = [precision_lvc2,precision_lr2,precision_knn2,precision_rf2,precision_dt2]

ax = sns.barplot(x=f1_x, y=f1_y,palette='plasma')

plt.title('Classification Model Precision Score Values',fontsize=14)

plt.ylabel('F1 Score')

plt.xticks(rotation=90)



for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2, p.get_y() + p.get_height(), round(p.get_height(),2), fontsize=12, ha="center", va='bottom')

plt.show()
# Compare Precision score values between models

fig,axis = plt.subplots(figsize=(12, 6))

f1_x = ['Linear Support Vector','Logistic Regression','K-Nearest Neighbors','Random Forest','Decsion Tree']

f1_y = [recall_lvc2,recall_lr2,recall_knn2,recall_rf2,recall_dt2]

ax = sns.barplot(x=f1_x, y=f1_y,palette='plasma')

plt.title('Classification Model Precision Score Values',fontsize=14)

plt.ylabel('F1 Score')

plt.xticks(rotation=90)



for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2, p.get_y() + p.get_height(), round(p.get_height(),2), fontsize=12, ha="center", va='bottom')

plt.show()
f1 = make_scorer(f1_score , average='macro')

classifier = [RandomForestClassifier(n_estimators = 2000, max_depth = 20),LinearSVC(C = 1, max_iter = 500)]

cross_val = []

for c in classifier:

    cross_val.append(np.sqrt(abs(cross_val_score(c, X_train, y=y_train, scoring= f1, cv=KFold(n_splits=5, random_state=0, shuffle=True)))))

cross_val_mean = [i.mean() for i in cross_val] 

cross_val_df = pd.DataFrame({"Model": ["RandomForest", "LinearSVC"],"F1 Score": cross_val_mean})

pd.DataFrame(cross_val_df.sort_values("F1 Score", ascending=True))
#RUN MODEL AGAIN

modelstart= time.time()

final_model = LinearSVC(C = 1, max_iter = 500, penalty = 'l2')

final_model.fit(X_train, y_train)

y_pred_final = final_model.predict(X_test)

final_score = f1_score(y_test, y_pred_final, average = 'macro')

print("Testing: Linear Support Vector")

print('Final F1 Score: ', final_score)

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))
print(metrics.classification_report(y_test,y_pred_final))
print(metrics.confusion_matrix(y_test,y_pred_final))
print(metrics.accuracy_score(y_test,y_pred_final))
#Log Parameters of all models

params ={'LVC_model type': 'LVC',

          'scaler': 'standard scaler',

         'params': str(lvc_parameters),

         

         'LR_model type': 'logisticregression',

         'LR_Params': str(lr_parameters),

         

         'KNN_model type': 'knn',

         'KNN_Params': str(knn_parameters),

         

         'RF_model type': 'randomforest',

         'RF_Params': str(rf_parameters),

         

         'DT_model type': 'decisiontree',

         'DT_Params': str(dt_parameters),

         

         'stratify':True

}
#Log metrics of all models 

metrics = {'LVC_F1': lvc_score2,

          'LR_F1':lr_score2,

          'KNN_F1':knn_score2,

          'RF_F1':rf_score2,

          'DT_F1':dt_score2,

           

           'LVC_Recall':recall_lvc2,

           'LR_Recall':recall_lr2,

           'KNN_Recall':recall_knn2,

           'RF_Recall':recall_rf2,

           'DT_Recall':recall_dt2,

           

           'LVC_Precision':precision_lvc2,

           'LR_Precision':precision_lr2,

           'KNN_Precision':precision_knn2,

           'RF_Precision':precision_rf2,

           'DT_Precision':precision_dt2,

           

          }
experiment.log_parameters(params)

experiment.log_metrics(metrics)
experiment.end()
experiment.display()
#SUBMISSION TO KAGGLE

submission = final_model.predict(test)

df_test['sentiment'] = submission

df_test.sentiment = df_test.sentiment.astype(int)

final_submission = df_test[['tweetid','sentiment']]

final_submission.to_csv('submission.csv', index=False)