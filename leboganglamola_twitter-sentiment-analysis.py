!pip install comet_ml
from comet_ml import Experiment

experiment = Experiment(api_key="gtqYH8ytl0mOgS2epkg2XgE2X",

                        project_name="general", workspace="bryan981")

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import string

import nltk

%matplotlib inline



from nltk import SnowballStemmer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import train_test_split

from nltk.tokenize.treebank import TreebankWordDetokenizer

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from wordcloud import WordCloud

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from sklearn.preprocessing import MaxAbsScaler, StandardScaler
# Importing The dataset

# Training Datasets

df_path1 = r"../input/climate-change-belief-analysis/train.csv"

train = pd.read_csv(df_path1)



# Testing Datasets

df_path2 = r"../input/climate-change-belief-analysis/test.csv"

test = pd.read_csv(df_path2)
variables = pd.DataFrame(["sentiment", "message", "tweetid"], columns = ["variables"])

variables["definition"] = pd.DataFrame(["sentiment of twitter messages", "twitter messages", "twitter unique id"])

variables.head()
train.head()
test.head()
# Text Preprocessing

def data_cleaner(input_df):

    train2 = input_df.copy()

    # Removing Twitter Handles (@user)

    def remove_pattern(user_input, pattern):

        

        '''Twitter handles are masked as @user due to concerns

        surrounding privacy.Twitter handles do note 

        necessarily give necessary information around the 

        overall tweet.We will remove them'''

    

        r = re.findall(pattern, user_input)

    

        for element in r:

            user_input = re.sub(element, "", user_input)

    

        return user_input

    train2["message"] = np.vectorize(remove_pattern)(train2["message"], "@[\w]*")



    # Remove Special Characters,Numbers And Punctuations

    train2["message"] = train2["message"].str.replace("[^a-zA-Z#]", " ") 

    

    # Substituting multiple spaces with single space

    #train2["message"] = train2["message"].str.replace(r'\s+', ' ', flags = re.I)

    

    # Converting to Lowercase

    train2["message"] = train2["message"].apply(lambda x: x.lower())



    # Removing Short Words

    train2["message"] = train2["message"].apply(lambda x: ' '.join([word for word in x.split() if len(word)>3]))



    # Tokenization

    train2["message"] = train2["message"].apply(lambda x: nltk.word_tokenize(x))



    # Remove Stop Words

    def stop_words(user_input):

        

        '''A vast majoity of short words which are less than 3 letters long.

        Words like 'ohh' and 'lol' do not give us much 

        information, thus important to remove them'''

        

        stop_words = set(stopwords.words('english'))

        wordslist = [word for word in user_input if not word in stop_words]

    

        return wordslist

    train2["message"] = train2["message"].apply(lambda x: stop_words(x))



    # Lemmatization

    stemmer = SnowballStemmer("english")

    train2["message"] = train2["message"].apply(lambda x: [stemmer.stem(word) for word in x])

    

    # Untokenization

    train2["message"] = train2["message"].apply(lambda x: TreebankWordDetokenizer().detokenize(x))

    

    return train2
train_df = data_cleaner(train)

train_df.head()
test_df = data_cleaner(test)

test_df.head()
train.describe().astype(int)
unclean_tweets = train["message"].str.len()

clean_tweets = train_df["message"].str.len()



plt.hist(unclean_tweets, label = 'Uncleaned_Tweet')

plt.hist(clean_tweets, label = 'Cleaned_Tweet')

plt.legend()

plt.show()
# Visualizing All Words In Clean Train Dataset

allWords = ' '.join([text for text in train_df["message"]])

wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110).generate(allWords)

plt.figure(figsize=(10, 10)) 

plt.imshow(wordcloud, interpolation = "bilinear")

plt.title("Most Common Words")

plt.axis("off") 

plt.show()
# Visualizing All The Positive Words In Clean Train Dataset

positive_words =' '.join([text for text in train_df["message"][train_df["sentiment"] == 1]])

wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110).generate(positive_words) 

plt.figure(figsize=(10, 10))

plt.imshow(wordcloud, interpolation = "bilinear")

plt.title("Most Common Positive Words")

plt.axis("off")

plt.show()
# Visualizing All The Negative Words In Clean Train Dataset

negative_words =' '.join([text for text in train_df["message"][train_df["sentiment"] == -1]])

wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110).generate(negative_words) 

plt.figure(figsize=(10, 10))

plt.imshow(wordcloud, interpolation = "bilinear")

plt.title("Most Common Negative Words")

plt.axis("off")

plt.show()
# Visualizing All The Neutral Words In Clean Train Dataset

normal_words =' '.join([text for text in train_df["message"][train_df["sentiment"] == 0]])

wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110).generate(normal_words) 

plt.figure(figsize=(10, 10))

plt.imshow(wordcloud, interpolation = "bilinear")

plt.title("Most Common Neutral Words")

plt.axis("off")

plt.show()
# Visualizing All The News Broadcast Words In Clean Train Dataset

broadcast_words =' '.join([text for text in train_df["message"][train_df["sentiment"] == 2]])

wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size = 110).generate(broadcast_words) 

plt.figure(figsize=(10, 10))

plt.imshow(wordcloud, interpolation = "bilinear")

plt.title("Most Common News Broadcast Words")

plt.axis("off")

plt.show()
# Function To Extract hashtags

def hashtag_extract(user_input):

    hashtags = []

    # Loop over the words in the tweet

    for text in user_input:

        ht = re.findall(r"#(\w+)", text)

        hashtags.append(ht)



    return hashtags
# Extracting Hashtags From News Broadcast

HT_news = hashtag_extract(train_df["message"][train_df["sentiment"] == 2])

# Extracting Hashtags From Positive Sentiments

HT_positive = hashtag_extract(train_df["message"][train_df["sentiment"] == 1])

# Extracting Hashtags From Neutral Sentiments

HT_normal = hashtag_extract(train_df["message"][train_df["sentiment"] == 0])

# Extracting Hashtags From Negative Sentiments

HT_negative = hashtag_extract(train_df["message"][train_df["sentiment"] == -1])



# Unnesting List

ht_news = sum(HT_news,[])

ht_positive = sum(HT_positive,[])

ht_normal = sum(HT_normal,[])

ht_negative = sum(HT_negative,[])
# News Broadcast

a = nltk.FreqDist(ht_news)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     

d = d.nlargest(columns = "Count", n = 20) 

plt.figure(figsize = (16,5))

ax = sns.barplot(data = d, x = "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
# Positive Sentiments

a = nltk.FreqDist(ht_positive)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     

d = d.nlargest(columns = "Count", n = 20) 

plt.figure(figsize = (16,5))

ax = sns.barplot(data = d, x = "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
# Neutral Sentiments

a = nltk.FreqDist(ht_normal)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     

d = d.nlargest(columns = "Count", n = 20) 

plt.figure(figsize = (16,5))

ax = sns.barplot(data = d, x = "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
# Negative Sentiments

a = nltk.FreqDist(ht_negative)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     

d = d.nlargest(columns = "Count", n = 20) 

plt.figure(figsize = (16,5))

ax = sns.barplot(data = d, x = "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
test_df.describe().astype(int)
unclean_tweets = test["message"].str.len()

clean_tweets = test_df["message"].str.len()



plt.hist(unclean_tweets, label = 'Uncleaned_Tweet')

plt.hist(clean_tweets, label = 'Cleaned_Tweet')

plt.legend()

plt.show()
font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 16,

        }



fig = plt.figure()

train_df['sentiment'].value_counts().plot(kind = 'bar', color = 'darkred')

plt.xlabel("sentiment", fontdict = font)

plt.ylabel("Number Of Tweets", fontdict = font)

plt.title("Dataset labels distribution")

plt.show()
# Create A Column That Contains The Lengths Of The Messages

train_df["message_len"] = train_df["message"].apply(len)

sns_g = sns.FacetGrid(train_df, col = "sentiment")

sns_g.map(plt.hist, "message_len")
# Split Into X And Y

y = train_df["sentiment"]

X = train_df.drop(["sentiment"], axis = 1)
count_vect = CountVectorizer()

X_counts = count_vect.fit_transform(X["message"])

X_counts.shape

count_vect.vocabulary_.get(u'algorithm')



test_count = count_vect.transform(test_df["message"])
sc = MaxAbsScaler().fit(X_counts)

X_counts = sc.transform(X_counts)

test_count = sc.transform(test_count)
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size = 0.3, random_state = 42)
lr_clf = LogisticRegression(max_iter = 2000)

lr = lr_clf.fit(X_train, y_train)
# LOGISTIC REGRESSION

lr_pred = lr_clf.predict(X_test)

print('Classification Report')

print("F1-Score:", classification_report(y_test.astype(int), lr_pred))
svc_clf = SVC(kernel = "linear")

svc = svc_clf.fit(X_train, y_train)
svc_pred = svc_clf.predict(X_test)

print('Classification Report')

print("F1-Score:", classification_report(y_test.astype(int), svc_pred))
nb_clf = MultinomialNB()

nb = nb_clf.fit(X_train, y_train)
nb_pred = nb_clf.predict(X_test)

print('Classification Report')

print("F1-Score:", classification_report(y_test.astype(int), nb_pred))
test_pred = lr_clf.predict(test_count)
submission = pd.DataFrame(test_df["tweetid"], columns = ["tweetid"])

submission["sentiment"] = test_pred
submission.head()
#create a csv submission fo Kaggle

submission.to_csv('submission.csv', index=False)
# Saving Each Metrics To Add To A Dictionary For Logging

cr = classification_report(y_test.astype(int), lr_pred)
# Create A Dictionaries For The Data We Want To Log

params = {"random_state": 42,

          "model_type": "Logistic Regression",

          "scaler": "MaxAbsScaler",}



metrics = {"classification_report": cr,}
# Log Our Parameters And Results

experiment.log_parameters(params)

experiment.log_metrics(metrics)
experiment.end()
experiment.display()