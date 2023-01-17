import pandas as pd



new_york_tweets = pd.read_json("../input/new_york.json", lines=True)

print(len(new_york_tweets))

print(new_york_tweets.columns)

print(new_york_tweets.loc[12]["text"])

new_york_tweets.head()
london_tweets = pd.read_json("../input/london.json", lines=True)

paris_tweets = pd.read_json("../input/paris.json", lines=True)

print(len(london_tweets))

print(len(paris_tweets))
london_tweets.head()
paris_tweets.head()
new_york_text = new_york_tweets["text"].tolist()

london_text = london_tweets["text"].tolist()

paris_text = paris_tweets["text"].tolist()



all_tweets = new_york_text + london_text + paris_text

labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)
from sklearn.model_selection import train_test_split



train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size=0.2, random_state=1)



print(len(train_data))



print(len(test_data))
from sklearn.feature_extraction.text import CountVectorizer



counter = CountVectorizer()

counter.fit(train_data)

train_counts = counter.transform(train_data)

test_counts = counter.transform(test_data)

print(train_data[3])

print(train_counts[3])
from sklearn.naive_bayes import MultinomialNB



classifier = MultinomialNB()

classifier.fit(train_counts, train_labels)

predictions = classifier.predict(test_counts) 

# Be sure you correctly set up counter earlier 

# (use train data to fit, and transform to train/test counts with the same counter, 

# do NOT use test data to fit counter), otherwise dimension mismatch error.
from sklearn.metrics import accuracy_score



print(accuracy_score(test_labels, predictions))
from sklearn.metrics import confusion_matrix



print(confusion_matrix(test_labels, predictions))
tweet = "I want to get married at the top of the Empire State Building!"

tweet_counts = counter.transform([tweet])

print(classifier.predict(tweet_counts))
tweet = "I want to enjoy delicious afternoon tea!"

tweet_counts = counter.transform([tweet])

print(classifier.predict(tweet_counts))
# Funny experience: I tried so many English tweets to get it classified as "Paris" but failed.

# Then I realized, OMG the Paris tweets need to be in French!!! Duh!

tweet = "C'est mon premier voyage en Europe! Trés exité!" #This is my first trip to Europe! So excited!

tweet_counts = counter.transform([tweet])

print(classifier.predict(tweet_counts))