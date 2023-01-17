from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

import csv
file = '../input/record/record_fix.csv'

# tentukan lokasi file, nama file, dan inisialisasi csv
f = open(file, 'r')
reader = csv.reader(f)
stopword = set(stopwords.words('indonesian'))

#start process_tweet
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopword or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end
import re

f = open(file, 'r')
reader = csv.reader(f)
line = f.readline()

print("Case Folding : ")
print()

case_folding = []
for row in reader:
    cf = processTweet(row[0])
    featureVector = getFeatureVector(cf)
    #print(featureVector)
    case_folding.append(featureVector)
import array as arr
import csv

#Read the tweets one by one and process it
inpTweets = csv.reader(open('../input/data-training/testSentimen.csv', 'r'), delimiter=',', quotechar='|')
tweets = []
featureList = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    tweets.append((featureVector, sentiment))
    featureList = featureList + featureVector
#end loop
#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end
pos = 0
neg = 0
neu = 0
import nltk.classify
# Remove featureList duplicates
featureList = list(set(featureList))

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

f = open(file, 'r')
line = f.readline()


# Test the classifier
#testTweet = 'baru saja donor darah, tangan saya masih sakit'
testTweet = 'ujaran'
processedTestTweet = processTweet(testTweet)
sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
print ("testTweet = %s, sentiment = %s\n" % (testTweet, sentiment))
x = 0

print("ANALISIS SENTIMEN")
print()
sentimen=[]
while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
    line = f.readline()
    testTweet = processedTweet
    processedTestTweet = processTweet(testTweet)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
    sentimen.append(sentiment)
    print ("testTweet = %s, sentiment = %s\n" % (featureVector, sentiment))
    
    myData=featureVector
    x = x + 1
    if sentiment == 'positive':
        pos = 1 + pos
    elif sentiment == 'negative':
        neg = 1 + neg
    elif sentiment == 'neutral':
        neu = 1 + neu
 
print('Jumlah Sentiment Positive : ')
print(pos)
print('Jumlah Sentiment Negative : ')
print(neg)
print('Jumlah Sentiment Neutral :')
print(neu)
import matplotlib.pyplot as plt
print('Analisis Sentimen')
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos, neg, neu]
colors = ['steelblue', 'slategray', 'gray']
explode = (0.2, 0.1, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()