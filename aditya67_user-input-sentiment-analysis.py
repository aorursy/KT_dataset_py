import nltk
pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

tweets = []
for (words,sentiment) in pos_tweets + neg_tweets:
    filtered = [w.lower() for w in words.split() if len(w) >= 3]
    tweets.append((filtered,sentiment))
def words_tweets(tweets):
    all_words = []
    for (word,sentiment) in tweets:
        all_words.extend(word)
    return all_words
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    print(wordlist.most_common(10))
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(words_tweets(tweets))
def extract_features(doc):
    doc_words = set(doc)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in doc_words)
    return features
extract_features(['love','this','car'])
training_set = nltk.classify.apply_features(extract_features, tweets)

print(training_set)
classifier = nltk.NaiveBayesClassifier.train(training_set)
tweet = 'Larry is a good friend'
print(classifier.classify(extract_features(tweet.split())))

tweet = 'Larry is not a good friend'
print(classifier.classify(extract_features(tweet.split())))
