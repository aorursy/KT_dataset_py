import nltk
from nltk.corpus import movie_reviews
movie_reviews.fileids()[:5]
len(movie_reviews.fileids())
movie_reviews.fileids()[:5]
movie_reviews.fileids()[-5:]
neg_fileids=movie_reviews.fileids('neg')

pos_fileids=movie_reviews.fileids('pos')
len(neg_fileids),len(pos_fileids)
print(movie_reviews.raw(fileids=pos_fileids[0]))
movie_reviews.words(fileids=pos_fileids[0])
import string

string.punctuation
useless_words=nltk.corpus.stopwords.words("english")+list(string.punctuation)

useless_words[:10]
def build_bag_of_words_features_filtered(words):

    return {

        word:1 for word in words if not word in useless_words}
negative_features = [

    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'neg') \

    for f in neg_fileids

]
negative_features[:1]
positive_features = [

    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'pos') \

    for f in pos_fileids

]
positive_features[:1]
from nltk.classify import NaiveBayesClassifier
split = 800
sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])
nltk.classify.util.accuracy(sentiment_classifier, positive_features[:split]+negative_features[:split])*100
nltk.classify.util.accuracy(sentiment_classifier, positive_features[split:]+negative_features[split:])*100