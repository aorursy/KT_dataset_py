from nltk.corpus import brown
from nltk import FreqDist

suffix_fdist = FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
    
suffix_fdist
common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
common_suffixes[:10]
def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features

pos_features('test')
tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
featuresets[0]
from nltk import DecisionTreeClassifier
from nltk.classify import accuracy

cutoff = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
classifier = DecisionTreeClassifier.train(train_set) # NLTK is a teaching toolkit which is not really optimized for speed. Therefore, this may take forever. For speed, use scikit-learn for the classifiers.
accuracy(classifier, test_set)
classifier.classify(pos_features('cats'))
classifier.pseudocode(depth=4)