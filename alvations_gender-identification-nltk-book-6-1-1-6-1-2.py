def gender_features(word):

    return {'last_letter': word[-1]}



gender_features('Shrek')
import random

from nltk.corpus import names



# Read the names from the files.

# Label each name with the corresponding gender.

male_names = [(name, 'male') for name in names.words('male.txt')]

female_names = [(name, 'female') for name in names.words('female.txt')]



# Combine the lists.

labeled_names = male_names + female_names



# Shuffle the list.

random.shuffle(labeled_names)
from nltk import NaiveBayesClassifier



# Extract the features using the `gender_features()` function.

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]



# Split the dataset into train and test set.

train_set, test_set = featuresets[500:], featuresets[:500]



# Train a Naive Bayes classifier

classifier = NaiveBayesClassifier.train(train_set)
neo_gender = classifier.classify(gender_features('Neo'))

trinity_gender = classifier.classify(gender_features('Trinity'))

print("Neo is most probably a {}.".format(neo_gender))

print("Trinity is most probably a {}.".format(trinity_gender))
from nltk.classify import accuracy

print(accuracy(classifier, test_set))
classifier.show_most_informative_features(5)
# For instance, instance of using the last character 

# in the name, we add a new feature that takes 

# the last 2 characters of the name and we

# redefine the `gender_features()` function.



def gender_features(word):

    return {'last_letter': word[-1], 

            'last_two_letters': word[-2:]}



gender_features('Shrek')
from nltk.classify import apply_features



# Instead of looping through a list comprehension as we have

# demonstrated above that returns a list. 

train_set_old = [(gender_features(n), gender) for (n, gender) in labeled_names[500:]]

test_set_old = [(gender_features(n), gender) for (n, gender) in labeled_names[:500]]



# We can simply use the `apply_features()` to return a generator instead.

train_set = apply_features(gender_features, labeled_names[500:])

test_set = apply_features(gender_features, labeled_names[:500])



type(train_set_old), type(train_set)
# You can train a Naive Bayes classifier with either 

# `list` and `LazyMap` input, e.g.

classifier_old = NaiveBayesClassifier.train(train_set_old)

classifier = NaiveBayesClassifier.train(train_set)



# And we see that the accuracy is the same.

accuracy(classifier, test_set) == accuracy(classifier_old, test_set_old) 