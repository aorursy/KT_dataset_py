# Set seed for reproducibility

import random; random.seed(53)



# Import all we need from sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn import metrics
import pandas as pd



# Load data

tweet_df = pd.read_csv('../input/tweets.csv')



# Create target

y = tweet_df['author']



# Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(tweet_df['status'], y, random_state=53, test_size=0.33)
# Initialize count vectorizer

count_vectorizer = CountVectorizer(stop_words='english', min_df=0.05, max_df=0.9)



# Create count train and test variables

count_train = count_vectorizer.fit_transform(X_train, y_train)

count_test = count_vectorizer.transform(X_test)



# Initialize tfidf vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0.05, max_df=0.9)



# Create tfidf train and test variables

tfidf_train = tfidf_vectorizer.fit_transform(X_train, y_train)

tfidf_test = tfidf_vectorizer.transform(X_test)
# Create a MulitnomialNB model

tfidf_nb = MultinomialNB()



# ... Train your model here ...

tfidf_nb.fit(tfidf_train, y_train)



# Run predict on your TF-IDF test data to get your predictions

tfidf_nb_pred = tfidf_nb.predict(tfidf_test)



# Calculate the accuracy of your predictions

tfidf_nb_score = metrics.accuracy_score(tfidf_nb_pred, y_test)



# Create a MulitnomialNB model



count_nb = MultinomialNB()



# ... Train your model here ...

count_nb.fit(count_train, y_train)



# Run predict on your count test data to get your predictions

count_nb_pred = count_nb.predict(count_test)



# Calculate the accuracy of your predictions

count_nb_score = metrics.accuracy_score(count_nb_pred, y_test)



print('NaiveBayes Tfidf Score: ', tfidf_nb_score)

print('NaiveBayes Count Score: ', count_nb_score)
from matplotlib import pyplot as plt

import numpy as np

import itertools





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues,

                          figure=0):

    """

    See full source and example:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figure)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





def plot_and_return_top_features(classifier, vectorizer, top_features=20):

    """

    Plot the top features in a binary classification model and remove possible overlap.



    Adapted from https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

    and https://stackoverflow.com/a/26980472 by @kjam

    """

    class_labels = classifier.classes_

    feature_names = vectorizer.get_feature_names()

    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:top_features]

    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-top_features:]

    top_coefficients = np.hstack([topn_class1, topn_class2])

    if set(topn_class1).union(topn_class2):

        top_coefficients = topn_class1

        for ce in topn_class2:

            if ce not in topn_class1:

                top_coefficients.append(x)



    plt.figure(figsize=(15, 5))

    colors = ['red' if c < 0 else 'blue' for c in [tc[0] for tc in top_coefficients]]

    plt.bar(np.arange(len(top_coefficients)), [tc[0] for tc in top_coefficients], color=colors)

    plt.xticks(np.arange(len(top_coefficients)),

               [tc[1] for tc in top_coefficients], rotation=60, ha='right')

    plt.show()

    return top_coefficients
%matplotlib inline



# Calculate the confusion matrices for the tfidf_nb model and count_nb models

tfidf_nb_cm = metrics.confusion_matrix(y_test, tfidf_nb_pred)

count_nb_cm = metrics.confusion_matrix(y_test, count_nb_pred)



classes = ['Donald J. Trump', 'Justin Trudeau']



# Plot the tfidf_nb_cm confusion matrix

plot_confusion_matrix(tfidf_nb_cm, classes=classes, title="TF-IDF NB Confusion Matrix")



# Plot the count_nb_cm confusion matrix without overwriting the first plot 

plot_confusion_matrix(count_nb_cm, classes=classes, title="Count Vec NB Confusion Matrix", figure=1)
# Create a LinearSVM model

tfidf_svc = LinearSVC()



# ... Train your model here ...

tfidf_svc.fit(tfidf_train, y_train)



# Run predict on your tfidf test data to get your predictions

tfidf_svc_pred = tfidf_svc.predict(tfidf_test)



# Calculate your accuracy using the metrics module

tfidf_svc_score = metrics.accuracy_score(tfidf_svc_pred, y_test)



print("LinearSVC Score:   %0.3f" % tfidf_svc_score)



# Calculate the confusion matrices for the tfidf_svc model

svc_cm = metrics.confusion_matrix(y_test, tfidf_svc_pred)



# Plot the confusion matrix using the plot_confusion_matrix function

plot_confusion_matrix(svc_cm, classes=classes, title="TF-IDF LinearSVC Confusion Matrix")




# Import pprint from pprint

from pprint import pprint



# Get the top features using the plot_and_return_top_features function and your top model and tfidf vectorizer

top_features = plot_and_return_top_features(tfidf_svc, tfidf_vectorizer)



# pprint the top features

pprint(top_features)
# Write two tweets as strings, one which you want to classify as Trump and one as Trudeau

trump_tweet = "i am the president"



trudeau_tweet = "le canada is great"





# Vectorize each tweet using the TF-IDF vectorizer's transform method

trump_tweet_vectorized = tfidf_vectorizer.transform([trump_tweet])

trudeau_tweet_vectorized = tfidf_vectorizer.transform([trudeau_tweet])



# Call the predict method on your vectorized tweets

trump_tweet_pred = tfidf_svc.predict(trump_tweet_vectorized)

trudeau_tweet_pred = tfidf_svc.predict(trudeau_tweet_vectorized)



print("Predicted Trump tweet", trump_tweet_pred)

print("Predicted Trudeau tweet", trudeau_tweet_pred)