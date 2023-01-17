# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import random as rnd

from numpy import append

import csv

from string import punctuation, digits

import nltk



from nltk.corpus import stopwords

nltk.download('stopwords')

stop = stopwords.words('english')





with open("../input/fake-and-real-news-dataset/True.csv") as csvfile:

    true = np.array(list(csv.DictReader(csvfile)))





with open("../input/fake-and-real-news-dataset/Fake.csv") as csvfile:

    false = np.array(list(csv.DictReader(csvfile)))







tf = np.append(true , false)



stop





def extract_words(input_string):

    

    for c in punctuation + digits:

        input_string = input_string.replace(c, ' ' + c + ' ')



    return input_string.lower().split()





def bag_of_words(texts):

    dictionary = {} 

    for text in texts:

        word_list = extract_words(text)

        for word in word_list:

            if (word not in dictionary) and (word not in stop):

                dictionary[word] = len(dictionary)

    return dictionary



def extract_bow_feature_vectors(reviews, dictionary):

    """

    Inputs a list of string reviews

    Inputs the dictionary of words as given by bag_of_words

    Returns the bag-of-words feature matrix representation of the data.

    The returned matrix is of shape (n, m), where n is the number of reviews

    and m the total number of entries in the dictionary.



    Feel free to change this code as guided by Problem 9

    """



    num_reviews = len(reviews)

    feature_matrix = np.zeros([num_reviews, len(dictionary)])



    for i, text in enumerate(reviews):

        word_list = extract_words(text)

        for word in word_list:

            if word in dictionary:

                feature_matrix[i, dictionary[word]] = 1

    return feature_matrix





train = []

f = [1]

w = []

s = []



for i in range(len(tf)):

    s.append(tf[i]['text'])



for i in range(50):

    train.append(true[i]['text'])

    w.append(1)

    train.append(false[i]['text'])

    w.append(-1)





for i in range(len(true)-1):

    f.append([1])



for i in range(len(false)):

    f.append([-1])





dictionary = bag_of_words(train)



train_vector = extract_bow_feature_vectors(train, dictionary)



test_vector = extract_bow_feature_vectors(s, dictionary)

def perceptron_single_step_update(

        feature_vector,

        label,

        current_theta,

        current_theta_0):

    """

    Properly updates the classification parameter, theta and theta_0, on a

    single step of the perceptron algorithm.



    Args:

        feature_vector - A numpy array describing a single data point.

        label - The correct classification of the feature vector.

        current_theta - The current theta being used by the perceptron

            algorithm before this update.

        current_theta_0 - The current theta_0 being used by the perceptron

            algorithm before this update.



    Returns: A tuple where the first element is a numpy array with the value of

    theta after the current update has completed and the second element is a

    real valued number with the value of theta_0 after the current updated has

    completed.

    """

    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 0:

        current_theta += label * feature_vector

        current_theta_0 += label

    return (current_theta, current_theta_0)    





#pragma: coderesponse template

def perceptron (feature_matrix, labels, T):

    """

    Runs the full perceptron algorithm on a given set of data. Runs T

    iterations through the data set, there is no need to worry about

    stopping early.



    Args:

        feature_matrix -  A numpy matrix describing the given data. Each row

            represents a single data point.

        labels - A numpy array where the kth element of the array is the

            correct classification of the kth row of the feature matrix.

        T - An integer indicating how many times the perceptron algorithm

            should iterate through the feature matrix.



    Returns: A tuple where the first element is a numpy array with the value of

    theta, the linear classification parameter, after T iterations through the

    feature matrix and the second element is a real number with the value of

    theta_0, the offset classification parameter, after T iterations through

    the feature matrix.

    """



    (nsamples, nfeatures) = feature_matrix.shape

    theta = np.zeros(nfeatures)

    theta_0 = 0.0    

    for t in range(T):

        for i in range(len(feature_matrix)):

            theta, theta_0 = perceptron_single_step_update(

                feature_matrix[i], labels[i], theta, theta_0)

    return (theta, theta_0)





def accuracy(preds, targets):

 

    s = len(preds)

    e = 0

    for i in range(s):

        if(preds[i] == targets[i]):

            e+=1

    return (e/s)*100





def classify(feature_matrix, theta, theta_0):

    """

    A classification function that uses theta and theta_0 to classify a set of

    data points.



    Args:

        feature_matrix - A numpy matrix describing the given data. Each row

            represents a single data point.

                theta - A numpy array describing the linear classifier.

        theta - A numpy array describing the linear classifier.

        theta_0 - A real valued number representing the offset parameter.



    Returns: A numpy array of 1s and -1s where the kth element of the array is

    the predicted classification of the kth row of the feature matrix using the

    given theta and theta_0. If a prediction is GREATER THAN zero, it should

    be considered a positive classification.

    """ 

    (nsamples, nfeatures) = feature_matrix.shape

    predictions = np.zeros(nsamples)

    for i in range(nsamples):

        feature_vector = feature_matrix[i]

        prediction = np.dot(theta, feature_vector) + theta_0

        if (prediction > 0):

            predictions[i] = 1

        else:

            predictions[i] = -1

    return predictions





def classifier_accuracy(

        classifier,

        train_feature_matrix,

        val_feature_matrix,

        train_labels,

        val_labels,

        T):

    """

    Trains a linear classifier and computes accuracy.

    The classifier is trained on the train data. The classifier's

    accuracy on the train and validation data is then returned.



    Args:

        classifier - A classifier function that takes arguments

            (feature matrix, labels, **kwargs) and returns (theta, theta_0)

        train_feature_matrix - A numpy matrix describing the training

            data. Each row represents a single data point.

        val_feature_matrix - A numpy matrix describing the training

            data. Each row represents a single data point.

        train_labels - A numpy array where the kth element of the array

            is the correct classification of the kth row of the training

            feature matrix.

        val_labels - A numpy array where the kth element of the array

            is the correct classification of the kth row of the validation

            feature matrix.

        **kwargs - Additional named arguments to pass to the classifier

            (e.g. T or L)



    Returns: A tuple in which the first element is the (scalar) accuracy of the

    trained classifier on the training data and the second element is the

    accuracy of the trained classifier on the validation data.

    """

    theta, theta_0 = classifier(train_feature_matrix, train_labels, T)

    train_predictions = classify(train_feature_matrix, theta, theta_0)

    val_predictions = classify(val_feature_matrix, theta, theta_0)

    train_accuracy = accuracy(train_predictions, train_labels)

    validation_accuracy = accuracy(val_predictions, val_labels)

    return (train_accuracy, validation_accuracy)





tr_ac , tf_ac = classifier_accuracy(perceptron,train_vector,test_vector,w,f,10) 

print(tf_ac)