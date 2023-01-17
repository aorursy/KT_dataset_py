import numpy as np

import pandas as pd

import sklearn



docs = pd.read_csv('../input/basicnaivedata/example_train1.csv') 

#text in column 1, classifier in column 2.

docs
# convert label to a numerical variable

docs['Class'] = docs.Class.map({'cinema':0, 'education':1})

docs
numpy_array = docs.as_matrix()

X = numpy_array[:,0]

Y = numpy_array[:,1]

Y = Y.astype('int')

print("X")

print(X)

print("Y")

print(Y)
# create an object of CountVectorizer() class 

from sklearn.feature_extraction.text import CountVectorizer 

vec = CountVectorizer( )
vec.fit(X)

vec.vocabulary_
# removing the stop words

vec = CountVectorizer(stop_words='english' )

vec.fit(X)

vec.vocabulary_
# printing feature names

print(vec.get_feature_names())

print(len(vec.get_feature_names()))
# another way of representing the features

X_transformed=vec.transform(X)

X_transformed
print(X_transformed)
# converting transformed matrix back to an array

# note the high number of zeros

X=X_transformed.toarray()

X
# converting matrix to dataframe

pd.DataFrame(X, columns=vec.get_feature_names())
test_docs = pd.read_csv('../input/basicnaivedata/example_train1.csv') 

#text in column 1, classifier in column 2.

test_docs
# convert label to a numerical variable

test_docs['Class'] = test_docs.Class.map({'cinema':0, 'education':1})

test_docs
test_numpy_array = test_docs.as_matrix()

X_test = test_numpy_array[:,0]

Y_test = test_numpy_array[:,1]

Y_test = Y_test.astype('int')

print("X_test")

print(X_test)

print("Y_test")

print(Y_test)
X_test_transformed=vec.transform(X_test)

X_test_transformed
X_test=X_test_transformed.toarray()

X_test
# building a multinomial NB model

from sklearn.naive_bayes import MultinomialNB



# instantiate NB class

mnb=MultinomialNB()



# fitting the model on training data

mnb.fit(X,Y)



# predicting probabilities of test data

mnb.predict_proba(X_test)

proba=mnb.predict_proba(X_test)

print("probability of test document belonging to class CINEMA" , proba[:,0])

print("probability of test document belonging to class EDUCATION" , proba[:,1])
pd.DataFrame(proba, columns=['Cinema','Education'])
from sklearn.naive_bayes import BernoulliNB



# instantiating bernoulli NB class

bnb=BernoulliNB()



# fitting the model

bnb.fit(X,Y)



# predicting probability of test data

bnb.predict_proba(X_test)

proba_bnb=bnb.predict_proba(X_test)
pd.DataFrame(proba_bnb, columns=['Cinema','Education'])