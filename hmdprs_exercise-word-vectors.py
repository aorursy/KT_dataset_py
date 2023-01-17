# setup code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex3 import *

print("Setup is completed.")



%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd

review_data = pd.read_csv('../input/nlp-course/yelp_ratings.csv')

review_data.head()
reviews = review_data[:100]



# load the large model to get the vectors

import spacy

nlp = spacy.load('en_core_web_lg')



# turn off other pipes in model, we just want the vectors

import numpy as np

with nlp.disable_pipes():

    vectors = np.array([nlp(review['text']).vector for idx, review in reviews.iterrows()])

    

vectors.shape
# loading all document vectors from file

vectors = np.load('../input/nlp-course/review_vectors.npy')
# split train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    vectors, review_data['sentiment'], test_size=0.1, random_state=1

)



# create the LinearSVC model

from sklearn.svm import LinearSVC

model = LinearSVC(random_state=1, dual=False)



# fit the model

model.fit(X_train, y_train)



# print model accuracy

print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')



# check your work

q_1.check()
# lines below will give you a hint or solution code

# q_1.hint()

# q_1.solution()
# scratch space in case you want to experiment with other models

from sklearn.ensemble import RandomForestClassifier

second_model = RandomForestClassifier(n_jobs=-1)

second_model.fit(X_train, y_train)

print(f'Model test accuracy: {second_model.score(X_test, y_test)*100:.3f}%')
# check your answer (run this code cell to receive credit!)

q_2.solution()
review = """I absolutely love this place. The 360 degree glass windows with the 

Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 

transports you to what feels like a different zen zone within the city. I know 

the price is slightly more compared to the normal American size, however the food 

is very wholesome, the tea selection is incredible and I know service can be hit 

or miss often but it was on point during our most recent visit. Definitely recommend!



I would especially recommend the butternut squash gyoza."""



review_vec = nlp(review).vector





# calculate the mean for the document vectors, should have shape (300,)

vec_mean = vectors.mean(axis=0)



# center the document vectors

centered = vectors - vec_mean



# calculate similarities for each document in the dataset, subtract the mean from the review vector

def cosine_similarity(a, b):

    return np.dot(a, b) / np.sqrt(a.dot(a) * b.dot(b))

sims = np.array([cosine_similarity(review_vec - vec_mean, vec) for vec in centered])



# get the index for the most similar document

most_similar = sims.argmax()



# check your work

q_3.check()
# lines below will give you a hint or solution code

# q_3.hint()

# q_3.solution()
print(review_data.iloc[most_similar].text)
# check your answer (run this code cell to receive credit!)

q_4.solution()