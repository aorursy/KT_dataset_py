import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from pylab import ceil

from __future__ import print_function



#%% Import data

filename = '../input/pokemon.csv'

df = pd.read_csv(filename)

# Fix capture rate for Minior, then convert data to numeric

df.capture_rate.iloc[773] = 255  

df.capture_rate = pd.to_numeric(df.capture_rate)
print('Number of samples = number of Pokemon = ', df.shape[0])
# Processing Ability data

# Ability data is in a single string format: "['Ability1','Ability2']"

# Change it to a list of strings

def abilities_to_list(abil):

    abil_ = abil.replace(' ','')

    abil_ = abil_[2:-2].split("','")

    return abil_



#%%

# Cycle through ability data and transform it to list of string format

# Note: this step is slow

for n in range(len(df.abilities)):

    df.abilities.iloc[n] = abilities_to_list(df.abilities.iloc[n])

            

# Create a new dataframe containing the relevant information            

dfAbilities = df.loc[:,['pokedex_number','name','abilities','type1','type2']]

for n in range(len(dfAbilities.abilities)):

    dfAbilities.abilities.iloc[n] = " ".join(dfAbilities.abilities.iloc[n])
# Produce a list of unique abilities across all Pokemon    

abilitiesList = []

for n in range(len(df.abilities)):

    for ability in df.abilities.iloc[n]:

        if ability not in abilitiesList:

            abilitiesList.append(ability)

            

print('Number of unique abilities: ', len(abilitiesList))
# Define a function that splits our dataframe randomly into 

# train and test sets        

def train_test_split_manual(df, split=0.9):

    n_samples = df.shape[0]

    n_test = int(ceil(split*n_samples))

    indices = np.random.permutation(n_samples)

    X_train = df.iloc[indices[:n_test]]

    X_test = df.iloc[indices[n_test:]]

    return X_train, X_test

    

# Set the random seed so the results are the same each time we run this kernel

np.random.seed(801)

# Split the data

X_train, X_test = train_test_split_manual(dfAbilities)
#%% Feature Extration: Build dictionary of features from Abilities

# CountVectorizer creates sparse matrices of 

# counts of each word in the dictionary

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()



X_train_counts = count_vect.fit_transform(X_train.abilities.values)          



# Apply tf-idf downscaling since some Pokemon have more abilities than others

# Using method from scikit-learn tutorial

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Check the shape of the feature array

print(X_train_tfidf.shape)
# Try to predict a pokemon's type1 and type2 from its abilities

# Use a Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB

# Fit the classifier to Type 1 training data:

classifier = MultinomialNB().fit(X_train_tfidf, X_train.type1.values)



# Transform the testing set to sparse feature matrices of counts

X_test_counts = count_vect.transform(X_test.abilities.values)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)



predicted = classifier.predict(X_test_tfidf)

# Check the fraction of correct predictions

print("Fraction of correct predictions: Abilities predicting Type 1:")

print(predicted[predicted == X_test.type1.values].shape[0], '/', predicted.shape[0])

print((predicted[predicted == X_test.type1.values].shape[0])/float(predicted.shape[0]))

# Try the prediction WITHOUT tf-idf downscaling

classifier = MultinomialNB().fit(X_train_counts, X_train.type1.values)

predicted = classifier.predict(X_test_counts)

# Check the fraction of correct predictions

print("Fraction of correct predictions: Abilities predicting Type 1 without tf-idf:")

print(predicted[predicted == X_test.type1.values].shape[0], '/', predicted.shape[0])

print((predicted[predicted == X_test.type1.values].shape[0])/float(predicted.shape[0]))
# Fit the classifier to Type 2 training data:

# Preprocess type 2 data, since it contains null values when Pokemon

# have only one type and has mixed datatypes

type2 = np.array([str(value) for value in X_train.type2.values])

indices = type2 != 'nan'

classifier = MultinomialNB().fit(X_train_counts[indices], type2[indices])

type2_test = np.array([str(value) for value in X_test.type2.values])

indices_test = type2_test != 'nan'

predicted = classifier.predict(X_test_counts[indices_test])

# Check the fraction of correct predictions

print("Fraction of correct predictions: Abilities predicting Type 2 without tf-idf:")

print((predicted[predicted == type2_test[indices_test]].shape[0])/float(predicted.shape[0]))
