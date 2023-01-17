import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Library for drawing plots.

import seaborn as sns # An extension of matplotlib that helps make plots easier in less code

import nltk

import string
df_train = pd.read_csv('../input/train.csv') # Train dataset

df_test = pd.read_csv('../input/test.csv') # Test dataset
df_train.sample(10) # Look at 10 random samples of the dataframe


# We create a function to do the punctuation removal

def remove_punctuation(text):



    # For each punctuation in our list

    for punct in string.punctuation:

        # Replace the actual punctuation with a space.

        text = text.replace(punct,'')



    # Return the new text

    return text



# Now, we will apply the remove punctiation to all our text

df_train.text = df_train.text.apply(remove_punctuation)

df_test.text = df_test.text.apply(remove_punctuation)
APPLY_STEMMING = False



if APPLY_STEMMING:

    import nltk.stem as stm # Import stem class from nltk

    import re

    stemmer = stm.PorterStemmer()



    # Crazy one-liner code here...

    # Explanation above...

    df_train.text = df_train.text.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

    df_test.text = df_test.text.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))
from sklearn.feature_extraction.text import CountVectorizer # Import the library to vectorize the text



# Instantiate the count vectorizer with an NGram Range from 1 to 3 and english for stop words.

count_vect = CountVectorizer(ngram_range=(1,3),stop_words='english')



# Fit the text and transform it into a vector. This will return a sparse matrix.

count_vectorized = count_vect.fit_transform(df_train.text)
from sklearn.feature_extraction.text import TfidfVectorizer # Import the library to vectorize the text



# Instantiate the count vectorizer with an NGram Range from 1 to 3 and english for stop words.

tfidf_vect = TfidfVectorizer(ngram_range=(1,3), stop_words='english')



# Fit the text and transform it into a vector. This will return a sparse matrix.

tfidf_vectorized = tfidf_vect.fit_transform(df_train.text)
from sklearn.model_selection import train_test_split # Import the function that makes splitting easier.



# Split the vectorized data. Here we pass the vectorized values and the author column.

# Also, we specify that we want to use a 75% of the data for train, and the rest for test.



###########################

# COUNT VECTORIZED TOKENS #

###########################

X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(count_vectorized, df_train.author, train_size=0.75)



###########################

# TFIDF VECTORIZED TOKENS #

###########################

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_vectorized, df_train.author, train_size=0.75)
# First, import the Multinomial Naive bayes library from sklearn 

from sklearn.naive_bayes import MultinomialNB



# Instantiate the model.

# One for Count Vectorized words

model_count_NB = MultinomialNB()

# One for TfIdf vectorized words

model_tfidf_NB = MultinomialNB()



# Train the model, passing the x values, and the target (y)

model_count_NB.fit(X_train_count, y_train_count)

model_tfidf_NB.fit(X_train_tfidf, y_train_tfidf)
# Predict the values, using the test features for both vectorized data.

predictions_count = model_count_NB.predict(X_test_count)

predictions_tfidf = model_tfidf_NB.predict(X_test_tfidf)
# Primero calculamos el accuracy general del modelo

from sklearn.metrics import accuracy_score

accuracy_count = accuracy_score(y_test_count, predictions_count)

accuracy_tfidf = accuracy_score(y_test_tfidf, predictions_tfidf)

print('Count Vectorized Words Accuracy:', accuracy_count)

print('TfIdf Vectorized Words Accuracy:', accuracy_tfidf)
# Import the confusion matrix method from sklearn

from sklearn.metrics import confusion_matrix



# Calculate the confusion matrix passing the real values and the predicted ones

# Count

conf_mat_count = confusion_matrix(y_test_count, predictions_count)

# tfIdf

conf_mat_tfidf = confusion_matrix(y_test_tfidf, predictions_tfidf)



# Set plot size

plt.figure(figsize=(12,10))

# Use 2 subplots.

plt.subplot(1,2,1)



# Finally, plot the confusion matrix using seaborn's heatmap.

sns.heatmap(conf_mat_count.T, square=True, annot=True, fmt='d', cbar=True,

            xticklabels=y_test_count.unique(), yticklabels=y_test_count.unique())

plt.xlabel('True values')

plt.ylabel('Predicted Values');

plt.title('Count Vectorizer', fontsize=16)



plt.subplot(1,2,2)

# Finally, plot the confusion matrix using seaborn's heatmap.

sns.heatmap(conf_mat_tfidf.T, square=True, annot=True, fmt='d', cbar=True,

            xticklabels=y_test_tfidf.unique(), yticklabels=y_test_tfidf.unique())

plt.xlabel('True values')

plt.ylabel('Predicted Values');

plt.title('TfIdf Vectorizer', fontsize=16)
# Instantiate the model.

model_NB = MultinomialNB()



# Train the model, passing the x values, and the target (y)

# the vectorized variable contains all the test data.

model_NB.fit(count_vectorized, df_train.author)
# Transform the text to a vector, with the same shape of the trained data.

X_test = count_vect.transform(df_test.text)
# Run the prediction

predicted_values = model_NB.predict_proba(X_test)
model_NB.classes_
# Import the time library

import time



# Create the submission dataframe

df_submission = pd.DataFrame({

    'id': df_test.id.values,

    'EAP': predicted_values[:,0],

    'HPL': predicted_values[:,1],

    'MWS': predicted_values[:,2]

})





# Create the date and time string. (year month day _ hours minutes seconds)

datetime = time.strftime("%Y%m%d_%H%M%S")



# generate the file name with the date, the time and the accuracy of the count vectorized test.

filename = 'submission_' + datetime + '_acc_' + str(accuracy_count) + '.csv'



# Finally, convert it to csv. Index=True tells pandas not to include the index as a column

df_submission.to_csv(filename, index=False)



print('File',filename,'created.')