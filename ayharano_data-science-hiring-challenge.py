import os
print(os.listdir("../input"))
!head "../input/amazon_cells_labelled.txt"
# ! pip install --upgrade pandas
# ! pip install --upgrade nltk
# ! pip install --upgrade scipy numpy scikit-learn
#Q1 soln.

from pathlib import Path

import numpy
import pandas


# dict to store panda's dataframes loaded from txt
dataframes = dict()

# Iterate all labelled files and load them into memory
for labelled_file in Path("../input").glob("*_labelled.txt"):

    # Extract data set name from file name
    source = labelled_file.stem.replace("_labelled", "")
    
    # Read txt files as csv, using tab as separator
    dataframes[source] = pandas.read_csv(labelled_file, sep="\t", header=None)

# Setup training and testing data
x_training = numpy.concatenate((dataframes["imdb"][0].values, dataframes["yelp"][0].values))
y_training = numpy.concatenate((dataframes["imdb"][1].values, dataframes["yelp"][1].values))
x_testing = dataframes["amazon_cells"][0].values
y_testing = dataframes["amazon_cells"][1].values
#Q2 soln.

#Q2a soln.

#from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


tokenizer = TreebankWordTokenizer()

# Establish a pipeline
n_gram_1_to_3_classifier = Pipeline(
    [
        ('tfidf_vectorizer',
         TfidfVectorizer(
             analyzer="word",
             ngram_range=(1, 3),
             tokenizer=tokenizer.tokenize,
         )),
        ('classifier', LinearSVC()),
    ]
)

# Train model with training data
n_gram_1_to_3_classifier.fit(x_training, y_training)
#Q4 soln.

from sklearn.metrics import classification_report


y_prediction = n_gram_1_to_3_classifier.predict(x_testing)
print(classification_report(y_testing, y_prediction))
from operator import itemgetter


model = n_gram_1_to_3_classifier
n = 100


# Extract the vectorizer and the classifier from the pipeline
vectorizer = model.named_steps['tfidf_vectorizer']
classifier = model.named_steps['classifier']

# Zip the feature names with the coefs and sort
coefs = sorted(
    zip(classifier.coef_[0], vectorizer.get_feature_names()),
    key=itemgetter(0), reverse=True
)

# Get the top n and bottom n coef, name pairs
topn  = zip(coefs[:n], coefs[:-(n+1):-1])

# Create the output string to return
output = []

# Create two columns with most negative and most positive features.
for (cp, fnp), (cn, fnn) in topn:
    output.append(
        "{:0.4f}{: >20}         {:0.4f}{: >20}".format(
            cp, fnp, cn, fnn
        )
    )

print("\n".join(output))