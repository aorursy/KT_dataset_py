# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import libraries

import matplotlib.pyplot as plt

import seaborn as sns
# We want to see whole content (non-truncated)

pd.set_option('display.max_colwidth', None)



# Load the train and the test set

train = pd.read_csv("/kaggle/input/turkish-sentiment-analysis-data-beyazperdecom/train.csv", encoding= 'unicode_escape')

test = pd.read_csv("/kaggle/input/turkish-sentiment-analysis-data-beyazperdecom/test.csv", encoding= 'unicode_escape')



# Display first five rows

display(train.head())



# Descriptive Statistics

print(train.describe())



# Info

print(train.info())
# We do not need the Unnamed: 0 column

train.drop("Unnamed: 0", axis=1, inplace=True)



# Info

print(train.info())
# Get lengths

train["length"] = train["comment"].str.len()



# Get word counts

train["word_count"] = train["comment"].str.split().apply(len)



# Display the two columns

display(train[["length","word_count"]])



# Look at the distribution of length and word_count

sns.distplot(train["length"], bins=10)

plt.title("Distribution of comment lengths")

plt.show()



sns.distplot(train["word_count"], bins=10)

plt.title("Distribution of word counts")

plt.show()





# Print 10 bins of length column

print(pd.cut(train['length'], 10).value_counts())



# Print 10 bins of word_count column

print(pd.cut(train['word_count'], 10).value_counts())

# Print binned lengths by label

print(train['Label'].groupby(pd.cut(train['length'], 10)).mean())





# Print binned lengths by label

print(train['Label'].groupby(pd.cut(train['length'], 10)).mean())
# Import necessary tools from nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



# Import Snowballstemmer for stemming Turkish words

from snowballstemmer import TurkishStemmer



# Initialize the stemmer

tr_stemmer =TurkishStemmer()



def tokenize(comment):

    

    # Tokenize the comment

    tokenized = word_tokenize(comment)

    

    # Remove the stopwords (in Turkish)

    tokenized = [token for token in tokenized if token not in stopwords.words("turkish")]

    

    # Stemming the tokens

    tokenized = [tr_stemmer.stemWord(token) for token in tokenized]

    

    # Remove non-alphabetic characters

    tokenized = [token for token in tokenized if token.isalpha()]

    

    return tokenized



# Apply the function

train["Tokenized"] = train["comment"].str.lower().apply(tokenize)

test["Tokenized"] = test["comment"].str.lower().apply(tokenize)



# See the result

display(train["Tokenized"].head())

# Import TfidfVectorizer from sklearn

from sklearn.feature_extraction.text import TfidfVectorizer



# Convert tokenized words from list to string

train['tokenized_str']=[" ".join(token) for token in train['Tokenized'].values]



# Initialize a Tf-idf Vectorizer

vectorizer = TfidfVectorizer()



# Fit and transform the vectorizer

tfidf_matrix = vectorizer.fit_transform(train["tokenized_str"])



# Let's see what we have

display(tfidf_matrix)



# Create a DataFrame for tf-idf vectors and display the first five rows

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns= vectorizer.get_feature_names())



# Display the first five rows of the tfidf DataFrame

display(tfidf_df.head())
# Import wordcloud

from wordcloud import WordCloud



# Create a new DataFrame called frequencies

frequencies = pd.DataFrame(tfidf_matrix.sum(axis=0).T,index=vectorizer.get_feature_names(),columns=['total frequency'])



# Sort the words by frequency

frequencies.sort_values(by='total frequency',ascending=False, inplace=True)



# Display the most 20 frequent words

display(frequencies.head(20))



# Join the indexes

frequent_words = " ".join(frequencies.index)+" "



# Initialize the word cloud

wc = WordCloud(width = 500, height = 500, min_font_size = 10, max_words=2000, background_color ='white')



# Generate a world cloud

wc_general = wc.generate(frequent_words)



# Plot the world cloud                     

plt.figure(figsize = (10, 10), facecolor = None) 

plt.imshow(wc_general, interpolation="bilinear") 

plt.axis("off") 

plt.title("Common words in the comments")

plt.tight_layout(pad = 0) 

plt.show()
# Import necessary tools from sklearn

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



# Select the features and the target

X = train['tokenized_str']

y = train["Label"]



# Split the train set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)
# Create the tf-idf vectorizer

model_vectorizer = TfidfVectorizer()



# Fit and transform the vectorizer with X_train

tfidf_train = model_vectorizer.fit_transform(X_train)



# Tranform the vectorizer with X_test

tfidf_test = model_vectorizer.transform(X_test)



# Initialize the Bernoulli Naive Bayes classifier

nb = BernoulliNB()



# Fit the model

nb.fit(tfidf_train, y_train)



# Print the accuracy score

best_accuracy = cross_val_score(nb, tfidf_test, y_test, cv=10, scoring='accuracy').max()

print("Accuracy:",best_accuracy)



# Predict the labels

y_pred = nb.predict(tfidf_test)



# Print the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix\n")

print(cm)



# Print the Classification Report

cr = classification_report(y_test, y_pred)

print("\n\nClassification Report\n")

print(cr)

# Convert tokenized words from list to string

test['tokenized_str']=[" ".join(token) for token in test['Tokenized'].values]



# Look at the test data

display(test.head())

print(test.info())
# Get the tfidf of test data

tfidf_final = model_vectorizer.transform(test["tokenized_str"])



# Predict the labels

y_pred_final = nb.predict(tfidf_final)



# Print the Confusion Matrix

cm = confusion_matrix(test["Label"], y_pred_final)

print("Confusion Matrix\n")

print(cm)



# Print the Classification Report

cr = classification_report(test["Label"], y_pred_final)

print("\n\nClassification Report\n")

print(cr)
