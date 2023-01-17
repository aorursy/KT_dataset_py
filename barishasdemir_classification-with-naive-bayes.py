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
# Fundamentals

import matplotlib.pyplot as plt

import seaborn as sns



# Import NLTK to use its functionalities on texts

"""DO NOT forget to download followings if you do not have

# nltk.download('punkt')

#nltk.download('wordnet')

"""

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



# We will visualize the messages with a word cloud

from wordcloud import WordCloud



# Multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB



# Import Tf-idf Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# Import the Label Encoder

from sklearn.preprocessing import LabelEncoder



# Import the train test split

from sklearn.model_selection import train_test_split



# To evaluate our model

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score



# I will keep the resulting plots

%matplotlib inline



# Enable Jupyter Notebook's intellisense

%config IPCompleter.greedy=True
# Load the data

data = pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
# Display firt five rows

display(data.head())



# Display the summary statistics

display(data.describe())



# Print the info

print(data.info())
# Print the counts of each category

print(data['Category'].value_counts())



print()



# Print the proportions of each category

print(data['Category'].value_counts(normalize=True))



# Visualize the Categories

sns.countplot(data['Category'])

plt.title("Category Counts")

plt.show()
# Initialize the Label Encoder.

le = LabelEncoder()



# Encode the categories

data['Category_enc'] = le.fit_transform(data['Category'])



# Display the first five rows again to see the result

display(data.head())



# Print the datatypes

print(data.dtypes)
# Store the number of words in each messages

data['word_count'] = data['Message'].str.split().str.len()



# Print the average number of words in each category

print(data.groupby('Category')['word_count'].mean())



# Visualize the distribution of word counts in each category

sns.distplot(data[data['Category']=='spam']['word_count'], label='Spam')

sns.distplot(data[data['Category']=='ham']['word_count'], label='Ham'),

plt.legend()

plt.show()
# Make the letters lower case and tokenize the words

tokenized_messages = data['Message'].str.lower().apply(word_tokenize)



# Print the tokens to see how it looks like

print(tokenized_messages)
# Define a function to returns only alphanumeric tokens

def alpha(tokens):

    """This function removes all non-alphanumeric characters"""

    alpha = []

    for token in tokens:

        if str.isalpha(token) or token in ['n\'t','won\'t']:

            if token=='n\'t':

                alpha.append('not')

                continue

            elif token == 'won\'t':

                alpha.append('wont')

                continue

            alpha.append(token)

    return alpha



# Apply our function to tokens

tokenized_messages = tokenized_messages.apply(alpha)



print(tokenized_messages)
# Define a function to remove stop words

def remove_stop_words(tokens):

    """This function removes all stop words in terms of nltk stopwords"""

    no_stop = []

    for token in tokens:

        if token not in stopwords.words('english'):

            no_stop.append(token)

    return no_stop



# Apply our function to tokens

tokenized_messages = tokenized_messages.apply(remove_stop_words)



print(tokenized_messages)
# Define a function to lemmatization

def lemmatize(tokens):

    """This function lemmatize the messages"""

    # Initialize the WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    # Create the lemmatized list

    lemmatized = []

    for token in tokens:

            # Lemmatize and append

            lemmatized.append(lemmatizer.lemmatize(token))

    return " ".join(lemmatized)



# Apply our function to tokens

tokenized_messages = tokenized_messages.apply(lemmatize)



print(tokenized_messages)
# Replace the columns with tokenized messages

data['Message'] = tokenized_messages



# Display the first five rows

display(data.head())
# Get the spam messages

spam = data[data['Category']=='spam']['Message'].str.cat(sep=', ')



# Get the ham messages

ham = data[data['Category']=='ham']['Message'].str.cat(sep=', ')



# Initialize the word cloud

wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color ='white')



# Generate the world clouds for each type of message

spam_wc = wc.generate(spam)



# plot the world cloud for spam                     

plt.figure(figsize = (5, 5), facecolor = None) 

plt.imshow(spam_wc) 

plt.axis("off") 

plt.title("Common words in spam messages")

plt.tight_layout(pad = 0) 

plt.show() 

ham_wc = wc.generate(ham)



# plot the world cloud for spam                       

plt.figure(figsize = (5, 5), facecolor = None) 

plt.imshow(ham_wc) 

plt.axis("off")

plt.title("Common words in ham messages")

plt.tight_layout(pad = 0) 

plt.show() 
# Select the features and the target

X = data['Message']

y = data['Category_enc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)
# Create the tf-idf vectorizer

vectorizer = TfidfVectorizer(strip_accents='ascii')



# First fit the vectorizer with our training set

tfidf_train = vectorizer.fit_transform(X_train)



# Now we can fit our test data with the same vectorizer

tfidf_test = vectorizer.transform(X_test)
# Initialize the Multinomial Naive Bayes classifier

nb = MultinomialNB()



# Fit the model

nb.fit(tfidf_train, y_train)



# Print the accuracy score

print("Accuracy:",nb.score(tfidf_test, y_test))
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





# Print the Receiver operating characteristic Auc score

auc_score = roc_auc_score(y_test, y_pred)

print("\nROC AUC Score:",auc_score)



# Get probabilities.

y_pred_proba = nb.predict(tfidf_test)



# Get False Positive rate, True Positive rate and the threshold

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



# Visualize the ROC curve.

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FP Rate')

plt.ylabel('TP Rate')

plt.title('ROC')

plt.show()