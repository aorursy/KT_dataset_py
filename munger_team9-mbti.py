# Import the necessary packages

import numpy as np

import pandas as pd

import nltk

from nltk.corpus import stopwords

from nltk.corpus import wordnet

import re

import string

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import NearestCentroid

from sklearn.metrics import accuracy_score

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

from sklearn.metrics import classification_report
# Import the training and test data sets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Join the training and test data sets into one data set

df = pd.concat([train, test], sort=True, ignore_index=True)
#Printing information of all columns 

df.info()
df.isnull().sum()
# Countplot of the 16 personality types in the dataset

dims1 = (15.0, 4.0)

fig, ax = plt.subplots(figsize=dims1)

coolwarm = sns.color_palette("coolwarm", 16)

sns.set_palette(coolwarm)

sns.countplot(x="type", data=df, \

              order=["ENFJ","ENFP","ENTJ","ENTP","ESFJ","ESFP","ESTJ","ESTP",\

                     "INFJ","INFP","INTJ","INTP","ISFJ","ISFP","ISTJ","ISTP"])

plt.title("Brief Personality Types", fontsize=16)

plt.xlabel('Personality Types ')

plt.ylabel('Frequency')

plt.xticks(fontsize=12)

plt.yticks(fontsize=12);
# Store the the URL pattern used for any subsequent regex

url_pattern = 'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

# Loop through every set of posts (every row)

for i in range(len(df)):

    # Replace the '|||' separator with a whitespace character

    df.loc[i, 'posts'] = df.loc[i, 'posts'].replace('|||', ' ')

    # Replace any URL with 'website'

    for url in re.findall(url_pattern, df.loc[i, 'posts']):

        df.loc[i, 'posts'] = df.loc[i, 'posts'].replace(url, 'website')
# Print to console any posts that contain the string 'http'

rows = []

for i, row in df.iterrows():

    if 'http' in row['posts']:

        print('Row ', i, ' still contains \'http\'.')

        # Identify the index where the first 'http' instance is located

        a = row['posts'].index('http')

        # Only print part of the posts from the start of the abovementioned index

        print(row['posts'][a:a+100])

        print('_______________')

        rows = rows+[i]
# Construct a list containing all the problematic URLs

urls = ['http:youtube.com/watch?v=KAveNvDnL_A', 'http://www.youtube.com/watch?v=u2pu0m9iTo4website', 'https...',

        'http://website', 'https', 'http:// .....', 'http://website',

        'http: //personalitycafe.com/nfs-temperament-forum-dreamers/1054754-fed-up-infp.',

        'http//www.youtube.com/watch?v=UqhXBSW0Zns', 'http[colon]//www[dot]youtube[dot]com/watch?v=kbBiCiLmaZs',

        'https', 'http%3A%2F%2Fs1211.photobucket.com%2Falbums%2Fcc433%2FWAR0808%2F%3Faction%3Dview%26curr...']

# Replace these URLs with 'website'

for i in range(len(urls)):

    row = rows[i]

    url = urls[i]

    df.loc[row, 'posts'] = df.loc[row, 'posts'].replace(url, 'website')

# Print to console any remaining posts that stil contain the string 'http'    

for i, row in df.iterrows():

    if 'http' in row['posts']:

        print('Row ', i, ' still contains \'http\'.')

        a = row['posts'].index('http')

        print(row['posts'][a:a+100])

        print('_______________')
#counting number of  unique words posted online 

cleandata = df['type'].value_counts()

cleandata.head()
plt.hlines(y=list(range(16)), xmin=0, xmax=cleandata, color='skyblue')

plt.plot(cleandata, list(range(16)), "D")

# plt.stem(cleandata)

plt.yticks(list(range(16)), cleandata.index)

plt.ylabel('16 Different Personality Types')

plt.xlabel('Count Of Unique Words ')

plt.title("Distribution Of Number Of Posts", fontsize=16)

plt.show()
# Create a function that removes all punctuation in a post

def remove_punctuation(post):

    return ''.join([l for l in post if l not in string.punctuation])
# Use the above-created function to remove all punctuation from the posts

df['posts'] = df['posts'].apply(remove_punctuation)
# this code was taken from (https://www.kaggle.com/phuongpm/mbti-prediction)

def generate_wordcloud(text, title):

    # Create and generate a word cloud image:

    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(title, fontsize = 40)

    plt.show()

df_by_personality = df.groupby("type")['posts'].apply(' '.join).reset_index()
for i, t in enumerate(df_by_personality['type']):

    text = df_by_personality.iloc[i,1]

    generate_wordcloud(text, t)
# Convert the posts to a matrix of token counts, so that the data set can be used in subsequent models

vectorizer = CountVectorizer(stop_words='english',

                             min_df=2, # Ignore words that appear twice or fewer times in all posts 

                             max_df=0.5, # Ignore words that appear in more than 50% of all posts

                             ngram_range=(1, 1)) # Do not construct n-grams, i.e. only use single words as tokens

# Change the count vectorizer to an array for easier implementation in subsequent models

X = vectorizer.fit_transform(df['posts']).toarray()

# Re-construct the training and test sets

X_train = X[0:len(train)]

X_test = X[len(train):]

y_train = list(train['type'])

# List of MBTI types for later classification reports

type_labels = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ',

               'ISTP', 'ISFP', 'INFP', 'INTP',

               'ESTP', 'ESFP', 'ENFP', 'ENTP',

               'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
def submission(lst1, filename):

    # Copy the input list of personality types so that the original list is not modified

    lst = lst1.copy()

    # Create a dictionary of personality-type encoding

    personalityNumbers = {'I': 0, 'E': 1, 'S': 0, 'N': 1, 'F': 0, 'T': 1, 'P': 0, 'J': 1}

    # Encode the personality types

    for i in range(len(lst)):

        types = []

        for j in lst[i]:

            j = personalityNumbers[j]

            types = types + [j]

        lst[i] = types

    # Change the above-formed array into a dataframe for submission onto Kaggle

    yhat = pd.DataFrame(lst)

    yhat.columns = ['mind', 'energy', 'nature', 'tactics']

    yhat['id'] = test['id']

    yhat = yhat[['id', 'mind', 'energy', 'nature', 'tactics']]

    # Save the encoded personality types to a CSV file for subsequent uploading onto Kaggle

    return yhat.to_csv(filename, index=False)
#### Initialize a k-nearest centroid classifier

knc = NearestCentroid()

#### Train the classifer

knc.fit(X_train, y_train)

#### Print the classification report of the trained classifer on its own test data

print(classification_report(y_train, knc.predict(X_train), target_names=type_labels))

#### Predict the personality types of Kaggle's test dats set

predictions = list(knc.predict(X_test))

#### Encode the predicted personality types for submission onto Kaggle, and write the submission to a CSV file

submission(predictions, 'submission_nearest_centroid.csv')
#### Initialize a naive bayes classifier

nb = MultinomialNB(alpha=0.6)

#### Train the classifer

nb.fit(X_train, y_train)

#### Print the classification report of the trained classifer on its own test data

print(classification_report(y_train, nb.predict(X_train), target_names=type_labels))

#### Predict the personality types of Kaggle's test dats set

predictions = list(nb.predict(X_test))

#### Encode the predicted personality types for submission onto Kaggle, and write the submission to a CSV file

submission(predictions, 'submission_naive_bayes.csv')
#### Initialize a logistic regression classifier

lr = LogisticRegression(n_jobs=1, C=100, solver='newton-cg', multi_class='auto', penalty='l2', max_iter=100)

#### Train the classifer

lr.fit(X_train,y_train)

#### Print the classification report of the trained classifer on its own test data

print(classification_report(y_train, lr.predict(X_train), target_names=type_labels))

#### Predict the personality types of Kaggle's test dats set

predictions = list(lr.predict(X_test))

#### Encode the predicted personality types for submission onto Kaggle, and write the submission to a CSV file

submission(predictions, 'submission_log_regression.csv')
#### Initialize a SVM classifier

svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=50, tol=1e-3)

#### Train the classifer

svm.fit(X_train, y_train)

#### Print the classification report of the trained classifer on its own test data

print(classification_report(y_train, svm.predict(X_train), target_names=type_labels))

#### Predict the personality types of Kaggle's test dats set

predictions = list(svm.predict(X_test))

#### Encode the predicted personality types for submission onto Kaggle, and write the submission to a CSV file

submission(predictions, 'submission_svm.csv')