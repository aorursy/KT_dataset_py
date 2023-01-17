# Importing essential libraries

import numpy as np

import pandas as pd
# Loading the dataset

df = pd.read_csv('../input/movietr/kaggle_movie_train.csv')
df.columns
df.shape
df.head(10)
# Importing essential libraries for visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Visualizing the count of 'genre' column from the dataset

plt.figure(figsize=(12,12))

sns.countplot(x='genre', data=df)

plt.xlabel('Movie Genres')

plt.ylabel('Count')

plt.title('Genre Plot')

plt.show()
# Finding unique genres

movie_genre = list(df['genre'].unique())

movie_genre.sort()

movie_genre
# Mapping the genres to values

genre_mapper = {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}

df['genre'] = df['genre'].map(genre_mapper)

df.head(10)
# Finding any NaN values

df.isna().any()
# Removing the 'id' column

df.drop('id', axis=1, inplace=True)

df.columns
# Importing essential libraries for performing Natural Language Processing on given dataset

import nltk

import re

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
df.shape
# Cleaning the text

corpus = []

ps = PorterStemmer()



for i in range(0, df.shape[0]):



  # Cleaning special character from the dialog/script

  dialog = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['text'][i])



  # Converting the entire dialog/script into lower case

  dialog = dialog.lower()



  # Tokenizing the dialog/script by words

  words = dialog.split()



  # Removing the stop words

  dialog_words = [word for word in words if word not in set(stopwords.words('english'))]



  # Stemming the words

  words = [ps.stem(word) for word in dialog_words]



  # Joining the stemmed words

  dialog = ' '.join(words)



  # Creating a corpus

  corpus.append(dialog)
corpus[0:10]
df[df['genre']==4].index
len(corpus)
drama_words = []

for i in list(df[df['genre']==4].index):

  drama_words.append(corpus[i])



action_words = []

for i in list(df[df['genre']==1].index):

  action_words.append(corpus[i])



comedy_words = []

for i in list(df[df['genre']==3].index):

  comedy_words.append(corpus[i])



drama = ''

action = ''

comedy = ''

for i in range(0, 3):

  drama += drama_words[i]

  action += action_words[i]

  comedy += comedy_words[i]
# Creating wordcloud for drama genre

from wordcloud import WordCloud

wordcloud1 = WordCloud(background_color='white', width=3000, height=2500).generate(drama)

plt.figure(figsize=(8,8))

plt.imshow(wordcloud1)

plt.axis('off')

plt.title("Words which indicate 'DRAMA' genre ")

plt.show()
# Creating wordcloud for action genre

wordcloud2 = WordCloud(background_color='white', width=3000, height=2500).generate(action)

plt.figure(figsize=(8,8))

plt.imshow(wordcloud2)

plt.axis('off')

plt.title("Words which indicate 'ACTION' genre ")

plt.show()
# Creating wordcloud for comedy genre

wordcloud3 = WordCloud(background_color='white', width=3000, height=2500).generate(comedy)

plt.figure(figsize=(8,8))

plt.imshow(wordcloud3)

plt.axis('off')

plt.title("Words which indicate 'COMEDY' genre ")

plt.show()
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=10000, ngram_range=(1,2))

X = cv.fit_transform(corpus).toarray()
y = df['genre'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train, y_train)
# Predicting the Test set results

nb_y_pred = nb_classifier.predict(X_test)
# Calculating Accuracy

from sklearn.metrics import accuracy_score

score1 = accuracy_score(y_test, nb_y_pred)

print("---- Score ----")

print("Accuracy score is: {}%".format(round(score1*100,2)))
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

nb_cm = confusion_matrix(y_test, nb_y_pred)
nb_cm
# Plotting the confusion matrix

plt.figure(figsize=(15,12))

axis_labels = ['other', 'action', 'adventure', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller']

sns.heatmap(data=nb_cm, annot=True, cmap="Blues", xticklabels=axis_labels, yticklabels=axis_labels)

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.title('Confusion Matrix for Multinomial Naive Bayes Algorithm')

plt.show()
# Hyperparameter tuning the Naive Bayes Classifier

best_accuracy = 0.0

alpha_val = 0.0

for i in np.arange(0.1,1.1,0.1):

  temp_classifier = MultinomialNB(alpha=i)

  temp_classifier.fit(X_train, y_train)

  temp_y_pred = temp_classifier.predict(X_test)

  score = accuracy_score(y_test, temp_y_pred)

  print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))

  if score>best_accuracy:

    best_accuracy = score

    alpha_val = i

print('--------------------------------------------')

print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))
classifier = MultinomialNB(alpha=0.1)

classifier.fit(X_train, y_train)
def genre_prediction(sample_script):

  sample_script = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_script)

  sample_script = sample_script.lower()

  sample_script_words = sample_script.split()

  sample_script_words = [word for word in sample_script_words if not word in set(stopwords.words('english'))]

  ps = PorterStemmer()

  final_script = [ps.stem(word) for word in sample_script_words]

  final_script = ' '.join(final_script)



  temp = cv.transform([final_script]).toarray()

  return classifier.predict(temp)[0]
# For generating random integer

from random import randint
# Loading test dataset

test = pd.read_csv('../input/movier/kaggle_movie_test.csv')

test.columns
test.shape
test.drop('id', axis=1, inplace=True)

test.head(10)
# Predicting values

row = randint(0,test.shape[0]-1)

sample_script = test.text[row]



print('Script: {}'.format(sample_script))

value = genre_prediction(sample_script)

print('Prediction: {}'.format(list(genre_mapper.keys())[value]))
# Predicting values

row = randint(0,test.shape[0]-1)

sample_script = test.text[row]



print('Script: {}'.format(sample_script))

value = genre_prediction(sample_script)

print('Prediction: {}'.format(list(genre_mapper.keys())[value]))
# Predicting values

row = randint(0,test.shape[0]-1)

sample_script = test.text[row]



print('Script: {}'.format(sample_script))

value = genre_prediction(sample_script)

print('Prediction: {}'.format(list(genre_mapper.keys())[value]))
# Predicting values

row = randint(0,test.shape[0]-1)

sample_script = test.text[row]



print('Script: {}'.format(sample_script))

value = genre_prediction(sample_script)

print('Prediction: {}'.format(list(genre_mapper.keys())[value]))
# Predicting values

row = randint(0,test.shape[0]-1)

sample_script = test.text[row]



print('Script: {}'.format(sample_script))

value = genre_prediction(sample_script)

print('Prediction: {}'.format(list(genre_mapper.keys())[value]))