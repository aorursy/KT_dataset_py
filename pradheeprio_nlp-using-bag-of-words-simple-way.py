from IPython.display import Image



Image("../input/twitterimg/images.png", width = "400px")
import pandas as pd

import numpy as np

import seaborn as sns

import nltk

nltk.download('stopwords')

nltk.download('wordnet')

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
df=pd.read_csv("../input/nlp-getting-started/train.csv")



print(df.head())
df.shape
print(df.isnull().sum())
sns.countplot('target',data=df)
df["keyword"].value_counts()
data=df.drop(['location','keyword'],axis=1)

data.head()
# Cleaning the reviews



corpus = []

for i in range(0,7613):



  # Cleaning special character from the tweets

  review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=data['text'][i])#remove everything apart from capital A to Z and small a to z

  



  # Converting the entire tweets into lower case

  tweets = review.lower()



  # Tokenizing the tweetsby words

  tweets_words = tweets.split()

 

  # Removing the stop words

  tweets_words = [word for word in tweets_words if not word in set(stopwords.words('english'))]

  

  # lemmitizing  the words

  lemmatizer = WordNetLemmatizer()

  tweets= [lemmatizer.lemmatize(word) for word in tweets_words]



  # Joining the lemmitized words

  tweets = ' '.join(tweets)

  

  # Creating a corpus

  corpus.append(tweets)
corpus[:5]
cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()

y = data['target']

print(X.shape)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)





# Predicting the Test set results

y_pred = classifier.predict(X_test)

print(y_pred)
accuracy=confusion_matrix(y_test,y_pred )

print("confusion_matrix:",accuracy)

accuracy=accuracy_score(y_test,y_pred )

print("accuracy_score:",accuracy)



print(classification_report(y_test,y_pred ))
data_test=pd.read_csv("../input/nlp-getting-started/test.csv")

print(data_test.head())
data_test.isnull().sum()

data_tes=data_test.drop(['keyword','location'],axis=1)

data_tes.shape



corpus1 = []

for i in range(0,3263):





  # Cleaning special character from the tweets

  review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=data_tes['text'][i])

  



  # Converting the entire tweets into lower case

  tweets = review.lower()



  # Tokenizing the tweets by words

  tweets_words = review.split()

 

  # Removing the stop words

  tweets_words = [word for word in tweets_words if not word in set(stopwords.words('english'))]

  

  # lemmitizing the words

  lemmatizer = WordNetLemmatizer()

  tweets = [lemmatizer.lemmatize(word) for word in tweets_words]



  # Joining the lemmitized words

  tweets = ' '.join(tweets)



  y_pred=cv.transform([review]).toarray()

  pre=classifier.predict(y_pred)

  corpus1.append(pre)



print(len(corpus1))



# Create a submisison dataframe and append the relevant columns

submission = pd.DataFrame()



submission['id'] = data_tes['id']

submission['target'] = corpus1
# Let's convert our submission dataframe 'Survived' column to ints

submission['target'] = submission['target'].astype(int)

print('Converted Survived column to integers.')



print(submission.head())









# Are our test and submission dataframes the same length?

if len(submission) == len(data_tes):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")



# Convert submisison dataframe to csv for submission to csv 

# for Kaggle submisison

submission.to_csv('../submission_nlp1.csv', index=False)

print('Submission CSV is ready!')