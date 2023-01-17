import spacy
from spacy import displacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import LinearSVC
import string
# Importing Dataset from the Github Repository.
!git clone https://github.com/laxmimerit/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset
# Loading Spacy small model as nlp
nlp = spacy.load("en_core_web_sm")
# Gathering all the stopwords
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
print(len(stopwords))
# Loading yelp dataset
data_yelp = pd.read_csv('../working/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset/datasets/yelp_labelled.txt',
                        sep='\t', header= None)
data_yelp.head()
# Adding column names to the dataframe
columnName = ['Review','Sentiment']
data_yelp.columns = columnName
data_yelp.head()
print(data_yelp.shape)
# Adding Amazon dataset and adding its column name
data_amz = pd.read_csv("../working/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset/datasets/amazon_cells_labelled.txt",
                        sep='\t', header= None)
data_amz.columns = columnName
data_amz.head()
print(data_amz.shape)
# Adding IMdB dataset and adding its column name
data_imdb = pd.read_csv("../working/NLP-Tutorial-8---Sentiment-Classification-using-SpaCy-for-IMDB-and-Amazon-Review-Dataset/datasets/imdb_labelled.txt",
                        sep='\t', header= None)
data_imdb.columns = columnName
data_imdb.head()
print(data_imdb.shape)
# Merging all the three dataframes
data = data_yelp.append([data_amz, data_imdb], ignore_index=True)
print(data.shape)
# Sentiment ditribution in the dataset
data.Sentiment.value_counts()
# Getting information regarding the null entries in the dataset
data.isnull().sum()
punct = string.punctuation
print(punct)
def dataCleaning(sentence):
  doc = nlp(sentence)
  tokens = []
  for token in doc:
    if token.lemma_ != '-PRON-':
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
  clean_tokens = []
  for token in tokens:
    if token not in punct and token not in stopwords:
      clean_tokens.append(token)
  return clean_tokens
dataCleaning("Today we are having heavy rainfall, We recommend you to stay at your home and be safe, Do not start running here and there")
# All the useful words are returned, no punctuations no stop words and in the lemmatized form
# Spillting the train and test data
X = data['Review']
y = data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train.shape,y_test.shape)
# Creating the model and pipeline
tfidf = TfidfVectorizer(tokenizer = dataCleaning)
svm = LinearSVC()
steps = [('tfidf',tfidf),('svm',svm)]
pipe = Pipeline(steps)
# Training the model
pipe.fit(X_train,y_train)
# Testing on the test dataset
y_pred = pipe.predict(X_test)
# Printing the classification report and the confusion matrix
print(classification_report(y_test,y_pred))
print("\n\n")
print(confusion_matrix(y_test,y_pred))
# Testing on random inputs
pipe.predict(["Wow you are an amazing person"])
pipe.predict(["you suck"])