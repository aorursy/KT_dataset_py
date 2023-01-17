import numpy as np # For Linear Algebra

import pandas as pd # For I/O, Data Transformation

import os # For Elementary OS operations

from sklearn.metrics import precision_recall_fscore_support as score # For Evaluating the Model

from sklearn.metrics import accuracy_score as acs # For Evaluating the Model

import matplotlib.pyplot as plt # For Plotting

import seaborn as sns # For Plotting

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Creating Individual Data Frames

fakedataset = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv") # Make a DataFrame for Fake News

realdataset = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv") # Make a DataFrame for Real News



## Reducing Corpus Size

fakedataset = fakedataset[:5000] 

realdataset = realdataset[:5000] 



## Adding Classes

realdataset["class"] = 1 # Adding Class to Real News

fakedataset["class"] = 0 # Adding Class to Fake News



## Concatenating 'Title' and 'Text' together

realdataset["text"] = realdataset["title"] + " " + realdataset["text"] # Concatenating Text and Title into a single column for Real News DataFrame

fakedataset["text"] = fakedataset["title"] + " " + fakedataset["text"] # Concatenating Text and Title into a single column for Fake News DataFrame



## Removing Redundant Columns

realdataset = realdataset.drop(["subject", "date", "title"], axis = 1) # Removing Redundant features from Real News DataFrame

fakedataset = fakedataset.drop(["subject", "date", "title"], axis = 1) # Removing Redundant features from Fake News DataFrame



## Appending the DataFrames together

dataset = realdataset.append(fakedataset, ignore_index = True) # Making a Single DataFrame 



## Deleting the other dataframes

del realdataset, fakedataset 
dataset.head()
import nltk



nltk.download("stopwords")

nltk.download("punkt")
## Import Statements

import re

import string

stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.PorterStemmer()



## Functions, which counts number of punctuations

def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100





## Adding 'Body Length' and 'Punct' to the dataframe

dataset['body_len'] = dataset['text'].apply(lambda x: len(x) - x.count(" "))

dataset['punct%'] = dataset['text'].apply(lambda x: count_punct(x))



## Cleaning the Text from Punctuations and Stopwords

def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = [stemmer.stem(word) for word in tokens if word not in stopwords]

    return text
dataset.head()
## Import Statements

from sklearn.model_selection import train_test_split



## Choosing Columns from dataset as X(Input) and y(Label)

X=dataset[['text', 'body_len', 'punct%']]

y=dataset['class']



## Using train_test_split to split our dataset

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
## Import Statements

from sklearn.feature_extraction.text import TfidfVectorizer



## Declaring a Vectoriser

tfidf_vect = TfidfVectorizer(analyzer=clean_text)



## 'Fitting' the Vectoriser

tfidf_vect_fit = tfidf_vect.fit(X_train['text'])



## Creating 'Test' and 'Train' vectorised dataframes

tfidf_train = tfidf_vect_fit.transform(X_train['text'])

tfidf_test = tfidf_vect_fit.transform(X_test['text'])



## Adding 'Body_Len' and 'punct' columns to the front

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_train.toarray())], axis=1)

X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_test.toarray())], axis=1)



## Checking, if we did everything alright

X_train_vect.head()
## Import Statements

from sklearn.tree import DecisionTreeClassifier



## Instantiating a DecisionTreeClassifier

clf = DecisionTreeClassifier()



## 'Training' the Classifier

clf = clf.fit(X_train_vect,y_train)



## Predicting with the Classifier

y_pred = clf.predict(X_test_vect)



## Evaluating the Model against it's predictions

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label=1, average='binary')

print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(

    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_test,y_pred), 3)))



## Making the Confusion Matrix



### Import Statements

from sklearn.metrics import confusion_matrix



### Creating a confusion_matrix instance 

cm = confusion_matrix(y_test, y_pred)



### Making a Dataframe, of the metrics with classes

class_label = [0, 1]

df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)



### Plotting the Model

sns.heatmap(df_cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()