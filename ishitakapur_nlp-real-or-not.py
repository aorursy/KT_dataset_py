#References

#Notebook - https://www.kaggle.com/elcaiseri/nlp-the-simplest-way - for processing data

#Notebook - https://www.kaggle.com/kushbhatnagar/disaster-tweets-eda-nlp-classifier-models - as a reference for different types of models in ML

#Notebook - https://www.kaggle.com/philculliton/nlp-getting-started-tutorial - for basic understanding



#import python libraries

import re

import numpy as np

import pandas as pd

from sklearn import feature_extraction, linear_model, model_selection, preprocessing, metrics, naive_bayes, neighbors, tree

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.porter import PorterStemmer

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test_data.head()
train_data.target.value_counts()
x = train_data.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
train_data.shape
train_data[train_data["target"] == 0]["text"].values[1]
train_data[train_data["target"] == 1]["text"].values[1]
train_data.info()
distinct_keyword = train_data['keyword'].value_counts()

distinct_keyword
distinct_location = train_data['location'].value_counts()

distinct_location
# creating bool series True for NaN values for location 

bool_series_location = pd.isnull(train_data['location']) 



# filtering data  

# displaying data only with location = NaN  

train_data[bool_series_location]

print("Number of records with missing location : ",len(train_data[bool_series_location]))



# creating bool series True for NaN values  

bool_series_keyword = pd.isnull(train_data['keyword']) 

# filtering data  

# displaying data only with Keywords = NaN  

train_data[bool_series_keyword]

print("Number of records with missing keywords : ",len(train_data[bool_series_keyword]))

print('{}% of keywords are missing from all records'.format((len(train_data[bool_series_keyword])/len(train_data.index))*100))
#dropping unwanted column 'location'

train_data = train_data.drop(['location'],axis=1)

train_data.head()
#dropping missing 'keyword' records from train data set

train_data = train_data.drop(train_data[bool_series_keyword].index,axis=0)

#Resetting the index after droping the missing records

train_data = train_data.reset_index(drop=True)

print("Number of records after removing missing keyword records : ",len(train_data))

train_data.head()
corpus  = []

pstem = PorterStemmer()

for i in range(train_data['text'].shape[0]):

    #Remove unwanted words

    text = re.sub("[^a-zA-Z]", ' ', train_data['text'][i])

    #Transform words to lowercase

    text = text.lower()

    text = text.split()

    #Remove stopwords then Stemming it

    text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    #Append cleaned tweet to corpus

    corpus.append(text)

    

print("Corpus created")
def remove_punctuation(text):

    '''

        a function for removing punctuation

    '''

    import string

    # replacing the punctuations with no space, 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)
train_data['text'] = train_data['text'].apply(remove_punctuation)

train_data.head(10)
# extracting the stopwords from nltk library

sw = stopwords.words('english')

# displaying the stopwords

np.array(sw);
def stopwords(text):

    '''

        a function for removing the stopword

    '''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)
train_data['text'] = train_data['text'].apply(stopwords)

train_data.head(10)
# create an object of stemming function

stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 
train_data['text'] = train_data['text'].apply(stemming)

train_data.head(10)
count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

sample_train_vectors = count_vectorizer.fit_transform(train_data["text"][0:5])



## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(sample_train_vectors[0].todense().shape)

print(sample_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_data["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_data["text"])
#Create dictionary 

uniqueWords = {}

for text in corpus:

    for word in text.split():

        if(word in uniqueWords.keys()):

            uniqueWords[word] += 1

        else:

            uniqueWords[word] = 1

            

#Convert dictionary to dataFrame

uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])

uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)

print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))

uniqueWords.head(10)
#Get Maximum,Minimum and Mean occurance of a word 

print("Maximum Occurance of a word is {} times".format(uniqueWords['WordFrequency'].max()))

print("Minimum Occurance of a word is {} times".format(uniqueWords['WordFrequency'].min()))

print("Mean Occurance of a word is {} times".format(uniqueWords['WordFrequency'].mean()))
uniqueWords=uniqueWords[uniqueWords['WordFrequency']>=20]

print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = len(uniqueWords))

#Create Bag of Words Model , here X represent bag of words

X = cv.fit_transform(corpus).todense()

y = train_data['target'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=2020)
# Fitting Gaussian Naive Bayes to the Training set

classifier_gaussian = naive_bayes.GaussianNB()

classifier_gaussian.fit(X_train, y_train)

# Predicting the Train data set results

y_pred_gaussian = classifier_gaussian.predict(X_test)

# Making the Confusion Matrix

cm_gaussian = metrics.confusion_matrix(y_test, y_pred_gaussian)

cm_gaussian
print('GaussianNB Classifier Accuracy Score is {} for Train Data Set'.format(classifier_gaussian.score(X_train, y_train)))

print('GaussianNB Classifier Accuracy Score is {} for Test Data Set'.format(classifier_gaussian.score(X_test, y_test)))

print('GaussianNB Classifier F1 Score is {}'.format(metrics.f1_score(y_test, y_pred_gaussian)))
# Fitting K- Nearest neighbour to the Training set

classifier_knn = neighbors.KNeighborsClassifier(n_neighbors = 7,weights = 'distance',algorithm = 'brute')

classifier_knn.fit(X_train, y_train)

# Predicting the Train data set results

y_pred_knn = classifier_knn.predict(X_test)

# Making the Confusion Matrix

cm_knn = metrics.confusion_matrix(y_test, y_pred_knn)

cm_knn
#Calculating Model Accuracy

print('K-Nearest Neighbour Model Accuracy Score for Train Data set is {}'.format(classifier_knn.score(X_train, y_train)))

print('K-Nearest Neighbour Model Accuracy Score for Test Data set is {}'.format(classifier_knn.score(X_test, y_test)))

print('K-Nearest Neighbour Model F1 Score is {}'.format(metrics.f1_score(y_test, y_pred_knn)))
# Fitting multinomial naive bayes Model to the Training set

classifier_multinominal = naive_bayes.MultinomialNB()

classifier_multinominal.fit(X_train, y_train)

# Predicting the Train data set results

y_pred_multinominal = classifier_multinominal.predict(X_test)

# Making the Confusion Matrix

cm_multinominal = metrics.confusion_matrix(y_test, y_pred_multinominal)

cm_multinominal
print('MultinomialNB Model Accuracy Score for Train Data set is {}'.format(classifier_multinominal.score(X_train, y_train)))

print('MultinomialNB Model Accuracy Score for Test Data set is {}'.format(classifier_multinominal.score(X_test, y_test)))

print('MultinomialNB Model F1 Score is {}'.format(metrics.f1_score(y_test, y_pred_multinominal)))
classifier_log_regression = linear_model.LogisticRegression()

classifier_log_regression.fit(X_train, y_train)

# Predicting the Train data set results

y_pred_log_regression = classifier_log_regression.predict(X_test)

# Making the Confusion Matrix

cm_log_regression = metrics.confusion_matrix(y_test, y_pred_log_regression)

cm_log_regression
print('Logistic Regression Model Accuracy Score for Train Data set is {}'.format(classifier_log_regression.score(X_train, y_train)))

print('Logistic Regression Model Accuracy Score for Test Data set is {}'.format(classifier_log_regression.score(X_test, y_test)))

print('Logistic Regression Model F1 Score is {}'.format(metrics.f1_score(y_test, y_pred_log_regression)))
classifier_ridge = linear_model.RidgeClassifier()

classifier_ridge.fit(X_train, y_train)

# Predicting the Train data set results

y_pred_ridge = classifier_ridge.predict(X_test)

# Making the Confusion Matrix

cm_ridge = metrics.confusion_matrix(y_test, y_pred_ridge)

cm_ridge
#Calculating Model Accuracy

print('Ridge Classifier Model Accuracy Score for Train Data set is {}'.format(classifier_ridge.score(X_train, y_train)))

print('Ridge Classifier Model Accuracy Score for Test Data set is {}'.format(classifier_ridge.score(X_test, y_test)))

print('Ridge Classifier Model F1 Score is {}'.format(metrics.f1_score(y_test, y_pred_ridge)))
#Check number of records in Test Data set

print("Number of records present in Test Data Set are {}".format(len(test_data.index)))

#Check number of missing Keywords in Test Data set

print("Number of records without keywords in Test Data are {}".format(len(test_data[pd.isnull(test_data['keyword'])])))

print("Number of records without location in Test Data are {}".format(len(test_data[pd.isnull(test_data['location'])])))
#Drop Location column from Test Data

test_data = test_data.drop(['location'],axis=1)

test_data.head()
X_TestSet = cv.transform(test_data['text']).todense()
y_test_pred_gaussian = classifier_gaussian.predict(X_TestSet)

y_test_pred_knn = classifier_knn.predict(X_TestSet)

y_test_pred_multinominal = classifier_multinominal.predict(X_TestSet)

y_test_pred_log_regression = classifier_log_regression.predict(X_TestSet)

y_test_pred_ridge = classifier_ridge.predict(X_TestSet)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

output = pd.DataFrame({'id': test_data.id, 'target': y_test_pred_multinominal})

output.to_csv('sample_submission.csv', index=False)

sample_submission.head()