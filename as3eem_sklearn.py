import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# load library

from sklearn.datasets import load_iris



#import IRIS dataset

iris = load_iris()
# Describe the data

print(iris.DESCR)
x = iris.data

y = iris.target

features = iris.feature_names

target = iris.target_names



print("Feature Names:",features)

print("-"*100)

print("Target Names:", target)

print("-"*100)

print("data:", x[:10])

print("-"*100)
# checking shape of dataset before spliting

print(x.shape)

print(y.shape)



from sklearn.model_selection import train_test_split



# spliting data in test and train set keeping 70% data in train set and 30% data in test set. 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)

# checking shape of dataset "after" spliting

print("Train Data Details")

print(X_train.shape)

print(X_test.shape)



print("-"*50)



print("Test Data Details")

print(y_train.shape)

print(y_test.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics


# DT Classifier

dt = DecisionTreeClassifier()



# Lets fit the data into classifier 

dt.fit(X_train, y_train)



# predict on test data

y_pred = dt.predict(X_test)



#confusion matrix

cm = metrics.confusion_matrix(y_test, y_pred)





# Confusion Matrix

import seaborn as sns

sns.heatmap(cm, annot=True)
# let's check Classification accuracy

print( "Classification Accuracy:  ", dt.score(X_test, y_test))



print("*"*50)



# Recall Score

from sklearn.metrics import recall_score

print("Recall Score: ", recall_score(y_test, y_pred, average='macro'))



print("*"*50)



# Precision Score

from sklearn.metrics import precision_score

print("Precision Score: ", precision_score(y_test, y_pred, average='macro'))





print("*"*50)



# F1 Score 

# f1 Score = 2 * (precision * recall) / (precision + recall)



from sklearn.metrics import f1_score

print("F1 Score", f1_score(y_test, y_pred, average='macro'))

from sklearn.svm import LinearSVC



#Linear SVC Classifier

clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=100, multi_class='ovr')



# Fitting training data into classifier

clf.fit(X_train,y_train)



# Accuracy

print('Accuracy of linear SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))

print('Accuracy of linear SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) *100))



y_pred = clf.predict(X_test)
#confusion matrix

cm = metrics.confusion_matrix(y_test, y_pred)



# Confusion Matrix

import seaborn as sns

sns.heatmap(cm, annot=True)
# Recall Score

from sklearn.metrics import recall_score

print("Recall Score: ", recall_score(y_test, y_pred, average='macro'))



print("*"*50)



# Precision Score

from sklearn.metrics import precision_score

print("Precision Score: ", precision_score(y_test, y_pred, average='macro'))





print("*"*50)



# F1 Score 

# f1 Score = 2 * (precision * recall) / (precision + recall)



from sklearn.metrics import f1_score

print("F1 Score", f1_score(y_test, y_pred, average='macro'))

from sklearn.svm import SVC



# SVC Classifier

clf_SVC = SVC(C=100.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 

          probability=False, tol=0.001, cache_size=200, class_weight=None, 

          verbose=0, max_iter=-1, decision_function_shape="ovr", random_state = 0)



# Fitting training data

clf_SVC.fit(X_train,y_train)



# predictions

y_pred = clf_SVC.predict(X_test)



# predicting accuracies

print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))



print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

#confusion matrix

cm = metrics.confusion_matrix(y_test, y_pred)



# Confusion Matrix

import seaborn as sns

sns.heatmap(cm, annot=True)
# Recall Score

from sklearn.metrics import recall_score

print("Recall Score: ", recall_score(y_test, y_pred, average='macro'))



print("*"*50)



# Precision Score

from sklearn.metrics import precision_score

print("Precision Score: ", precision_score(y_test, y_pred, average='macro'))





print("*"*50)



# F1 Score 

# f1 Score = 2 * (precision * recall) / (precision + recall)



from sklearn.metrics import f1_score

print("F1 Score", f1_score(y_test, y_pred, average='macro'))

from sklearn.neural_network import MLPClassifier # neural network



# Classifier

clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)  # try once with solver='lbfgs',



#Fiting trainging data

clf.fit(X_train, y_train)



#predicting the data

y_pred = clf.predict(X_test)



print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(y_pred, y_test))
#confusion matrix

cm = metrics.confusion_matrix(y_test, y_pred)



# Confusion Matrix

import seaborn as sns

sns.heatmap(cm, annot=True)
# Recall Score

from sklearn.metrics import recall_score

print("Recall Score: ", recall_score(y_test, y_pred, average='macro'))



print("*"*50)



# Precision Score

from sklearn.metrics import precision_score

print("Precision Score: ", precision_score(y_test, y_pred, average='macro'))





print("*"*50)



# F1 Score 

# f1 Score = 2 * (precision * recall) / (precision + recall)



from sklearn.metrics import f1_score

print("F1 Score", f1_score(y_test, y_pred, average='macro'))

from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups()



print(twenty_train.DESCR)


# text data sample

train_data = twenty_train.data



# length of data

print("Length of complete Data: ",len(train_data))

print("*"*50)



#sample data point (news)

print("sample data point \n"+"* *"*20)

print(train_data[1])
import nltk

from nltk.corpus import stopwords

print("STOPWORDS \n",stopwords.words('english'))
# Reference: https://gist.github.com/sebleier/554280



import re

from tqdm import tqdm

from bs4 import BeautifulSoup





def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase





preprocessed = []

# tqdm is for printing the status bar

for sentance in tqdm(train_data):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = decontracted(sentance)

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords.words('english'))

    preprocessed.append(sentance.strip())
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer() #in scikit-learn

count_vect.fit(preprocessed)

print("some feature names ", count_vect.get_feature_names()[:10])

print('='*50)



final_counts = count_vect.transform(preprocessed)

print("the type of count vectorizer ",type(final_counts))

print("the shape of out text BOW vectorizer ",final_counts.get_shape())

print("the number of unique words ", final_counts.get_shape()[1])
from sklearn.feature_extraction.text import TfidfVectorizer



tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)

tf_idf_vect.fit(preprocessed)

print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])

print('='*50)



final_tf_idf = tf_idf_vect.transform(preprocessed)

print("the type of count vectorizer ",type(final_tf_idf))

print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])
#import the necessary module

from sklearn import preprocessing



# create the Labelencoder object

le = preprocessing.LabelEncoder()



#convert the categorical columns into numeric

encoded_value = le.fit_transform(["NLP", "ML", "Artificial Intelligence", "Deep Learning", "ML"])

print("""Encoding of ["NLP", "ML", "Artificial Intelligence", "Deep Learning", "ML"] """)

print(encoded_value)