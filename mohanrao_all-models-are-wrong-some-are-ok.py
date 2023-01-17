# Importing the requisite libraries
import os
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#Importing input files
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#checking the training data
df_train.head()
#checking the training data
df_test.head()
# Exploring the training dataset
print("The data-set has %d rows and %d columns"%(df_train.shape[0],df_train.shape[1]))
# Exploring the testing dataset
print("The data-set has %d rows and %d columns"%(df_test.shape[0],df_test.shape[1]))
# Checking the number of categories for target in the tranining data
category_counter={x:0 for x in set(df_train['target'])}
for each_cat in df_train['target']:
    category_counter[each_cat]+=1

print(category_counter)
#corpus means collection of text. For this particular data-set, in our case it is Review_text
corpus=df_train.question_text
corpus
#Initializing TFIDF vectorizer to conver the raw corpus to a matrix of TFIDF features 
#and also enabling the removal of stopwords.
no_features = 500
vectorizer = TfidfVectorizer(max_df=0.70, min_df=0.001, max_features=no_features, stop_words='english',ngram_range=(1,2))
#creating TFIDF features sparse matrix by fitting it on the specified corpus
tfidf_matrix=vectorizer.fit_transform(corpus).todense()
#grabbing the name of the features.
tfidf_names=vectorizer.get_feature_names()

print("Number of TFIDF Features: %d"%len(tfidf_names)) #same info can be gathered by using tfidf_matrix.shape
# Training data split into training and test data set using 60-40% ratio 

#considering the TFIDF features as independent variables to be input to the classifier

variables = tfidf_matrix

#considering the category values as the class labels for the classifier.

labels = df_train.target

#splitting the data into random training and test sets for both independent variables and labels.

variables_train, variables_test, labels_train, labels_test  =   train_test_split(variables, labels, test_size=.2)
#analyzing the shape of the training and test data-set:
print('Shape of Training Data: '+str(variables_train.shape))
print('Shape of Test Data: '+str(variables_test.shape))
#Applying Logistic Regression

#initializing the object
Logreg_classifier= LogisticRegression(random_state=0)

#fitting the classifier or training the classifier on the training data
Logreg_classifier=Logreg_classifier.fit(variables_train, labels_train)

#after the model has been trained, we proceed to test its performance on the test data
Logreg_predictions=Logreg_classifier.predict(variables_test)

#the trained classifier has been used to make predictions on the test data-set. To evaluate the performance of the model,
#there are a number of metrics that can be used as follows:

Logreg_ascore=sklearn.metrics.accuracy_score(labels_test, Logreg_predictions)

print ("Accuracy Score of Logistic Regression Classifier: %f" %(Logreg_ascore))
#Predicting on the test data 

#Preparing the TF-IDF out of the test data

corpus1=df_test['question_text']

tfidf_matrix1=vectorizer.transform(corpus1).todense()

variables1 = tfidf_matrix1

Logreg_Test_predictions = Logreg_classifier.predict(variables1)

test_id = df_test['qid']

sub_file = pd.DataFrame({'qid': test_id, 'prediction': Logreg_Test_predictions}, columns=['qid', 'prediction'])

sub_file.to_csv('submission.csv', index=False)