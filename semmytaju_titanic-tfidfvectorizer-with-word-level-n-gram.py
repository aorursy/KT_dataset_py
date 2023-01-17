import os
print(os.listdir("../input"))

"""
Details:
    @Author: ---
    @Created on Mon Mar 19 20:28:35 2018

Packages:
    Python => 3.6.3 :: Anaconda custom (64-bit)
    
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
import math

##########################################################
# LOAD DATASET
##########################################################
# Kaggle load data format
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
get_test_class = pd.read_csv('../input/gender_submission.csv')

# Load datasets from windows
#train = pd.read_csv('D:/YZU CLASSES/Machine Learning Class/ASSIGNMENTS/ASSG 1/Code/train.csv')
#test = pd.read_csv('D:/YZU CLASSES/Machine Learning Class/ASSIGNMENTS/ASSG 1/Code/test.csv')
#get_test_class = pd.read_csv('D:/YZU CLASSES/Machine Learning Class/ASSIGNMENTS/ASSG 1/Code/gender_submission.csv')

# View only 5 rows data
train.head(5);

# Get column names (Selection features, only)
col_names = ['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
# col_names = list(train.columns.values)
print(col_names);

##########################################################
# TRAINING DATA
##########################################################
# Store passenger ID
trnPassengerId = train['PassengerId']
trnPassengerSurvived = train['Survived']

# Delete columns - axis number (0 for rows and 1 for columns)
train = train.drop('PassengerId', axis=1)
train = train.drop('Survived', axis=1)

# Convert multiple columns to string
train = train[col_names].astype(str) 
train.dtypes # Updated datatype

# Join multiple columns into single string column with apply function, sep by space
train['join-columns'] = train[col_names].apply(lambda x: ' '.join(x), axis=1)

# Get str train and class (dataframe to list)
strTrain = train['join-columns'].values.tolist()
trainClass = trnPassengerSurvived.values.tolist()


##########################################################
# TESTING DATA
##########################################################
# Store passenger ID
testPassengerId = test['PassengerId']

# Convert multiple columns to string
test = test[col_names].astype(str) 
test.dtypes # Updated datatype
print(test.dtypes);

# Join multiple string columns with apply function, sep by single space
test['join-columns'] = test[col_names].apply(lambda x: ' '.join(x), axis=1)

# Get str train and class (dataframe to list)
strTest = test['join-columns'].values.tolist()
testClass = get_test_class['Survived'].values.tolist()


##########################################################
# FEATURE GENERATION: TfidfVectorizer
##########################################################
# Vectorizing data
make_vectorizer = TfidfVectorizer(lowercase=True, 
                     binary=False,
                     analyzer='word', # Word level (Acc: 88.75%)
                     #analyzer='char_wb', # Char level with whitespace baundary
                     # unigram, bigram and 3-gram (trigram), 4-gram
                     ngram_range=(1, 4) # (Acc: 86.60%)
                     #ngram_range=(3, 3) # 3-gram
                     #,min_df = 0
                  )

trainset = make_vectorizer.fit_transform(strTrain)
#print(trainset.toarray())
print(trainset.shape)
print(type(make_vectorizer))
#print(make_vectorizer.vocabulary_)

# Transform test set data
testset = make_vectorizer.transform(strTest)
##########################################################
# PREDICTIONS
##########################################################
# Training Perceptron classifier
my_classifier = Perceptron(max_iter=100)
my_classifier.fit(trainset, trainClass)
print('Perceptron Accuracy: ', my_classifier.score(testset, testClass))

# Label predictions
results_class_pred = my_classifier.predict(testset)
SubmissionFile1 = pd.DataFrame({ 'PassengerId': testPassengerId, 'Survived': results_class_pred })
SubmissionFile1.to_csv("PerceptronSubmissionFile.csv", index=False)

# Training Multinomial Naive Bayes classifier
my_classifier = MultinomialNB(alpha=0.001)
my_classifier.fit(trainset, trainClass)
print('MultinomialNB Accuracy: ', my_classifier.score(testset, testClass))
# Label predictions
results_class_pred = my_classifier.predict(testset)
SubmissionFile2 = pd.DataFrame({ 'PassengerId': testPassengerId, 'Survived': results_class_pred })
SubmissionFile2.to_csv("MultinomialNBSubmissionFile.csv", index=False)

# Training Gaussian Naive Bayes classifier
my_classifier = GaussianNB(priors=None)
my_classifier.fit(trainset.toarray(), trainClass)
print('GaussianNB Accuracy: ', my_classifier.score(testset.toarray(), testClass))
# Label predictions
results_class_pred = my_classifier.predict(testset.toarray())
SubmissionFile3 = pd.DataFrame({ 'PassengerId': testPassengerId, 'Survived': results_class_pred })
SubmissionFile3.to_csv("GaussianNBSubmissionFile.csv", index=False)

# Training Random Forest classifier
my_classifier = RandomForestClassifier(max_depth=2, random_state=0)
my_classifier.fit(trainset, trainClass)
print('RandomForest Accuracy: ', my_classifier.score(testset, testClass))
#print('RandomForest Feature Importances: ', my_classifier.feature_importances_)
# Label predictions
results_class_pred = my_classifier.predict(testset)
SubmissionFile4 = pd.DataFrame({ 'PassengerId': testPassengerId, 'Survived': results_class_pred })
SubmissionFile4.to_csv("RandomForestSubmissionFile.csv", index=False)

# Training Nearest Neighbors classifier
n_neighbors = 50
weights = 'distance'
my_classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
my_classifier.fit(trainset, trainClass)
print('Nearest Neighbors Accuracy: ', my_classifier.score(testset, testClass))
# Label predictions
results_class_pred = my_classifier.predict(testset)
SubmissionFile5 = pd.DataFrame({ 'PassengerId': testPassengerId, 'Survived': results_class_pred })
SubmissionFile5.to_csv("NearestNeighborsSubmissionFile.csv", index=False)

# Training Support Vector Machine - SVM - classifier
C = 1.0  # SVM regularization parameter
my_classifier = svm.SVC(kernel='linear', gamma=0.9, C=C, probability=True)
my_classifier.fit(trainset, trainClass)
print('SVM Accuracy: ', my_classifier.score(testset, testClass))
# Label predictions
results_class_pred = my_classifier.predict(testset)
SubmissionFile6 = pd.DataFrame({ 'PassengerId': testPassengerId, 'Survived': results_class_pred })
SubmissionFile6.to_csv("SVMSubmissionFile.csv", index=False)

# Training Nearest Centroid Classifier
my_classifier = NearestCentroid(metric='euclidean', shrink_threshold=None)
my_classifier.fit(trainset, trainClass)
print('Nearest Centroid Accuracy: ', my_classifier.score(testset, testClass))
# Label predictions
results_class_pred = my_classifier.predict(testset)
SubmissionFile7 = pd.DataFrame({ 'PassengerId': testPassengerId, 'Survived': results_class_pred })
SubmissionFile7.to_csv("NearestCentroidSubmissionFile.csv", index=False)

# Label predictions (High acc)
print("Predicted label (TfidfVectorizer): \n",results_class_pred)






