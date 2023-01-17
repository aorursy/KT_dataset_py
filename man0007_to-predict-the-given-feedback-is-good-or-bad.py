import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset_train = pd.read_csv("../input/antworks_assignment/train.csv",delimiter = '~')

dataset_test  = pd.read_csv("../input/antworks_assignment/test.csv",delimiter = '~')
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

y_train_responce = dataset_train.iloc[:, 4].values

y_train_responce = labelencoder.fit_transform(y_train_responce) 
import re

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
X_train_description = []

for i in range(0,30172):

    review = re.sub('[^a-zA-Z]', ' ', dataset_train['Description'][i]) # removing punctuations

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Stemming and removing stop words

    review = ' '.join(review)

    X_train_description.append(review)
X_test_description = []

for i in range(0,8760):

    review = re.sub('[^a-zA-Z]', ' ', dataset_test['Description'][i]) # removing punctuations

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Stemming and removing stop words

    review = ' '.join(review)

    X_test_description.append(review)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_description = cv.fit_transform(X_train_description).toarray() # Creating sparse matrix for training set

X_test_description  = cv.transform(X_test_description).toarray()  # Creating sparse matrix for test set 
from sklearn.ensemble import RandomForestClassifier

classifier_RF = RandomForestClassifier()
# Undersatanding the accuracy of the model by usin Cross_val_Score

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier_RF, X = X_train_description ,

                             y = y_train_responce, cv=5, n_jobs = 1) # data by itself will be broken down into train and teset set in CV so no need to use train set seperately

mean = accuracies.mean() 

variance = accuracies.std() 
print("Score: ",mean,"Variance: ",variance)
classifier_RF.fit(X_train_description,y_train_responce)
y_test_responce = classifier_RF.predict(X_test_description)

y_test_responce = labelencoder.inverse_transform(y_test_responce)

test_responce = np.concatenate((np.reshape(np.array(dataset_test['User_ID']),(len(dataset_test),1)), np.reshape(y_test_responce,(1,len(dataset_test))).T), axis=1)

np.savetxt("test_responce.csv", test_responce, delimiter="~", header="User_ID~predicted_Response", fmt='%s')
