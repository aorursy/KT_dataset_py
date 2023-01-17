import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix 
#dataset is separated by tab, so we use seperator='\t'

data = pd.read_csv('../input/sms-data-labelled-spam-and-non-spam/SMSSpamCollection', sep='\t', names=['label', 'message'])
data.head()
data.info()
#use '1' for spam and '0' for not spam

data['label'] = data.label.map({'ham':0, 'spam':1})

data.head()
# split into training and testing sets

# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data['message'], 

                                                    data['label'], 

                                                    test_size =0.2, 

                                                    random_state=1)





print('Number of rows in the total set: {}'.format(data.shape[0]))

print('Number of rows in the training set: {}'.format(X_train.shape[0]))

print('Number of rows in the test set: {}'.format(X_test.shape[0]))
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
# Fit the training data and then return the matrix

training_data = count_vector.fit_transform(X_train).toarray()



# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()

testing_data = count_vector.transform(X_test).toarray()
frequency_matrix = pd.DataFrame(training_data, 

                                columns = count_vector.get_feature_names())

frequency_matrix.head()
testing_data
#Train the data

clf = LogisticRegression(random_state=0).fit(training_data, y_train)
#predict the value

predictions = clf.predict(testing_data)
predictions
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))

print('Precision score: ', format(precision_score(y_test, predictions)))

print('Recall score: ', format(recall_score(y_test, predictions)))

print('F1 score: ', format(f1_score(y_test, predictions)))

print('\nConfusion Matrix :\n', confusion_matrix(y_test, predictions)) 