import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import string

import spacy

import re

nlp = spacy.load("en_core_web_lg")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
full_train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

full_submission_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(full_submission_data.head())

print(full_train_data.head())

train_data_shape = full_train_data.shape[0]
sns.countplot(full_train_data['target'])
df = pd.concat([full_train_data, full_submission_data])

df.shape
def clean_text(text):

    # remove_URL

    url = re.compile(r'https?://\S+|www\.\S+')

    text =  url.sub(r'', text)



    # remove_html

    html = re.compile(r'<.*?>')

    text = html.sub(r'', text)



    # remove_emoji

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags = re.UNICODE)

    text = emoji_pattern.sub(r'', text)



    # remove_punct

    table = str.maketrans('', '', string.punctuation)

    text = text.translate(table)

    

    return text





df['text'] = df['text'].apply(lambda x : clean_text(x))
with nlp.disable_pipes():

    doc_vectors = np.array([nlp(text).vector for text in df["text"]])
train_doc_vectors = doc_vectors[:train_data_shape]

submission_doc_vectors = doc_vectors[train_data_shape:]
from sklearn.model_selection import train_test_split



X = train_doc_vectors

y = full_train_data.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)
from sklearn import svm

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



predictions = model.predict(X_test)

score = accuracy_score(predictions, y_test) * 100

print(f'Accuracy: {score}.')
from sklearn.linear_model import LogisticRegression # Logistic Regression

from sklearn.ensemble import RandomForestClassifier # Random Forest

from sklearn.neighbors import KNeighborsClassifier # KNN

from sklearn.naive_bayes import GaussianNB # Naive bayes

from sklearn.tree import DecisionTreeClassifier # Decision Tree

from sklearn.model_selection import train_test_split # Training and testing data split

from sklearn.metrics import confusion_matrix # For confusion matrix
model = svm.SVC(kernel='rbf')

model.fit(X_train, y_train)

predictions_2 = model.predict(X_test)

score = accuracy_score(y_test, predictions_2) * 100

print(f'Accuracy of SVM-r: {score}.')
model2 = LogisticRegression(C=1.5)

model2.fit(X_train, y_train)

predictions_3 = model2.predict(X_test)

score = accuracy_score(y_test, predictions_3) * 100

print(f'Accuracy of Logistic Regression: {score}.')
model=DecisionTreeClassifier()

model.fit(X_train, y_train)

predictions_4 = model.predict(X_test)

score = accuracy_score(y_test, predictions_4) * 100

print(f'Accuracy of Decision Tree: {score}.')
model=RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

predictions_5 = model.predict(X_test)

score = accuracy_score(y_test, predictions_5) * 100

print(f'Accuracy of Random Forest: {score}.')
model=KNeighborsClassifier() 

model.fit(X_train, y_train)

predictions_6 = model.predict(X_test)

score = accuracy_score(y_test, predictions_6) * 100

print(f'Accuracy of K-Nearest Neighbours: {score}.')
# model = GaussianNB()

# model.fit(X_train, y_train)

# predictions_7 = model.predict(X_test)

# score = accuracy_score(y_test, predictions_7) * 100

# print(f'Accuracy of Naive Bayes: {score}.')
# from sklearn.model_selection import GridSearchCV



# C = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# gamma=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# kernel=['rbf', 'linear']

# hyper={'kernel':kernel, 'C':C, 'gamma':gamma}



# gd = GridSearchCV(estimator=svm.SVC(), param_grid = hyper, verbose=True)

# gd.fit(X,y)

# print(gd.best_score_)

# print(gd.best_estimator_)
# C_param_range = [0.001,0.01,0.1,1,10,100]

# penalty = ['l1','l2', 'elasticnet']

# hyper={'C':C_param_range}



# gd = GridSearchCV(estimator = LogisticRegression(), param_grid = hyper, verbose = True)

# gd.fit(X,Y)

# print(gd.best_score_)

# print(gd.best_estimator_)
# from sklearn.ensemble import VotingClassifier

# ensemble_voting = VotingClassifier(estimators = [('RBF',svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.1)),

#                                               ('LR',LogisticRegression(C=1.5)),

#                                               ('svm',svm.SVC(kernel='linear', probability=True))

#                                              ], 

#                        voting='soft').fit(X_train, y_train)



# predictions_vote = ensemble_voting.predict(X_test)

# score = accuracy_score(y_test, predictions_vote) * 100

# print(f'Accuracy of Voting Classifier: {score}.')
# from sklearn.ensemble import AdaBoostClassifier



# ada = AdaBoostClassifier(n_estimators = 200, random_state = 0, learning_rate = 0.1)

# ada.fit(X_train, y_train)

# predictions_6 = ada.predict(X_test)

# score = accuracy_score(y_test, predictions_6) * 100

# print(f'Accuracy of AdaBoost: {score}.')
# Make predictions using the trained model

model_final = LogisticRegression(C=1.5)

model_final.fit(X, y)

predictions_final = model_final.predict(submission_doc_vectors)
# # Generate results

results = pd.Series(predictions_final, name="target")

tweet_ids = full_submission_data['id']



submission = pd.concat([tweet_ids, results], axis=1)

submission.head()

submission.to_csv("submission.csv",index=False)