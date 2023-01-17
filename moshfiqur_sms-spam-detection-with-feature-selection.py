# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/spam.csv', encoding='latin-1')
df.head()
df.shape
# Remove garbage columns

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Remove any empty rows

df.dropna(inplace=True)
# Set column names to something meaningful

df.columns = ['type', 'sms']
# Convert target values to numeric

df.loc[df['type'] == 'ham', 'type'] = 0

df.loc[df['type'] == 'spam', 'type'] = 1
df.head()
# Remove any html tags from text (just if any) and lowercase the words

from bs4 import BeautifulSoup

df['sms'] = df['sms'].apply(lambda x: BeautifulSoup(x.lower(), 'html.parser').get_text())
# Separating the features and target values

X = df['sms']

y = df['type'].astype('int')
X.shape, y.shape
from sklearn.model_selection import train_test_split



# We are using 70% of data for training, rest for the testing

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111, test_size=0.3)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
# Lets have a look at the vectorized features

vectorizer.get_feature_names()
# So, we have total 8669 features (or words if its easier to think)

X_train_vectorized.shape
from sklearn.neighbors import KNeighborsClassifier



# Create a K-Nearest Neighbors classifier model with 3 neighbors

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_vectorized, y_train)
# CAUTION: We used the same vectorizer we used for the training data.

# For test data, we are only doing transform, not fit_transform, as we 

# want to use the same vectorizer which was fitted on train data.

X_test_vectorized = vectorizer.transform(X_test)
y_pred = knn.predict(X_test_vectorized)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_pred)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



skb = SelectKBest(score_func=chi2, k=100)

features_fit = skb.fit(X_train_vectorized, y_train)
features_fit.scores_
X_train_selected = features_fit.transform(X_train_vectorized)
X_train_selected.shape
knn_selected = KNeighborsClassifier(n_neighbors=3)

knn_selected.fit(X_train_selected, y_train)
X_test_selected = features_fit.transform(X_test_vectorized)

y_pred_selected = knn_selected.predict(X_test_selected)
accuracy_score(y_test, y_pred_selected)
import pickle



pickle.dump(knn_selected, open('knn_selected.model', 'wb'))



# The thing we always miss is that we also have to save the 

# vectorizer and the feature selection model for future use.

# We have trained our model on those modules and any future 

# data has to be processed by them before we can perform 

# any prediction. Otherwise, we will ended up having a 

# shape mismatch error when we try to predict future data

# on loaded model using processed by fresh vectorizer and 

# feature selection model



# Save the vectorizer

pickle.dump(vectorizer, open('knn.vect', 'wb'))



# Save the feature selection model

pickle.dump(features_fit, open('feature_selector.feat', 'wb'))
os.listdir(os.getcwd())
knn_selected_saved = pickle.load(open('knn_selected.model', 'rb'))

vectorizer_saved = pickle.load(open('knn.vect', 'rb'))

features_fit_saved = pickle.load(open('feature_selector.feat', 'rb'))
saved_pred = knn_selected_saved.predict(X_test_selected)

accuracy_score(y_test, saved_pred)
validation_data = pd.DataFrame.from_dict({

        'sms': ['Baa, baa, black sheep, have you any wool? Yes sir, yes sir, three bags full! One for the master, And one for the dame, One for the little boy Who lives down the lane']

    })



validation_data.head()
# Similar data clean up we did earlier on our train/test dataset

validation_data['sms'] = validation_data['sms'].apply(lambda x: BeautifulSoup(x.lower(), 'html.parser').get_text())
# Note we are processing our data using our loaded vectorizer and 

# feature selection model and only doing transform, not fit_transform

validation_features = vectorizer_saved.transform(validation_data['sms'])

validation_features = features_fit_saved.transform(validation_features)
# Lets see if our feature selection worked

validation_features.shape
# Predict on new unseen data

knn_selected.predict(validation_features)
knn_selected.predict_proba(validation_features)