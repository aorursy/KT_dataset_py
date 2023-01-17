import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle # I need this to pickle python objects and save them on disks, because I need them on another project



dataset = pd.read_csv('../input/fake-news-detection/data.csv')
print("Total instances : ", len(dataset))

dataset.head()
print("Total NaNs:")

dataset.isna().sum()
dataset=  dataset.drop(['URLs'], axis=1)

dataset = dataset.dropna()
dataset.head()
X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import re

ps = PorterStemmer()

for i in range(len(dataset)):

    X[i][0] = ' '.join([ps.stem(word) for word in re.sub('[^a-zA-Z]', ' ', X[i][0]).lower().split() if not word in stopwords.words('english')])

    X[i][1] = ' '.join([ps.stem(word) for word in re.sub('[^a-zA-Z]', ' ', X[i][1]).lower().split() if not word in stopwords.words('english')])
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)

mat_body = cv.fit_transform(X[:,1]).todense()

pickle.dump(cv, open(r"cv_body.pkl", "wb"))
cv_head = CountVectorizer(max_features=5000)

mat_head = cv_head.fit_transform(X[:,0]).todense()

pickle.dump(cv_head, open(r"cv_head.pkl", "wb"))
print("Body matrix :", mat_body.shape, "Heading matrix :", mat_head.shape)


X_mat = np.hstack(( mat_head, mat_body))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_mat,y, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier

classifier_dtr = DecisionTreeClassifier(criterion='entropy')

classifier_dtr.fit(X_train, y_train)

y_pred_dtr = classifier_dtr.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_dtr)
print(cm)
from sklearn.externals import joblib

joblib.dump(classifier_dtr, "classifier_dtr_fakenews_nourl.pkl")