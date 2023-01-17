import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset=pd.read_csv('../input/liverpool-fc-stats-from-19932018/liverpoolfcstats.csv')

X=dataset.iloc[:,2:4]
for a in X:

    a.replace(" ","")
print(X)
y=dataset.iloc[:,6].values
print(y)
X['Match']=X['HomeTeam']+X['AwayTeam']

print(X)
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 958):

  review = re.sub('[^a-zA-Z]', ' ', X['Match'][i])

  review = review.lower()

  review = review.split()

  ps = PorterStemmer()

  all_stopwords = stopwords.words('english')

  #all_stopwords.remove('not')

  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

  review = ' '.join(review)

  corpus.append(review)

print(corpus)
X2=X.iloc[:,2]
print(X2)
for match in X2:

    match.replace(" ","")

print(X2)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)

X2 = cv.fit_transform(corpus).toarray()

# from sklearn.compose import ColumnTransformer

# from sklearn.preprocessing import OneHotEncoder

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# X2 = X2.values.reshape(len(y), 1)

# X2 = np.array(ct.fit_transform(X2))

# print(X2)
# from sklearn.compose import ColumnTransformer

# from sklearn.preprocessing import OneHotEncoder

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# X2 = y.reshape(len(X2),1)

# X2 = np.array(ct.fit_transform(X2))

# print(X2)

# trainDfDummies = pd.get_dummies(X2, columns=['Match'])

# print(trainDfDummies)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

print(y)

y = y.reshape(len(y),1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 1/3, random_state = 0)
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)

# X_test = sc.transform(X_test)

# print(X_train)

# print(X_test)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# from sklearn.metrics import confusion_matrix, accuracy_score

# cm = confusion_matrix(y_test, y_pred)

# print(cm)

# accuracy_score(y_test, y_pred)
# I=input('Enter match as LiverpoolLeeds')

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# I = le.fit_transform([I])

# I = I.reshape(len(I),1)

# print(sc.transform(I))

# print(classifier.predict(sc.transform(I)))
# from sklearn.naive_bayes import GaussianNB

# classifier = GaussianNB()

# classifier.fit(X_train, y_train)
# I=input('Enter match as LiverpoolLeeds')

# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# I = le.fit_transform([I])

# I = I.reshape(len(I),1)

# print(sc.transform(I))

# print(classifier.predict(sc.transform(I)))
# from sklearn.metrics import confusion_matrix, accuracy_score

# cm = confusion_matrix(y_test, y_pred)

# print(cm)

# accuracy_score(y_test, y_pred)
# from sklearn.naive_bayes import GaussianNB

# classifier = GaussianNB()

# classifier.fit(X_train, y_train)
# I=input('Enter match as LiverpoolLeeds')



# #I = I.reshape(len(I),1)

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# I = np.array(ct.fit_transform(I))

# I = I.values.reshape(len(I),1)

# print(I)







# #I = I.reshape(len(I),1)

# print(sc.transform(I))

# print(classifier.predict(sc.transform(I)))
y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)