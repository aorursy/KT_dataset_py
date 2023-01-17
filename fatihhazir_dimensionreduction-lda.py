import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/winewithclassification/Wine.csv')
df.head()
df.shape
X = df.iloc[:, 0:13]

y = df.iloc[:, 13]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2) # kac oznitelik alinacagi

X_train_lda = lda.fit_transform(X_train,y_train)

X_test_lda = lda.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
classifier_lda = LogisticRegression(random_state=0)

classifier_lda.fit(X_train_lda,y_train)

y_pred_lda = classifier_lda.predict(X_test_lda)
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_pred,y_pred_lda)

print(confmat)