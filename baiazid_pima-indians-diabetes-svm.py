import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv('../input/diabetes.csv')
print(dataset.describe())

X = dataset.iloc[:, 0:8]

y = dataset.iloc[:, 8]


sns.heatmap(X.corr(), annot = True)

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']



for column in zero_not_accepted:

    X[column] = X[column].replace(0, np.NaN)

    mean = int(X[column].mean(skipna=True))

    X[column] = X[column].replace(np.NaN, mean)
## Var[X] = p(1-p)

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

X_filtered = sel.fit_transform(X)



print(X.head(1))

print(X_filtered[0])

#DiabetesPedigreeFunction was dropped

X = X.drop('DiabetesPedigreeFunction', axis=1)



top_4_features = SelectKBest(score_func=chi2, k=4)

X_top_4_features = top_4_features.fit_transform(X, y)

print(X.head())

print(X_top_4_features)

X = X.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
classifier = SVC(random_state=0, kernel='rbf')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print (cm)

print(f1_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))