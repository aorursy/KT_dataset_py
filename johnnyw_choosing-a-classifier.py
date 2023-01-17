import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, accuracy_score, classification_report

from sklearn.cluster import KMeans



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# load the data into a DataFrame and take a look at the first rows

with open('../input/HR_comma_sep.csv', 'r') as f:

    df = pd.read_csv(f)

print(df.head())
# look at the data some more

df.describe()
# convert categorical data (e.g. salary) to numeric categories

def to_categorical(series):

    decoder = dict(enumerate(np.unique(series)))

    encoder = dict((v, k) for (k, v) in decoder.items())

    return np.array([encoder[key] for key in series])



df['sales'] = to_categorical(df['sales'])

df['left'] = to_categorical(df['left'])

df['salary'] = to_categorical(df['salary'])
# shuffle the data and split the target variable

df = df.sample(frac=1).reset_index(drop=True)

X = df.drop(['left'], axis=1).values

y = df['left'].values
# scale the data (0 - 1)

scl = MinMaxScaler()

X = scl.fit_transform(X)
# principal component analysis

pca = PCA(n_components=5)

pca.fit(X)

print(pca.explained_variance_)

reduced_data = pca.fit_transform(X)
# training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
# train classifier

clf = SVC()

clf.fit(X_train, y_train)
# make predictions and evaluate

#score = f1_score(y_test, clf.predict(X_test))

report = classification_report(y_test, clf.predict(X_test))

report
# evaluate some other classifiers

clfs = (SVC(), DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GaussianNB())

trained = map(lambda x: x.fit(X_train, y_train), clfs)

predictions = map(lambda x: x.predict(X_test), trained)

scores = [classification_report(y_test, p) for p in predictions]

for s in scores:

    print(s + "\n")

    
rfc = RandomForestClassifier()

gs = GridSearchCV(rfc, {'n_estimators': [5, 10, 25, 75, 100, 150]}, verbose=1)

gs.fit(X_train, y_train)

gs.best_estimator_