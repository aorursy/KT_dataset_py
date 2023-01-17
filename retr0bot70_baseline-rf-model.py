import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
top50country = pd.read_csv("../input/top-50-spotify-songs-by-each-country/top50contry.csv", encoding="latin1")

top50 = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv", encoding="latin1")
top50.head()
numerical_values=["bpm", "nrgy", "dnce", "dB", "live", "val", "dur", "acous", "spch"]
from sklearn import preprocessing



x = top50country.drop(["title", "artist", "top genre", "added", "year", "country"], axis=1) #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled)
df.columns = ["nothing","bpm", "nrgy", "dnce", "dB", "live", "val", "dur", "acous", "spch", "popularity"]

df = df.drop(["nothing"], axis=1)
for val in ("bpm", "nrgy", "dnce", "dB", "live", "val", "dur", "acous", "spch"):

    df.plot.scatter(x="popularity", y=val)
df = pd.get_dummies(top50country)

df = df.dropna()

df.head(0)
labels = np.array(df['pop'])



features= df.drop('pop', axis = 1)



feature_list = list(features.columns)



features = np.array(features)
from sklearn.model_selection import train_test_split



train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)



rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)



errors = abs(predictions - test_labels)



print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / test_labels)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
df = top50country.dropna()

df = df.drop(["title","artist","top genre", "year", "added", "Unnamed: 0"], axis=1)

df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('country', axis=1), df['country'], test_size=0.20, random_state=0)
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score

baseline = DummyClassifier(strategy='most_frequent', random_state=0).fit(X_train, y_train)

y_pred = baseline.predict(X_test)

print(round(accuracy_score(y_test, y_pred),4))
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier()

    ]

for classifier in classifiers:

    model = classifier.fit(X_train, y_train)

    print(classifier)

    print("model score: %.3f" % model.score(X_test, y_test))
from sklearn.model_selection import GridSearchCV

n_estimators = [100, 300, 500, 800, 1200]

max_depth = [5, 8, 15, 25, 30]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]

param_grid = dict(n_estimators = n_estimators, max_depth = max_depth,  

              min_samples_split = min_samples_split, 

             min_samples_leaf = min_samples_leaf)

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid)

best_model = grid_search.fit(X_train, y_train)

print(round(best_model.score(X_test, y_test),2))

print(best_model.best_params_)
from sklearn.metrics import classification_report

y_pred_best = best_model.predict(X_test)

print(classification_report(y_test, y_pred_best))