import os

import numpy as np

import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from io import BytesIO

import requests

import tarfile



url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"



def load_imdb_dataset(extract_path="../input", overwrite=False):

    #check if existed already

    if os.path.isfile(os.path.join(extract_path, "aclImdb", "README")) and not overwrite:

        print("IMDB dataset is already in place.")

        return

    

    print("Downloading the dataset from:  ", url)

    response = requests.get(url)

    

    tar = tarfile.open(mode= "r:gz", fileobj = BytesIO(response.content))

    

    data = tar.extractall(extract_path)

    

load_imdb_dataset()
#change if you have it in alternative location

PATH_TO_IMDB = "../input/aclImdb"



reviews_train = load_files(os.path.join(PATH_TO_IMDB, "train"),

                           categories=['pos', 'neg'])

text_train, y_train = reviews_train.data, reviews_train.target



reviews_test = load_files(os.path.join(PATH_TO_IMDB, "test"),

                          categories=['pos', 'neg'])

text_test, y_test = reviews_test.data, reviews_test.target
# # Alternatively, load data from previously pickled objects. 

# import pickle

# with open('../../data/imdb_text_train.pkl', 'rb') as f:

#     text_train = pickle.load(f)

# with open('../../data/imdb_text_test.pkl', 'rb') as f:

#     text_test = pickle.load(f)

# with open('../../data/imdb_target_train.pkl', 'rb') as f:

#     y_train = pickle.load(f)

# with open('../../data/imdb_target_test.pkl', 'rb') as f:

#     y_test = pickle.load(f)
print("Number of documents in training data: %d" % len(text_train))

print(np.bincount(y_train))

print("Number of documents in test data: %d" % len(text_test))

print(np.bincount(y_test))
print(text_train[1])
y_train[1] # bad review
text_train[2]
y_train[2] # good review
# import pickle

# with open('../../data/imdb_text_train.pkl', 'wb') as f:

#     pickle.dump(text_train, f)

# with open('../../data/imdb_text_test.pkl', 'wb') as f:

#     pickle.dump(text_test, f)

# with open('../../data/imdb_target_train.pkl', 'wb') as f:

#     pickle.dump(y_train, f)

# with open('../../data/imdb_target_test.pkl', 'wb') as f:

#     pickle.dump(y_test, f)
cv = CountVectorizer()

cv.fit(text_train)



len(cv.vocabulary_)
print(cv.get_feature_names()[:50])

print(cv.get_feature_names()[50000:50050])
X_train = cv.transform(text_train)

X_train
print(text_train[19726])
X_train[19726].nonzero()[1]
X_train[19726].nonzero()
X_test = cv.transform(text_test)
%%time

logit = LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=7)

logit.fit(X_train, y_train)
round(logit.score(X_train, y_train), 3), round(logit.score(X_test, y_test), 3),
def visualize_coefficients(classifier, feature_names, n_top_features=25):

    # get coefficients with large absolute values 

    coef = classifier.coef_.ravel()

    positive_coefficients = np.argsort(coef)[-n_top_features:]

    negative_coefficients = np.argsort(coef)[:n_top_features]

    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

    # plot them

    plt.figure(figsize=(15, 5))

    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]

    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)

    feature_names = np.array(feature_names)

    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha="right");

def plot_grid_scores(grid, param_name):

    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_train_score'],

        color='green', label='train')

    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_test_score'],

        color='red', label='test')

    plt.legend();

    
visualize_coefficients(logit, cv.get_feature_names())
%%time

from sklearn.pipeline import make_pipeline



text_pipe_logit = make_pipeline(CountVectorizer(),

                                # for some reason n_jobs > 1 won't work 

                                # with GridSearchCV's n_jobs > 1

                                LogisticRegression(solver='lbfgs', 

                                                   n_jobs=1,

                                                   random_state=7))



text_pipe_logit.fit(text_train, y_train)

print(text_pipe_logit.score(text_test, y_test))
%%time

from sklearn.model_selection import GridSearchCV



param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}

grid_logit = GridSearchCV(text_pipe_logit, 

                          param_grid_logit, 

                          return_train_score=True, 

                          cv=3, n_jobs=-1)



grid_logit.fit(text_train, y_train)
grid_logit.best_params_, grid_logit.best_score_
plot_grid_scores(grid_logit, 'logisticregression__C')
grid_logit.score(text_test, y_test)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=200, 

                                n_jobs=-1, random_state=17)
%%time

forest.fit(X_train, y_train)
round(forest.score(X_test, y_test), 3)
# creating dataset

rng = np.random.RandomState(0)

X = rng.randn(200, 2)

y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired);
def plot_boundary(clf, X, y, plot_title):

    xx, yy = np.meshgrid(np.linspace(-3, 3, 50),

                     np.linspace(-3, 3, 50))

    clf.fit(X, y)

    # plot the decision function for each datapoint on the grid

    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]

    Z = Z.reshape(xx.shape)



    image = plt.imshow(Z, interpolation='nearest',

                           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

                           aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)

    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,

                               linetypes='--')

    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired)

    plt.xticks(())

    plt.yticks(())

    plt.xlabel(r'$x_1$')

    plt.ylabel(r'$x_2$')

    plt.axis([-3, 3, -3, 3])

    plt.colorbar(image)

    plt.title(plot_title, fontsize=12);
plot_boundary(LogisticRegression(solver='lbfgs'), X, y,

              "Logistic Regression, XOR problem")
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
logit_pipe = Pipeline([('poly', PolynomialFeatures(degree=2)), 

                       ('logit', LogisticRegression(solver='lbfgs' ))])
plot_boundary(logit_pipe, X, y,

              "Logistic Regression + quadratic features. XOR problem")