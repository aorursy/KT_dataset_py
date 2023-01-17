%pylab inline

%matplotlib inline

import time

import itertools

# Data processing

import json

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import numpy as np

# Plotting

import matplotlib.pyplot as plt

import pylab as pl



from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import learning_curve, validation_curve

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.decomposition import PCA

# Classifiers

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

# Metrics

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score
df = pd.read_csv("../input/eeg-data.csv")



# convert to arrays from strings

df.raw_values = df.raw_values.map(json.loads)

df.eeg_power = df.eeg_power.map(json.loads)



df.head(2)
relax = df[df.label == 'relax']

math = df[(df.label == 'math1') |

          (df.label == 'math2') |

          (df.label == 'math3') |

          (df.label == 'math4') |

          (df.label == 'math5') |

          (df.label == 'math6') |

          (df.label == 'math7') |

          (df.label == 'math8') |

          (df.label == 'math9') |

          (df.label == 'math10') |

          (df.label == 'math11') |

          (df.label == 'math12') ]



len(relax), len(math)
def vectors_labels (list1, list2):

    def label (l):

        return lambda x: l

    X = list1 + list2

    y = list(map(label(0), list1)) + list(map(label(1), list2))

    return X, y
train_data = pd.DataFrame(columns=["delta", "theta", "low_alpha", "high_alpha", \

                                 "low_beta", "high_beta", "low_gamma", "mid_gamma"])

for id in unique(math["id"]):

    one_math = math[math['id']==id]

    one_relax = relax[relax['id']==id]

    X, y = vectors_labels(one_math.eeg_power.tolist(), one_relax.eeg_power.tolist())

    X = np.matrix(X)

    y = np.array(y)

    data = pd.DataFrame(data=X,

                        index=y,

                        columns=["delta", "theta", "low_alpha", "high_alpha", \

                                 "low_beta", "high_beta", "low_gamma", "mid_gamma"])

    frames = [train_data, data]

    train_data = pd.concat(frames)

train_data.head(2)
train_data.index.value_counts()
train_data.info()
train_data.describe().T
import seaborn as sns

corr = train_data.corr()

figsize(8, 6)

sns.heatmap(corr)
plots = train_data.hist()
scaler = StandardScaler()

data = scaler.fit_transform(train_data)

pca = PCA(n_components=2)

pca_data = pca.fit_transform(data)

plot_data = train_data

plot_data["relax"] = train_data.index

scatter(pca_data[:, 0], pca_data[:, 1],\

        c=plot_data["relax"].apply(lambda relax: 'red' if relax else 'green'))
X_train, X_test, y_train, y_test = train_test_split(

       train_data, train_data.index, test_size=0.33, random_state=42)

X_train = X_train.drop("relax", axis=1)

X_test = X_test.drop("relax", axis=1)

X_train.shape
X_train = X_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)

X_train = pd.DataFrame(scaler.fit_transform(X_train))

X_test = pd.DataFrame(scaler.fit_transform(X_test))
forest = RandomForestClassifier(100, max_features=None, min_samples_split=3, min_samples_leaf=3,\

                                random_state=42, n_jobs=-1, oob_score=True)

forest.fit(X_train, y_train)
features = pd.DataFrame(forest.feature_importances_,

                        index=train_data.columns[0:8], 

                        columns=['Importance']).sort(['Importance'], 

                                                     ascending=False)

features
knn = KNeighborsClassifier(n_neighbors=10)

dectree = DecisionTreeClassifier(max_depth=None, min_samples_split=2)

extree = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2)

grboost = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1)

#mlp = MLPClassifier(random_state=42)

svclf = SVC(random_state=42)

gpc = GaussianProcessClassifier()

clfs = [knn, dectree, extree, grboost, svclf, gpc, forest]
cv = StratifiedKFold(n_splits=10)

for clf in clfs:

    print(type(clf).__name__, np.mean(cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)))
def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")



param_dist = {"max_depth": [3, 5, 7],

              "n_estimators": np.arange(2, 100, 2),

              "max_features": np.arange(2, 8),

              "min_samples_split": np.arange(2, 8),

              "min_samples_leaf": np.arange(2, 100, 2),

              "criterion": ["gini", "entropy"]}



n_iter_search = 20

random_search = RandomizedSearchCV(forest, param_distributions=param_dist,

                                   n_iter=n_iter_search, scoring="roc_auc", n_jobs=-1)



random_search.fit(X_train, y_train)

print("RandomizedSearchCV for %d candidates"

      " parameter settings." % (n_iter_search))

report(random_search.cv_results_)
train_sizes, train_scores, valid_scores = learning_curve(

    forest, X_train, y_train, train_sizes=[50, 100, 150, 300, 500, 800], scoring='roc_auc')

pl.plot(train_sizes, np.mean(train_scores, axis=1), label='training scores', color='green')

pl.plot(train_sizes, np.mean(valid_scores, axis=1), label='validation scores', color='red')

pl.legend()

pl.figure(num=None, figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')

pl.show;
cv_param_values = np.arange(2, 100, 2)

train_scores, valid_scores = validation_curve(forest, X, y, "n_estimators", cv_param_values,\

                                              cv=5, scoring='roc_auc')

pl.plot(cv_param_values, np.mean(train_scores, axis=1), label='training scores', c='green')

pl.plot(cv_param_values, np.mean(valid_scores, axis=1), label='validation scores', c='red')

plt.legend()

plt.show();
forest = RandomForestClassifier(54, max_features=None, min_samples_split=3, min_samples_leaf=44,\

                                random_state=42, n_jobs=-1, oob_score=True)

forest.fit(X_train, y_train)
accuracy_score(y_test, forest.predict(X_test))
roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, forest.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, color='darkorange',

         lw=2, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show();
f1_score(y_test, forest.predict(X_test))
confusion_matrix(y_test, forest.predict(X_test))
from scipy import stats

from scipy.interpolate import interp1d

import itertools

import numpy as np



def spectrum (vector):

    '''get the power spectrum of a vector of raw EEG data'''

    A = np.fft.fft(vector)

    ps = np.abs(A)**2

    ps = ps[:len(ps)//2]

    return ps



def binned (pspectra, n):

    '''compress an array of power spectra into vectors of length n'''

    l = len(pspectra)

    array = np.zeros([l,n])

    for i,ps in enumerate(pspectra):

        x = np.arange(1,len(ps)+1)

        f = interp1d(x,ps)#/np.sum(ps))

        array[i] = f(np.arange(1, n+1))

    index = np.argwhere(array[:,0]==-1)

    array = np.delete(array,index,0)

    return array



def feature_vector (readings, bins=100): # A function we apply to each group of power spectra

  '''

  Create 100, log10-spaced bins for each power spectrum.

  For more on how this particular implementation works, see:

  http://coolworld.me/pre-processing-EEG-consumer-devices/

  '''

  bins = binned(list(map(spectrum, readings)), bins)

  return np.log10(np.mean(bins, 0))



ex_readings = one_relax.raw_values[:3]

feature_vector(ex_readings)



def grouper(n, iterable, fillvalue=None):

    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"

    args = [iter(iterable)] * n

    return itertools.zip_longest(*args, fillvalue=fillvalue)



def vectors (df):

    return [feature_vector(group) for group in list(grouper(3, df.raw_values.tolist()))[:-1]]
X,y = vectors_labels(

    vectors(math),

    vectors(relax))



X = pd.DataFrame(X)

X.shape
X_train, X_test, y_train, y_test = train_test_split(

       X, y, test_size=0.33, random_state=42)

len(X_train)
forest = RandomForestClassifier(100, max_features=None, min_samples_split=3, min_samples_leaf=3,\

                                random_state=42, n_jobs=-1, oob_score=True)

knn = KNeighborsClassifier(n_neighbors=10)

dectree = DecisionTreeClassifier(max_depth=None, min_samples_split=2)

extree = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2)

grboost = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1)

#mlp = MLPClassifier(random_state=42)

svclf = SVC(random_state=42)

gpc = GaussianProcessClassifier()

clfs = [knn, dectree, extree, grboost, svclf, gpc, forest]
cv = StratifiedKFold(n_splits=5)

for clf in clfs:

    print(type(clf).__name__, np.mean(cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)))
param_dist = {'kernel': ['linear', 'poly', 'rbf'],

               'gamma': [1e-3, 1e-4],

               'C': [1, 10, 100, 1000]}



n_iter_search = 20

random_search = RandomizedSearchCV(svclf, param_distributions=param_dist,

                                   n_iter=n_iter_search, scoring="roc_auc", n_jobs=-1)



random_search.fit(X_train, y_train)

print("RandomizedSearchCV for %d candidates"

      " parameter settings." % (n_iter_search))

report(random_search.cv_results_)
svclf = SVC(C=10, gamma=0.001, probability=True, random_state=42)

svclf.fit(X_train, y_train)
accuracy_score(y_test, svclf.predict(X_test))
roc_auc_score(y_test, svclf.predict_proba(X_test)[:, 1])
fpr, tpr, thresholds = roc_curve(y_test, svclf.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, color='darkorange',

         lw=2, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show();
f1_score(y_test, svclf.predict(X_test))
confusion_matrix(y_test, svclf.predict(X_test))