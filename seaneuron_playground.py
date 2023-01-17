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

from mpl_toolkits.mplot3d import Axes3D



from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import learning_curve, validation_curve

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.preprocessing import MultiLabelBinarizer, label_binarize

from sklearn.decomposition import PCA

# Classifiers

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OneVsRestClassifier

# Metrics

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score;
df = pd.read_csv("../input/eeg-data.csv")



# convert to arrays from strings

df.raw_values = df.raw_values.map(json.loads)

df.eeg_power = df.eeg_power.map(json.loads)



columns = df.columns.values

df.shape
for i in [0, 2, 3, 4, 10, 11]:

    df = df.drop(columns[i], axis=1)

#df = df.drop('Unnamed: 0', axis=1)

#df = df.drop('indra_time', axis=1)

#df = df.drop('browser_latency', axis=1)

#df = df.drop('reading_time', axis=1)

#df = df.drop('createdAt', axis=1)

#df = df.drop('updatedAt', axis=1)



df.columns.values
df[['id', 'attention_esense', 'meditation_esense', 'signal_quality']].hist()
df[df['signal_quality']==200]['label'].value_counts()
df = df.drop('signal_quality', axis=1)



df.shape
proc_df = df.drop('raw_values', axis=1)

raw_df = df.drop('eeg_power', axis=1)

proc_df.shape, raw_df.shape
to_series = pd.Series(proc_df['eeg_power'])

eeg_features=pd.DataFrame(to_series.tolist(), columns=["delta", "theta", "low_alpha", "high_alpha", \

                                                      "low_beta", "high_beta", "low_gamma", "mid_gamma"])

proc_df = pd.concat([proc_df,eeg_features], axis=1)

proc_df = proc_df.drop('eeg_power', axis=1)

proc_df = proc_df.drop('id', axis=1)

proc_df.shape
relax = proc_df[proc_df.label == 'relax']

math = proc_df[(proc_df.label == 'math1') |

              (proc_df.label == 'math2') |

              (proc_df.label == 'math3') |

              (proc_df.label == 'math4') |

              (proc_df.label == 'math5') |

              (proc_df.label == 'math6') |

              (proc_df.label == 'math7') |

              (proc_df.label == 'math8') |

              (proc_df.label == 'math9') |

              (proc_df.label == 'math10') |

              (proc_df.label == 'math11') |

              (proc_df.label == 'math12') ]

music = proc_df[proc_df.label == 'music']





len(relax), len(math), len(music)
relax['label'] = 0

music['label'] = 1

math['label'] = 2

train_data = pd.concat([relax, music, math])

train_data['label'].value_counts()
data = train_data.drop('label', axis=1)



scaler = StandardScaler()

data = scaler.fit_transform(data)

pca = PCA(n_components=3)

pca_data = pca.fit_transform(data)



fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=50)

ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],\

        c=train_data["label"].apply(lambda label: {0:'red', 1:'green', 2:'blue'}[label]))
pca.explained_variance_
#mlb = MultiLabelBinarizer()

#y = []

#for i in train_data['label']:

#    i = [i]

#    y.append(i)

#y = mlb.fit_transform(y)

y = train_data['label']

y = label_binarize(y, classes=[0, 1, 2])

X_train, X_test, y_train, y_test = train_test_split(

       train_data, y, test_size=0.33, random_state=42)



X_train = X_train.drop("label", axis=1)

X_test = X_test.drop("label", axis=1)

X_train.shape, y_train.shape
X_train = X_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)

X_train = pd.DataFrame(scaler.fit_transform(X_train))

X_test = pd.DataFrame(scaler.fit_transform(X_test))
forest = RandomForestClassifier(100, max_features=None, min_samples_split=3, min_samples_leaf=3,\

                                random_state=42, n_jobs=-1, oob_score=True)

forest.fit(X_train, y_train)
features = pd.DataFrame(forest.feature_importances_,

                        index=train_data.drop('label', axis=1).columns, 

                        columns=['Importance']).sort_values(['Importance'], 

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
cv = 5

for clf in clfs:

    print(type(clf).__name__, np.mean(cross_val_score(OneVsRestClassifier(clf),\

                                                      X_train, y_train, cv=cv, scoring="f1_samples", n_jobs=-1)))
mcforest = OneVsRestClassifier(dectree)
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



param_dist = {"estimator__max_depth": [3, 5, 7],

              "estimator__min_samples_split": np.arange(2, 10),

              "estimator__min_samples_leaf": np.arange(2, 100, 2),

              "estimator__criterion": ["gini", "entropy"]}



n_iter_search = 20

random_search = RandomizedSearchCV(mcforest, param_distributions=param_dist,

                                   n_iter=n_iter_search, scoring="f1_samples", n_jobs=-1)



random_search.fit(X_train, y_train)

print("RandomizedSearchCV for %d candidates"

      " parameter settings." % (n_iter_search))

report(random_search.cv_results_)

dectree = DecisionTreeClassifier(max_depth=7, min_samples_split=5, criterion='entropy', min_samples_leaf=62)

mcforest = OneVsRestClassifier(dectree)

mcforest.fit(X_train, y_train)
accuracy_score(y_test, mcforest.predict(X_test))
f1_score(y_test, mcforest.predict(X_test), average='weighted')