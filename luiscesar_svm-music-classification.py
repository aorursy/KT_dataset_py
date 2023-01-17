# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d, Axes3D
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from itertools import product

import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load datasets
df_data = pd.read_csv('../input/data.csv')
#df_genre = pd.read_csv('../input/data_2genre.csv')

#Joins datasets
#df_genre['label'] = df_genre['label'].apply(lambda x: 'pop' if x == 1 else 'classical')

#frames = [df_data, df_genre]
#df = pd.concat(frames)

df = df_data
df.head(3)
_ = df["label"].value_counts().plot.pie( autopct='%.2f', figsize=(10, 10),fontsize=20)
df.describe()
df.hist(bins=50, figsize=(20,15))
plt.show()
df.drop(["filename"], axis=1, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
for train_index, cross_index in split.split(df, df["label"]):
    strat_train_set = df.iloc[train_index]
    strat_cross_set = df.iloc[cross_index]
    
split_cross = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
for test_index, valid_index in split_cross.split(df, df["label"]):
    strat_test_set = df.iloc[test_index]
    strat_valid_set = df.iloc[valid_index]
music = strat_train_set.copy()
_ = music["label"].value_counts().plot.pie( autopct='%.2f', figsize=(6, 6))
attributes = ["beats", "tempo", "spectral_centroid",
              "spectral_bandwidth", "rolloff", "zero_crossing_rate" ]
sm = scatter_matrix(music[attributes], figsize=(20, 15), diagonal = "kde");

#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)];
[s.set_yticks(()) for s in sm.reshape(-1)];

for ax in sm.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 14)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 14)
music = strat_train_set.drop("label", axis=1)
music_labels = strat_train_set["label"].copy()
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: %0.2f" % scores.mean())
    print("Standard deviation: %0.2f" % (scores.std() * 2))
music = scale(music);
folds = 10
n_jobs=-1
verbose=5
clf = SVC(kernel='linear', C=0.01)
scores = cross_val_score(clf, music, music_labels, cv=folds, n_jobs=n_jobs, verbose=verbose)
display_scores(scores)
clf = SVC(kernel="poly", degree=5, coef0=1, C=5)
scores = cross_val_score(clf, music, music_labels, cv=folds, n_jobs=n_jobs, verbose=verbose)
display_scores(scores)
clf = SVC(kernel="rbf", gamma=5, C=1)
scores = cross_val_score(clf, music, music_labels, cv=folds, n_jobs=n_jobs, verbose=verbose)
display_scores(scores)
clf = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(clf, music, music_labels, cv=folds, n_jobs=n_jobs, verbose=verbose)
display_scores(scores)
music_valid = strat_valid_set.drop("label", axis=1)
music_valid_labels = strat_valid_set["label"].copy()

music_valid = scale(music_valid);

param_grid = [{'kernel':["poly"], 'degree': [1,2,3,4,5],'gamma': [0.01, 0.1, 0.5], 'coef0': [0, 0.1, 1], 'C': [0.001, 0.01, 1, 5]}]
svm_poly = SVC()
grid_search = GridSearchCV(svm_poly, param_grid, cv=folds, n_jobs=n_jobs)
grid_search.fit(music_valid, music_valid_labels)
grid_search.best_params_
def compute_ratio(matrix, score):
    np.set_printoptions(precision=2)

    FP = matrix[0,1]
    FN = matrix[1,0]
    TP = matrix[1,1]
    TN = matrix[0,0]

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    # Sensitivity or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
        
    print("Accuracy:   %0.3f" % score)    
    print("Specificity or true negative rate (AVG):   %0.3f" % TNR)
    print("Sensitivity or true positive rate (AVG):   %0.3f" % TPR)
X_test = strat_test_set.drop("label", axis=1)
y_test = strat_test_set["label"].copy()
X_test_prepared = scale(X_test)
clf = grid_search.best_estimator_

y_expected = music_valid_labels
y_predic = clf.predict(music_valid)
    
# Computes the accuracy (the fraction) of correct predictions.
score = metrics.accuracy_score(y_expected, y_predic)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_expected, y_predic)
compute_ratio(cnf_matrix, score)
y_expected = music_labels
y_predic = clf.predict(music)
    
# Computes the accuracy (the fraction) of correct predictions.
score = metrics.accuracy_score(y_expected, y_predic)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_expected, y_predic)
compute_ratio(cnf_matrix, score)
clf = grid_search.best_estimator_

y_expected = y_test
y_predic = clf.predict(X_test_prepared)
    
# Computes the accuracy (the fraction) of correct predictions.
score = metrics.accuracy_score(y_expected, y_predic)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_expected, y_predic)
compute_ratio(cnf_matrix, score)
df_pca= df
df_pca['label']=pd.Categorical(df_pca['label'])
my_color=df['label'].cat.codes
df_pca = df_pca.drop('label', 1)

plt.figure(1);
# Get current size
fig_size = plt.rcParams["figure.figsize"]

# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
y = my_color
X = df.drop('label', 1)


#In general a good idea is to scale the data
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)    

pca = PCA()
x_new = pca.fit_transform(X)

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

#Call the function. Use only the 2 PCs.
myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()
print(np.array2string(pca.explained_variance_ratio_, formatter={'float_kind':lambda x: "%.2f" % (x*100)}))
