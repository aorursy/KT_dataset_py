# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from scipy.cluster.hierarchy import dendrogram, linkage

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/faults.csv")

df.describe().T
df["Fault"] = 0

df.loc[df.Pastry == 1, ["Fault"]] = 1

df.loc[df.Z_Scratch == 1, ["Fault"]] = 2

df.loc[df.K_Scatch == 1, ["Fault"]] = 3

df.loc[df.Stains == 1, ["Fault"]] = 4

df.loc[df.Dirtiness == 1, ["Fault"]] = 5

df.loc[df.Bumps == 1, ["Fault"]] = 6

df.loc[df.Other_Faults == 1, ["Fault"]] = 7
f, ax = plt.subplots(figsize=(20,15))

sns.heatmap(df.corr(), annot=True)
X = df.ix[:, 1:27]

y = df["Fault"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train_array = X_train.values

y_train_array = y_train.values





def Run(X_train_array, y_train_array, model):

    pipe = Pipeline([('sc', StandardScaler()),

                ('model', model)])

    

    kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=0)

    scores = []

    for k, (train, test) in enumerate(kfold):

        pipe.fit(X_train_array[train], y_train_array[train])

        score = pipe.score(X_train_array[test], y_train_array[test])

        scores.append(score)

        print('Fold: %s, Class dist.: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train_array[train]), (score * 100)))

    

    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores) * 100, np.std(scores) *100))



    scores = cross_val_score(estimator=pipe, X=X_train_array, y=y_train_array, cv=10, n_jobs=-1)

    print('Cross Validation Accuracy scores: %s' % scores)

    print('Cross Validation Accuracy: %.3f +/- %.3f' % (np.mean(scores)*100, np.std(scores)*100))

Run(X_train_array, y_train_array, SVC(kernel='rbf', C=3, gamma=0.05, random_state=0))
Run(X_train_array, y_train_array, MLPClassifier(hidden_layer_sizes = (27, 27, 27), solver='adam', max_iter=500, random_state=0))
Run(X_train_array, y_train_array, LogisticRegression(C=10, random_state=0, penalty ='l2'))
Run(X_train_array, y_train_array, RandomForestClassifier(random_state=0, min_samples_split=2, n_estimators=50))
Z = linkage(X, 'ward')

plt.figure(figsize=(25, 25))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('sample index')

plt.ylabel('distance')

dendrogram(

    Z,

    truncate_mode='lastp',  

    p=12,

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,

)

plt.show()