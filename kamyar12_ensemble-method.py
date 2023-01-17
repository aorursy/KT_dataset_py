# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
warnings.filterwarnings("ignore")

train = pd.read_csv("../input/train.csv")
X = train.values[:,2:21]
y = train.values[:,21]
y = y.astype('int')
test = pd.read_csv("../input/test.csv")
test_X = test.values[:,2:21]
inds = np.where(pd.isnull(X))
col_mean = np.nanmean(X, axis=0)
X[inds] = np.take(col_mean, inds[1])
ran = RandomForestClassifier(max_depth=2, n_estimators = 100)
ran.fit(X, y)
print("RFC accuracy:" , ran.score(X, y))
knn =  KNeighborsClassifier(3, weights='uniform', algorithm = 'kd_tree', leaf_size = 10)
knn.fit(X, y)
print("Knn accuracy:" , knn.score(X, y))
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(13, 5), random_state=1)
mlp.fit(X, y)
print("Mlp accuracy:" , mlp.score(X, y))
svm = SVC(C=1, gamma='auto', kernel='rbf', probability=True)
svm.fit(X, y)
print("Svm accuracy:" , svm.score(X, y))
tree =  DecisionTreeClassifier(max_depth=10)
tree.fit(X, y)
print("Tree accuracy:" , tree.score(X, y))
ensembleVoting = VotingClassifier(estimators=[('ran', ran), ('knn', knn), ('mlp', mlp), ('svm', svm), ('tree', tree)], voting='soft', weights=[1, 1.2, 1, 1.5, 1.3], flatten_transform=True)
ensembleVoting = ensembleVoting.fit(X, y)
print("ensembleVoting accuracy:" , ensembleVoting.score(X, y))
print(ensembleVoting.predict(test_X))

cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [ensembleVoting.predict([test_X[i]])[0] for i in range(440)] }
submission = pd.DataFrame(cols)
print(submission)
submission.to_csv("submission.csv", index=False)

