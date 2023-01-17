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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
print (X.shape)
print (y.shape)
test = pd.read_csv("../input/test.csv")
test_X = test.values[:,2:21]
print (test_X.shape)
# replacing nan values with column mean :
inds = np.where(pd.isnull(X))
col_mean = np.nanmean(X, axis=0)
X[inds] = np.take(col_mean, inds[1])
ran = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 70)
ran.fit(X, y)
print("RandomForestClassifier:" , ran.score(X, y))
ada = AdaBoostClassifier(n_estimators=10)
ada.fit(preprocessing.scale(preprocessing.normalize(X)), y)
print("AdaBoostClassifier:" , ada.score(X, y))
grad = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth = 1, random_state = 0)
grad.fit(preprocessing.scale(preprocessing.normalize(X)), y)
print("GradientBoostingClassifier:" , grad.score(X, y))
svc1 = SVC(C=1, cache_size=400, class_weight=None, coef0=0.01,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=False,
    tol=0.0128, verbose=False)
svc1.fit(X, y)
print("svm1:" , svc1.score(X, y))
svc2 = SVC(kernel="linear", C=1)
svc2.fit(X, y)
print("svc2:" , svc2.score(X, y))
svc3 = SVC(gamma=4, C=1, probability=True)
svc3.fit(X, y)
print("svc3:" , svc3.score(X, y))
svc4 = LinearSVC(C=1)
svc4.fit(X, y)
print("svc4:" , svc4.score(X, y))
mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(14, 5), random_state=1)
mlp1.fit(X, y)
print("mlp1:" , mlp1.score(X, y))
knn =  KNeighborsClassifier(2, weights='uniform', algorithm = 'kd_tree', leaf_size = 40)
knn.fit(X, y)
print("knn:" , knn.score(X, y))
nc = NearestCentroid(metric='euclidean', shrink_threshold=None)
nc.fit(X, y)
print("NearestCentroid:" , nc.score(X, y))
tree =  DecisionTreeClassifier(max_depth=6)
tree.fit(X, y)
print("tree:" , tree.score(X, y))
Quad =  QuadraticDiscriminantAnalysis()
Quad.fit(X, y)
print("QuadraticDiscriminantAnalysis:" , Quad.score(X, y))
MNB =  MultinomialNB(alpha=0.2, fit_prior=True, class_prior=None)
MNB.fit(X, y)
print("MultinomialNBAnalysis:" , MNB.score(X, y))
# SGDClassifier =  SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
#        eta0=0.1, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=None, n_iter=None,
#        n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True, tol=None, verbose=0, warm_start=False)
# SGDClassifier.fit(X, y)
# print("SGDClassifier:" , SGDClassifier.score(X, y))
lg = LogisticRegression(random_state=20)
lg.fit(X, y)
print("lg:" , lg.score(X, y))
eclf1 = VotingClassifier(estimators=[('tree', tree), ('knn', knn), ('mlp1', mlp1), ('ran', ran), ('lg', lg)], 
                         voting='soft', weights=[1.1, 1, 1, 0.9, 1.3], flatten_transform=True)
eclf1 = eclf1.fit(X, y)
print("eclf1:" , eclf1.score(X, y))
print(eclf1.predict(test_X))

eclf2 = VotingClassifier(estimators=[('svc3', svc3), ('eclf1', eclf1)], voting='soft', weights=[2, 5], flatten_transform=True)
eclf2 = eclf2.fit(X, y)
print("eclf2:" , eclf2.score(X, y))
from sklearn.decomposition import PCA
pca = PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
pca.fit(X)
cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [eclf1.predict([test_X[i]])[0] for i in range(440)] }
submission = pd.DataFrame(cols)
print(submission)
submission.to_csv("submission.csv", index=False)
